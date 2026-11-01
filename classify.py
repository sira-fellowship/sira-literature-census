#!/usr/bin/env python3
"""Step 4: classify candidates with Claude against the societal-impacts definition.

Run in this order:

    python3 classify.py label-sheet            # free: 200 random papers for you to hand-label
    python3 classify.py trial --model haiku    # cents: Claude labels the same 200, projects full cost
    python3 classify.py trial --model sonnet   # optional, under $1: a stronger model for comparison
    python3 classify.py validate               # free: agreement between your labels and Claude's
    python3 classify.py submit                 # paid (about $10): every candidate, via the Batches API
    python3 classify.py collect                # free: waits for the batches, writes data/classified.csv

Needs ANTHROPIC_API_KEY (see README.md) and `pip install anthropic`.
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path


def sdk():
    """The Anthropic SDK, imported only by the steps that call the API (label-sheet and validate run without it)."""
    try:
        import anthropic
    except ImportError:
        sys.exit("This step needs the Anthropic SDK: pip install anthropic")
    return anthropic


def client():
    """API client. Uses ANTHROPIC_API_KEY if set, otherwise the key in .anthropic_key (git-ignored)."""
    anthropic = sdk()
    key_file = Path(".anthropic_key")
    if not os.environ.get("ANTHROPIC_API_KEY") and key_file.exists():
        return anthropic.Anthropic(api_key=key_file.read_text().strip())
    return anthropic.Anthropic()

DATA = Path("data")
csv.field_size_limit(sys.maxsize)

MODELS = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}
# Dollars per million tokens (input, output), Anthropic list prices as of October 2026.
PRICE = {"haiku": (0.10, 0.50), "sonnet": (2.00, 10.00)}
BATCH_DISCOUNT = 0.5
BATCH_CHUNK = 20_000          # requests per batch; keeps each batch well under the 256 MB limit
ABSTRACT_CHARS = 3000         # long abstracts are cut here; the opening carries the design

THREAT_MODELS = [
    "human_ai_interaction",          # trust, over-reliance, companionship, anthropomorphism
    "persuasion_manipulation",       # persuasion, disinformation, deception, influence
    "economic_labor",                # jobs, wages, productivity, market concentration
    "institutional_organizational",  # organizations, institutions, adoption, inequality, public opinion
    "governance_policy",             # regulation, governance, accountability
    "other_societal",                # societal but none of the above
    "not_applicable",                # paper does not meet the definition
]
METHODS = [
    "experiment",                    # randomized or controlled experiment, incl. survey experiments
    "survey",                        # questionnaire or interview-based survey data
    "observational_data",            # usage logs, administrative or market data, field data
    "qualitative",                   # interviews, ethnography, case studies, content analysis
    "simulation_or_modeling",        # formal models, agent-based or LLM simulations
    "conceptual",                    # theory, argument, framework, commentary
    "review",                        # literature review, meta-analysis, survey of the field
    "other",
]

SYSTEM = """You classify academic papers for a census of social science research on the societal impacts of AI.

A paper MEETS the definition if it studies how AI affects people, organizations, institutions, economies, or politics, or how people and institutions perceive, use, or govern AI. "AI" includes machine learning systems, large language models, chatbots, generative AI, algorithmic decision systems, and AI agents.

A paper does NOT meet the definition if:
- it only uses AI as a research tool for a question that is not about AI (for example, machine learning to predict housing prices or to classify tweets about elections);
- it is a technical paper about building or improving an AI system, with no study of effects on people or institutions;
- it is a how-to guide or practitioner tip sheet (for example, "ten ways to use ChatGPT in your classroom");
- it is not a research paper (front matter, editorial board, conference notice).

Edge cases: a study of whether ChatGPT changes student learning MEETS the definition; a guide to teaching with ChatGPT does NOT. A study of how firms adopt AI MEETS it; a model that uses AI to forecast firm revenue does NOT. A normative or conceptual paper about AI's societal effects MEETS it. AI must be a central subject: a paper about digitalization or technology in general that mentions AI only in passing does NOT meet it. Judge papers in any language the same way.

For papers that meet the definition, choose the closest threat model:
- human_ai_interaction: trust, over-reliance, automation bias, companionship, anthropomorphism, emotional or cognitive effects on users
- persuasion_manipulation: persuasion, disinformation, deception, influence operations, deepfakes
- economic_labor: labor markets, jobs, wages, productivity, market concentration, economic inequality
- institutional_organizational: organizations, institutions, adoption across groups, public opinion, education systems, social inequality, epistemics
- governance_policy: regulation, governance, accountability, policy design
- other_societal: societal but none of the above
For papers that do not meet the definition, use not_applicable.

Method describes the paper's main research design. Judge from the title and abstract only; if the abstract is missing, judge from the title and lower your confidence. Keep the reason under 20 words."""

SCHEMA = {
    "type": "object",
    "properties": {
        "meets_definition": {"type": "boolean"},
        "threat_model": {"type": "string", "enum": THREAT_MODELS},
        "method": {"type": "string", "enum": METHODS},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "reason": {"type": "string"},
    },
    "required": ["meets_definition", "threat_model", "method", "confidence", "reason"],
    "additionalProperties": False,
}
OUT_COLUMNS = ["paperId", "meets_definition", "threat_model", "method", "confidence", "reason",
               "input_tokens", "output_tokens", "model", "status"]


def load_candidates():
    with open(DATA / "candidates.csv", newline="", encoding="utf-8") as f:
        papers = {r["paperId"]: r for r in csv.DictReader(f)}
    abstracts = {}
    if (DATA / "abstracts.csv").exists():
        with open(DATA / "abstracts.csv", newline="", encoding="utf-8") as f:
            abstracts = {r["paperId"]: r["abstract"] for r in csv.DictReader(f)}
    for pid, p in papers.items():
        p["abstract"] = abstracts.get(pid, "")
    return papers


def user_text(p):
    body = (p.get("abstract") or "").strip()
    label = "Abstract"
    if not body and p.get("tldr"):
        body, label = p["tldr"].strip(), "Summary (abstract unavailable)"
    if not body:
        body, label = "(none available)", "Abstract"
    if len(body) > ABSTRACT_CHARS:
        body = body[:ABSTRACT_CHARS] + " ..."
    return f"Title: {p['title']}\nYear: {p['year']}\nVenue: {p.get('venue') or 'unknown'}\n{label}: {body}"


def params(model_key, p):
    """Request body for one paper. Haiku runs with thinking off at low effort; this is a short
    labeling task. Claude Sonnet 5.5 rejects disabled thinking, so it runs adaptive at low effort."""
    body = dict(
        model=MODELS[model_key],
        max_tokens=400 if model_key == "haiku" else 2000,
        system=SYSTEM,
        messages=[{"role": "user", "content": user_text(p)}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
    )
    if model_key == "haiku":
        body["thinking"] = {"type": "disabled"}
    return body


def parse(message):
    if message.stop_reason == "refusal":
        return {"status": "refused"}
    text = next((b.text for b in message.content if b.type == "text"), "")
    try:
        out = json.loads(text)
    except json.JSONDecodeError:
        return {"status": "unparseable"}
    out["status"] = "ok"
    out["meets_definition"] = "yes" if out.get("meets_definition") else "no"
    return out


def cost(model_key, tokens_in, tokens_out, batch=False):
    pin, pout = PRICE[model_key]
    factor = BATCH_DISCOUNT if batch else 1
    return (tokens_in * pin + tokens_out * pout) / 1e6 * factor


def clock(start):
    secs = int(time.time() - start)
    return f"{secs // 3600}:{secs // 60 % 60:02d}:{secs % 60:02d}"


def progress(api, state):
    """One status line across all batches: elapsed since the first submission, percent done, rough time left."""
    total = done = 0
    for b in state["batches"]:
        if b["status"] == "collected":
            total += b["count"]
            done += b["count"]
            continue
        rc = api.messages.batches.retrieve(b["id"]).request_counts
        total += b["count"]
        done += rc.succeeded + rc.errored + rc.canceled + rc.expired
    start = min(b.get("submitted_at", time.time()) for b in state["batches"])
    elapsed = time.time() - start
    line = f"[{clock(start)} elapsed] {done:,} of {total:,} papers done ({done / total:.0%})"
    if 0 < done < total:
        left = int(elapsed / done * (total - done))
        line += f", roughly {left // 3600}h {left // 60 % 60:02d}m left"
    return line + "; checking again in 60s"


# --- subcommands -------------------------------------------------------------

def cmd_label_sheet(args):
    papers = load_candidates()
    path = DATA / "hand_labels.csv"
    if path.exists() and not args.force:
        sys.exit(f"{path} already exists (it may hold your labels). Use --force to overwrite.")
    rng = random.Random(args.seed)
    sample = rng.sample(sorted(papers), args.n)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["paperId", "your_meets_definition", "your_threat_model", "notes",
                    "title", "year", "venue", "abstract", "url"])
        for pid in sample:
            p = papers[pid]
            w.writerow([pid, "", "", "", p["title"], p["year"], p.get("venue", ""),
                        p.get("abstract") or p.get("tldr") or "", p.get("url", "")])
    print(f"Wrote {args.n} random papers to {path}.")
    print("Fill in your_meets_definition (yes/no) for each; your_threat_model is optional, using:")
    print("  " + ", ".join(t for t in THREAT_MODELS if t != "not_applicable"))


def cmd_trial(args):
    papers = load_candidates()
    with open(DATA / "hand_labels.csv", newline="", encoding="utf-8") as f:
        ids = [r["paperId"] for r in csv.DictReader(f)]
    api = client()
    out_path = DATA / f"trial_{args.model}.csv"
    tin = tout = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        w.writeheader()
        for n, pid in enumerate(ids, 1):
            body = params(args.model, papers[pid])
            if args.model == "sonnet":
                # Server-side fallback reruns a refused request on another model (not available on Batches).
                msg = api.beta.messages.create(**body, betas=["server-side-fallback-2026-07-01"], fallbacks="default")
            else:
                msg = api.messages.create(**body)
            row = parse(msg)
            row.update(paperId=pid, model=msg.model, input_tokens=msg.usage.input_tokens,
                       output_tokens=msg.usage.output_tokens)
            w.writerow({k: row.get(k, "") for k in OUT_COLUMNS})
            tin += msg.usage.input_tokens
            tout += msg.usage.output_tokens
            if n % 25 == 0:
                print(f"  {n}/{len(ids)}")
    n = len(ids)
    print(f"Wrote {out_path}. This trial cost ${cost(args.model, tin, tout):.2f} "
          f"({tin / n:.0f} input and {tout / n:.0f} output tokens per paper).")
    total = len(papers)
    est = cost(args.model, tin / n * total, tout / n * total, batch=True)
    print(f"Projected cost for all {total:,} candidates with {args.model} on the Batches API: ${est:,.2f}")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def cmd_validate(args):
    with open(DATA / "hand_labels.csv", newline="", encoding="utf-8") as f:
        human = {r["paperId"]: r for r in csv.DictReader(f)
                 if r["your_meets_definition"].strip().lower() in ("yes", "no")}
    if not human:
        sys.exit("No hand labels yet: fill in your_meets_definition in data/hand_labels.csv.")
    with open(DATA / "candidates.csv", newline="", encoding="utf-8") as f:
        total = sum(1 for _ in csv.DictReader(f))
    k = sum(1 for r in human.values() if r["your_meets_definition"].strip().lower() == "yes")
    lo, hi = wilson(k, len(human))
    print(f"Your labels: {len(human)} papers, {k} meet the definition ({k / len(human):.0%}, "
          f"95% CI {lo:.0%} to {hi:.0%}).")
    print(f"Implied candidates meeting the definition: about {k / len(human) * total:,.0f} "
          f"({lo * total:,.0f} to {hi * total:,.0f}).\n")
    for model_key in MODELS:
        path = DATA / f"trial_{model_key}.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            model = {r["paperId"]: r for r in csv.DictReader(f) if r["status"] == "ok"}
        both = [pid for pid in human if pid in model]
        c = Counter((human[p]["your_meets_definition"].strip().lower(), model[p]["meets_definition"]) for p in both)
        tp, fp, fn, tn = c["yes", "yes"], c["no", "yes"], c["yes", "no"], c["no", "no"]
        n = len(both)
        acc = (tp + tn) / n
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
        kappa = (acc - pe) / (1 - pe) if pe < 1 else 1.0
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"{model_key}: {n} papers compared")
        print(f"  agreement {acc:.0%}, Cohen's kappa {kappa:.2f}")
        print(f"  precision {prec:.0%} (of papers Claude says meet it, share you agree do)")
        print(f"  recall {rec:.0%} (of papers you say meet it, share Claude caught)")
        tm = [p for p in both if human[p]["your_threat_model"].strip() and model[p]["meets_definition"] == "yes"]
        if tm:
            same = sum(human[p]["your_threat_model"].strip() == model[p]["threat_model"] for p in tm)
            print(f"  threat model agreement {same}/{len(tm)}")
        full = DATA / "classified.csv"
        if model_key == "haiku" and full.exists() and tp:
            with open(full, newline="", encoding="utf-8") as f:
                n_yes = sum(r["meets_definition"] == "yes" for r in csv.DictReader(f) if r["status"] == "ok")
            # Claude's yes count, scaled by how often its yeses are right and how many true papers it finds.
            print(f"  full census: Claude says {n_yes:,} meet it; corrected by precision/recall, about {n_yes * prec / rec:,.0f}")
        wrong = [p for p in both if human[p]["your_meets_definition"].strip().lower() != model[p]["meets_definition"]]
        for p in wrong[:args.show]:
            print(f"    disagree: you={human[p]['your_meets_definition']} {model_key}={model[p]['meets_definition']} "
                  f"| {human[p]['title'][:90]} | {model[p]['reason']}")
        print()


def cmd_submit(args):
    papers = load_candidates()
    done = set()
    if (DATA / "classified.csv").exists():
        with open(DATA / "classified.csv", newline="", encoding="utf-8") as f:
            # Refused papers are not resent (a rerun would likely refuse again); errored/expired ones are.
            done = {r["paperId"] for r in csv.DictReader(f) if r["status"] in ("ok", "refused")}
    todo = [pid for pid in sorted(papers) if pid not in done]
    state_path = DATA / "batches.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"batches": []}
    if any(b["status"] != "collected" for b in state["batches"]) and not args.force:
        sys.exit("Earlier batches are still uncollected; run `collect` first (or --force).")
    if not args.yes:
        print(f"About to submit {len(todo):,} papers to {MODELS['haiku']} on the Batches API.")
        print("Run `trial --model haiku` first for a cost projection; add --yes to submit.")
        return
    api = client()
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request
    start = time.time()
    for i in range(0, len(todo), BATCH_CHUNK):
        chunk = todo[i:i + BATCH_CHUNK]
        print(f"[{clock(start)}] uploading papers {i + 1:,}-{i + len(chunk):,} of {len(todo):,}")
        batch = api.messages.batches.create(requests=[
            Request(custom_id=pid, params=MessageCreateParamsNonStreaming(**params("haiku", papers[pid])))
            for pid in chunk
        ])
        state["batches"].append({"id": batch.id, "count": len(chunk), "status": "submitted",
                                 "submitted_at": time.time()})
        state_path.write_text(json.dumps(state, indent=1))
        print(f"[{clock(start)}] submitted {batch.id} ({len(chunk):,} papers)")
    print(f"[{clock(start)}] all submitted. Most batches finish within an hour (24 hours at most).")
    print("Next: python3 classify.py collect   (waits with a progress timer, then writes data/classified.csv)")


def cmd_collect(args):
    state_path = DATA / "batches.json"
    if not state_path.exists():
        sys.exit("Nothing submitted yet: run `python3 classify.py submit --yes` first.")
    state = json.loads(state_path.read_text())
    api = client()
    out_path = DATA / "classified.csv"
    new = not out_path.exists()
    tin = tout = 0
    statuses = Counter()
    with open(out_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLUMNS)
        if new:
            w.writeheader()
        for b in state["batches"]:
            if b["status"] == "collected":
                continue
            while True:
                batch = api.messages.batches.retrieve(b["id"])
                if batch.processing_status == "ended":
                    break
                print(progress(api, state))
                time.sleep(60)
            for result in api.messages.batches.results(b["id"]):
                row = {"paperId": result.custom_id, "model": MODELS["haiku"]}
                if result.result.type == "succeeded":
                    msg = result.result.message
                    row.update(parse(msg), input_tokens=msg.usage.input_tokens,
                               output_tokens=msg.usage.output_tokens)
                    tin += msg.usage.input_tokens
                    tout += msg.usage.output_tokens
                else:
                    row["status"] = result.result.type  # errored / canceled / expired: rerun submit
                statuses[row["status"]] += 1
                w.writerow({k: row.get(k, "") for k in OUT_COLUMNS})
            b["status"] = "collected"
            state_path.write_text(json.dumps(state, indent=1))
    print(f"Wrote {out_path}: " + ", ".join(f"{k} {v:,}" for k, v in statuses.items()))
    print(f"Cost of these batches: ${cost('haiku', tin, tout, batch=True):,.2f}")
    if any(s != "ok" for s in statuses):
        print("Papers without an ok status are resubmitted by running `submit --yes` again.")


def main():
    sys.stdout.reconfigure(line_buffering=True)  # show progress lines immediately, even through `tee`
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("label-sheet")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--seed", type=int, default=2026)
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("trial")
    s.add_argument("--model", choices=list(MODELS), default="haiku")
    s = sub.add_parser("validate")
    s.add_argument("--show", type=int, default=10, help="disagreements to print per model")
    s = sub.add_parser("submit")
    s.add_argument("--yes", action="store_true", help="actually submit (costs money)")
    s.add_argument("--force", action="store_true")
    sub.add_parser("collect")
    args = ap.parse_args()
    {"label-sheet": cmd_label_sheet, "trial": cmd_trial, "validate": cmd_validate,
     "submit": cmd_submit, "collect": cmd_collect}[args.cmd](args)


if __name__ == "__main__":
    main()
