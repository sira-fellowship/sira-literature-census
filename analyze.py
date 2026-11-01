#!/usr/bin/env python3
"""Step 5: summarize the classified census. Writes data/analysis.json and prints the key numbers.

    python3 analyze.py
"""

import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path("data")
csv.field_size_limit(sys.maxsize)
EMPIRICAL = {"experiment", "survey", "observational_data", "qualitative"}
YEARS = list(range(2015, 2027))


def load():
    with open(DATA / "classified.csv", newline="", encoding="utf-8") as f:
        labels = {r["paperId"]: r for r in csv.DictReader(f) if r["status"] == "ok"}
    with open(DATA / "candidates.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["paperId"] in labels:
                labels[r["paperId"]].update({k: r[k] for k in (
                    "title", "year", "publicationDate", "venue", "citationCount", "fieldsOfStudy",
                    "s2FieldsOfStudy", "doi", "url", "authors")})
    return list(labels.values())


def write_results(out):
    """Small, shareable summaries in results/ (the paper-level data in data/ stays local)."""
    res = Path("results")
    res.mkdir(exist_ok=True)
    (res / "analysis.json").write_text(json.dumps({k: v for k, v in out.items() if k != "most_cited"}, indent=1))
    with open(res / "by_year.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["year", "papers_meeting_definition", "empirical", "experiments", "empirical_share",
                    "all_social_science_papers", "one_in_n"])
        for y in YEARS:
            w.writerow([y, out["by_year"][y], out["empirical_by_year"][y], out["experiment_by_year"][y],
                        out["empirical_share_by_year"][y], out.get("baseline_by_year", {}).get(y, ""),
                        out.get("one_in_n_by_year", {}).get(y, "")])
    with open(res / "by_threat_model.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threat_model", "papers", "papers_2019", "papers_2025", "empirical", "experiments"])
        for tm, n in out["threat_model_total"].items():
            m = out["method_by_threat_model"][tm]
            w.writerow([tm, n, out["threat_model_by_year"][tm][2019], out["threat_model_by_year"][tm][2025],
                        sum(c for k, c in m.items() if k in EMPIRICAL), m.get("experiment", 0)])


def main():
    rows = load()
    yes = [r for r in rows if r["meets_definition"] == "yes"]
    out = {"classified": len(rows), "meets": len(yes)}
    nogov = [r for r in yes if r["threat_model"] != "governance_policy"]
    out["meets_excluding_governance"] = len(nogov)

    by_year = Counter(int(r["year"]) for r in yes)
    by_year_nogov = Counter(int(r["year"]) for r in nogov)
    out["by_year"] = {y: by_year[y] for y in YEARS}
    out["by_year_excluding_governance"] = {y: by_year_nogov[y] for y in YEARS}
    cand_year = Counter(int(r["year"]) for r in rows)
    out["share_meeting_by_year"] = {y: round(by_year[y] / cand_year[y], 3) for y in YEARS}

    # How many "2026" papers actually have an earlier publication date (online-first, issue-dated later)?
    pd = Counter()
    for r in yes:
        if r["year"] == "2026":
            d = r["publicationDate"][:4]
            pd[d or "no date"] += 1
    out["papers_2026_by_publication_date_year"] = dict(pd)

    tm_year = defaultdict(Counter)
    for r in yes:
        tm_year[r["threat_model"]][int(r["year"])] += 1
    out["threat_model_total"] = dict(Counter(r["threat_model"] for r in yes).most_common())
    out["threat_model_by_year"] = {tm: {y: c[y] for y in YEARS} for tm, c in tm_year.items()}

    out["method_total"] = dict(Counter(r["method"] for r in yes).most_common())
    emp = Counter(int(r["year"]) for r in yes if r["method"] in EMPIRICAL)
    exp = Counter(int(r["year"]) for r in yes if r["method"] == "experiment")
    out["empirical_share_by_year"] = {y: round(emp[y] / by_year[y], 3) if by_year[y] else None for y in YEARS}
    out["empirical_by_year"] = {y: emp[y] for y in YEARS}
    out["experiment_by_year"] = {y: exp[y] for y in YEARS}
    out["empirical_total"] = sum(emp.values())
    out["experiment_total"] = sum(exp.values())
    out["method_by_threat_model"] = {
        tm: dict(Counter(r["method"] for r in yes if r["threat_model"] == tm).most_common())
        for tm in out["threat_model_total"]}

    fields = Counter()
    for r in yes:
        fs = {f.strip() for f in (r["fieldsOfStudy"] + ";" + r["s2FieldsOfStudy"]).split(";") if f.strip()}
        fields.update(fs)
    out["fields_of_study"] = dict(fields.most_common(14))
    out["meets_excluding_education_only"] = sum(
        1 for r in yes
        if {f.strip() for f in (r["fieldsOfStudy"] + ";" + r["s2FieldsOfStudy"]).split(";") if f.strip()}
        - {"Education", "Computer Science"})

    out["confidence"] = dict(Counter(r["confidence"] for r in yes))

    top = {}
    for tm in out["threat_model_total"]:
        if tm == "not_applicable":
            continue
        ranked = sorted((r for r in yes if r["threat_model"] == tm),
                        key=lambda r: int(r["citationCount"] or 0), reverse=True)[:8]
        top[tm] = [{"title": r["title"], "year": r["year"], "citations": int(r["citationCount"] or 0),
                    "method": r["method"], "venue": r["venue"], "url": r["url"]} for r in ranked]
    out["most_cited"] = top

    # Share of all social science papers in the same fields (data/baseline.json, from baseline.py).
    if (DATA / "baseline.json").exists():
        base = json.loads((DATA / "baseline.json").read_text())
        out["baseline_by_year"] = {y: base[str(y)] for y in YEARS}
        out["one_in_n_by_year"] = {y: round(base[str(y)] / by_year[y]) for y in YEARS if by_year[y]}

    (DATA / "analysis.json").write_text(json.dumps(out, indent=1))
    write_results(out)
    for k, v in out.items():
        if k not in ("most_cited", "threat_model_by_year", "method_by_threat_model"):
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
