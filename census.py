#!/usr/bin/env python3
"""Literature census: social science research on the societal impacts of AI, 2020 onward.

Collects papers from the Semantic Scholar Academic Graph API with a keyword matrix
(AI terms x social science terms, grouped into SIRA's threat models), one query per
category per year, and writes them incrementally to CSV.

    export SEMANTIC_SCHOLAR_API_KEY=...        # see README.md
    python3 census.py --dry-run                # one call per query-year: how many papers?
    python3 census.py                          # full run; safe to stop and restart

Outputs (in --out-dir, default ./data):
    ai_social_science_census.csv   one row per paper (deduplicated across queries)
    ai_social_science_hits.csv     one row per (paper, category, year) match, so overlap
                                   between categories is kept even though papers appear once
    checkpoint.json                progress; lets an interrupted run resume where it stopped

Uses only the requests library. The CSVs load directly with pandas.read_csv.
"""

import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import requests

API = "https://api.semanticscholar.org/graph/v1"
SEARCH_URL = f"{API}/paper/search/bulk"   # token-paginated, up to 1,000 papers per page
BATCH_URL = f"{API}/paper/batch"          # up to 500 ids per call; supports tldr

# Bulk search does not return tldr, so it is fetched separately through the batch endpoint.
SEARCH_FIELDS = "paperId,title,authors,year,citationCount,externalIds,fieldsOfStudy,s2FieldsOfStudy,venue,publicationDate"
BATCH_FIELDS = "tldr"
BATCH_SIZE = 500

MIN_INTERVAL = 1.5      # seconds between any two requests; 1.05 drew frequent 429s even with a key
MAX_RETRIES = 25        # with the 120s cap, about 40 minutes of waiting before giving up
TIMEOUT = 60

# Social science fields used to filter results (Semantic Scholar field names).
SOCIAL_FIELDS = "Sociology,Economics,Psychology,Political Science,Business,Law,Education,Philosophy"

# Keyword matrix. Every query is (AI terms) AND (one category's terms).
# Bulk search syntax: | is OR, + is AND, "..." is a phrase, * is a prefix wildcard.
# "machine learning" and "algorithmic" are left out: in social science fields they mostly match
# papers that use ML as a research method, not papers that study AI's effects.
# Semantic Scholar ignores * inside quotes ("persuasi*" matches only the literal text), so
# wildcards are unquoted single words and phrases list their plural forms explicitly.
# "machine learning" and "algorithmic" are left out: in social science fields they mostly match
# papers that use ML as a research method, not papers that study AI's effects.
AI_TERMS = [
    '"artificial intelligence"', '"large language model"', '"large language models"', '"generative AI"',
    "LLM*", "ChatGPT", '"GPT-4"', '"AI system"', '"AI systems"', '"AI agent"', '"AI agents"', "chatbot*",
]

CATEGORIES = {
    "tm1_human_ai_interaction": [
        '"trust"', '"overreliance"', '"over-reliance"', '"automation bias"', "anthropomorph*",
        "companion*", '"loneliness"', '"parasocial"', '"emotional attachment"', "sycophan*",
        '"human-AI interaction"', '"well-being"', '"dependence"',
    ],
    "tm2_persuasion_manipulation": [
        "persuasi*", "manipulat*", '"disinformation"', '"misinformation"', '"propaganda"',
        '"influence operation"', '"influence operations"', '"deception"', "deepfake*", '"microtargeting"',
        '"belief change"', "conspirac*",
    ],
    "tm3_economic_labor": [
        '"labor market"', '"labor markets"', '"labour market"', '"labour markets"', '"employment"', "wage*",
        '"productivity"', '"automation"', '"job displacement"', '"inequality"', '"market concentration"',
        '"economic impact"', '"economic impacts"', '"future of work"',
    ],
    "tm4_institutional_organizational": [
        '"institutional trust"', '"organizational"', '"organisational"', "institution*",
        '"adoption"', '"digital divide"', "sociolog*", '"social impact"', '"social impacts"',
        '"societal impact"', '"societal impacts"', '"public opinion"', '"epistemic"', '"power concentration"',
        '"institutional risk"',
    ],
    "governance_policy": [
        '"governance"', '"regulation"', '"public policy"', '"accountability"', '"AI safety"',
        '"AI risk"', '"AI risks"', "democra*",
    ],
}

CSV_COLUMNS = [
    "paperId", "title", "year", "publicationDate", "venue", "citationCount",
    "authors", "authorIds", "doi", "arxiv", "externalIds",
    "fieldsOfStudy", "s2FieldsOfStudy", "tldr",
    "first_category", "first_query_year", "url",
]
HIT_COLUMNS = ["paperId", "category", "query_year"]


def build_query(category):
    return "(" + " | ".join(AI_TERMS) + ") + (" + " | ".join(CATEGORIES[category]) + ")"


class Client:
    """Rate-limited Semantic Scholar client with exponential backoff."""

    def __init__(self, api_key):
        self.session = requests.Session()
        if api_key:
            self.session.headers["x-api-key"] = api_key
        self.session.headers["User-Agent"] = "sira-literature-census/1.0"
        self._last = 0.0

    def _wait(self):
        gap = time.monotonic() - self._last
        if gap < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - gap)
        self._last = time.monotonic()

    def request(self, method, url, **kw):
        for attempt in range(MAX_RETRIES):
            self._wait()
            try:
                r = self.session.request(method, url, timeout=TIMEOUT, **kw)
            except requests.RequestException as e:
                delay = min(2 ** attempt, 120) + random.random()
                log(f"  network error ({e.__class__.__name__}); retrying in {delay:.0f}s")
                time.sleep(delay)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:
                retry_after = r.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2 ** attempt, 120)
                delay += random.random()
                log(f"  HTTP {r.status_code}; backing off {delay:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(delay)
                continue
            if r.status_code in (401, 403):
                sys.exit(f"HTTP {r.status_code}: the API key was rejected. Check SEMANTIC_SCHOLAR_API_KEY.")
            sys.exit(f"HTTP {r.status_code} from {url}: {r.text[:500]}")
        sys.exit(f"Giving up after {MAX_RETRIES} attempts: {url}")

    def search(self, query, year, token=None, field_filter=True):
        params = {"query": query, "year": str(year), "fields": SEARCH_FIELDS}
        if field_filter:
            params["fieldsOfStudy"] = SOCIAL_FIELDS
        if token:
            params["token"] = token
        return self.request("GET", SEARCH_URL, params=params)

    def tldrs(self, ids):
        out = {}
        for i in range(0, len(ids), BATCH_SIZE):
            chunk = ids[i:i + BATCH_SIZE]
            rows = self.request("POST", BATCH_URL, params={"fields": BATCH_FIELDS}, json={"ids": chunk})
            for row in rows:
                if row and row.get("tldr"):
                    out[row["paperId"]] = row["tldr"].get("text") or ""
        return out


def flatten(p, category, year, tldr):
    ext = p.get("externalIds") or {}
    authors = p.get("authors") or []
    s2 = []
    for f in p.get("s2FieldsOfStudy") or []:
        if f.get("category") and f["category"] not in s2:
            s2.append(f["category"])
    return {
        "paperId": p.get("paperId"),
        "title": (p.get("title") or "").strip(),
        "year": p.get("year"),
        "publicationDate": p.get("publicationDate") or "",
        "venue": p.get("venue") or "",
        "citationCount": p.get("citationCount"),
        "authors": "; ".join(a.get("name") or "" for a in authors),
        "authorIds": "; ".join(a.get("authorId") or "" for a in authors),
        "doi": ext.get("DOI") or "",
        "arxiv": ext.get("ArXiv") or "",
        "externalIds": json.dumps(ext, ensure_ascii=False, sort_keys=True),
        "fieldsOfStudy": "; ".join(p.get("fieldsOfStudy") or []),
        "s2FieldsOfStudy": "; ".join(s2),
        "tldr": tldr or "",
        "first_category": category,
        "first_query_year": year,
        "url": f"https://www.semanticscholar.org/paper/{p.get('paperId')}",
    }


class Store:
    """Appends to the CSVs and keeps a checkpoint, flushing after every page."""

    def __init__(self, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.papers_path = out_dir / "ai_social_science_census.csv"
        self.hits_path = out_dir / "ai_social_science_hits.csv"
        self.ckpt_path = out_dir / "checkpoint.json"
        self.seen = set()
        self.hits_seen = set()
        if self.papers_path.exists():
            with open(self.papers_path, newline="", encoding="utf-8") as f:
                self.seen = {row["paperId"] for row in csv.DictReader(f)}
        if self.hits_path.exists():
            with open(self.hits_path, newline="", encoding="utf-8") as f:
                self.hits_seen = {(r["paperId"], r["category"], r["query_year"]) for r in csv.DictReader(f)}
        self.papers = self._open(self.papers_path, CSV_COLUMNS)
        self.hits = self._open(self.hits_path, HIT_COLUMNS)
        self.ckpt = json.loads(self.ckpt_path.read_text()) if self.ckpt_path.exists() else {"done": [], "tokens": {}}

    @staticmethod
    def _open(path, cols):
        new = not path.exists() or path.stat().st_size == 0
        f = open(path, "a", newline="", encoding="utf-8")
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        return f, w

    def is_done(self, key):
        return key in self.ckpt["done"]

    def token(self, key):
        return self.ckpt["tokens"].get(key)

    def write_page(self, rows, hits):
        f, w = self.papers
        for row in rows:
            if row["paperId"] not in self.seen:
                w.writerow(row)
                self.seen.add(row["paperId"])
        hf, hw = self.hits
        for h in hits:
            k = (h["paperId"], h["category"], str(h["query_year"]))
            if k not in self.hits_seen:
                hw.writerow(h)
                self.hits_seen.add(k)
        for fh in (f, hf):
            fh.flush()
            os.fsync(fh.fileno())

    def save(self, key, token=None, done=False):
        if done:
            self.ckpt["tokens"].pop(key, None)
            if key not in self.ckpt["done"]:
                self.ckpt["done"].append(key)
        else:
            self.ckpt["tokens"][key] = token
        tmp = self.ckpt_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.ckpt, indent=1))
        tmp.replace(self.ckpt_path)  # atomic: a crash never leaves a half-written checkpoint

    def close(self):
        self.papers[0].close()
        self.hits[0].close()


def log(msg):
    print(msg, flush=True)


def run(args):
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        log("Warning: SEMANTIC_SCHOLAR_API_KEY is not set. Running on the shared public rate limit.")
    client = Client(api_key)
    years = range(args.start_year, args.end_year + 1)
    cats = args.categories or list(CATEGORIES)

    if args.dry_run:
        total = 0
        log(f"{'category':34}" + "".join(f"{y:>9}" for y in years))
        for cat in cats:
            counts = [client.search(build_query(cat), y, field_filter=not args.no_field_filter).get("total", 0) for y in years]
            total += sum(counts)
            log(f"{cat:34}" + "".join(f"{c:>9,}" for c in counts))
        log(f"\nMatches across all query-years: {total:,} (before removing papers that match several categories).")
        return

    store = Store(Path(args.out_dir))
    try:
        for cat in cats:
            query = build_query(cat)
            for year in years:
                key = f"{cat}|{year}"
                if store.is_done(key):
                    continue
                token, pages = store.token(key), 0
                log(f"{cat} {year}" + (" (resuming)" if token else ""))
                while True:
                    data = client.search(query, year, token, field_filter=not args.no_field_filter)
                    papers = [p for p in data.get("data") or [] if p.get("paperId")]
                    if pages == 0 and not token:
                        log(f"  {data.get('total', 0):,} matches")
                    new_ids = [p["paperId"] for p in papers if p["paperId"] not in store.seen]
                    tldr = {} if args.no_tldr or not new_ids else client.tldrs(new_ids)
                    store.write_page(
                        [flatten(p, cat, year, tldr.get(p["paperId"])) for p in papers],
                        [{"paperId": p["paperId"], "category": cat, "query_year": year} for p in papers],
                    )
                    pages += 1
                    token = data.get("token")
                    log(f"  page {pages}: {len(papers)} papers, {len(new_ids)} new, {len(store.seen):,} total")
                    if not token or (args.max_pages and pages >= args.max_pages):
                        store.save(key, done=True)
                        break
                    store.save(key, token)
    except KeyboardInterrupt:
        log("\nStopped. Progress is saved; run the same command to resume.")
    finally:
        store.close()
    log(f"Papers in {store.papers_path}: {len(store.seen):,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--start-year", type=int, default=2020)
    ap.add_argument("--end-year", type=int, default=2026)
    ap.add_argument("--categories", nargs="+", choices=list(CATEGORIES), help="default: all")
    ap.add_argument("--max-pages", type=int, default=0, help="cap pages (of up to 1,000) per category-year; 0 = no cap")
    ap.add_argument("--no-field-filter", action="store_true", help="search all fields, not just the social sciences")
    ap.add_argument("--no-tldr", action="store_true", help="skip the TL;DR lookups (a third as many requests)")
    ap.add_argument("--dry-run", action="store_true", help="report match counts per category and year, then stop")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
