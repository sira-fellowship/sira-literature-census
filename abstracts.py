#!/usr/bin/env python3
"""Step 3: fetch abstracts for the candidates from Semantic Scholar (500 papers per request).

    python3 abstracts.py

Reads data/candidates.csv and appends to data/abstracts.csv (paperId, abstract).
Resumable: papers already fetched are skipped. Run it after census.py has finished,
since both share the same rate limit.
"""

import csv
import os
import sys
from pathlib import Path

from census import BATCH_SIZE, BATCH_URL, Client, log

DATA = Path("data")
OUT = DATA / "abstracts.csv"

csv.field_size_limit(sys.maxsize)


def main():
    with open(DATA / "candidates.csv", newline="", encoding="utf-8") as f:
        ids = [row["paperId"] for row in csv.DictReader(f)]
    done = set()
    if OUT.exists():
        with open(OUT, newline="", encoding="utf-8") as f:
            done = {row["paperId"] for row in csv.DictReader(f)}
    todo = [i for i in ids if i not in done]
    log(f"{len(ids):,} candidates; {len(done):,} already fetched; {len(todo):,} to go "
        f"({-(-len(todo) // BATCH_SIZE)} requests)")

    client = Client(os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))
    new = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["paperId", "abstract"])
        if new:
            w.writeheader()
        for i in range(0, len(todo), BATCH_SIZE):
            chunk = todo[i:i + BATCH_SIZE]
            rows = client.request("POST", BATCH_URL, params={"fields": "abstract"}, json={"ids": chunk})
            got = {r["paperId"]: (r.get("abstract") or "").strip() for r in rows if r}
            for pid in chunk:
                w.writerow({"paperId": pid, "abstract": got.get(pid, "")})
            f.flush()
            log(f"  {min(i + BATCH_SIZE, len(todo)):,} / {len(todo):,}")
    log(f"Done: {OUT}")


if __name__ == "__main__":
    main()
