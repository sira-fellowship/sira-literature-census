#!/usr/bin/env python3
"""All papers in the same social science fields per year, as the denominator for the census.

    python3 baseline.py

Writes data/baseline.json ({year: total}). Uses the same field filter as census.py, with no
keyword query, so the census counts can be read as a share of all social science output.
"""

import json
import os
from pathlib import Path

from census import SEARCH_URL, SOCIAL_FIELDS, Client, log


def main():
    client = Client(os.environ.get("SEMANTIC_SCHOLAR_API_KEY"))
    out = {}
    for year in range(2015, 2027):
        data = client.request("GET", SEARCH_URL, params={"year": str(year), "fields": "title",
                                                         "fieldsOfStudy": SOCIAL_FIELDS})
        out[year] = data.get("total", 0)
        log(f"{year}: {out[year]:,}")
    Path("data/baseline.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
