#!/usr/bin/env python3
"""Step 2: free filters on the census, before any paid classification.

    python3 filter.py

Reads data/ai_social_science_census.csv and writes:
    data/candidates.csv       papers that pass, for abstracts.py and classify.py
    data/filtered_out.csv     papers removed, with the reason (spot-check these)

The filters are deliberately conservative: anything doubtful passes through to the
classifier, which applies the full definition. Each rule only removes clear misses.
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path

DATA = Path("data")
SRC = DATA / "ai_social_science_census.csv"

csv.field_size_limit(sys.maxsize)

SOCIAL_FIELDS = {
    "Sociology", "Economics", "Psychology", "Political Science", "Business", "Law",
    "Education", "Philosophy", "History", "Geography",
}

# Not papers: front matter, notices, corrections.
JUNK_TITLE = re.compile(
    r"^\s*(front matter|back matter|editorial board|table of contents|contents|index|"
    r"erratum|corrigendum|correction to|retraction|retracted|withdrawn|editor'?s note|"
    r"call for papers|book review|reviewer acknowledgement|cover image|issue information)\b",
    re.I,
)

# AI used as a research tool: a prediction or detection task with a named technique...
TOOL_TASK = re.compile(r"\b(predict\w*|forecast\w*|detect\w*|classif\w*|estimat\w*|recogni\w*|segment\w*)\b", re.I)
TOOL_TECH = re.compile(
    r"\b(machine learning|deep learning|neural network\w*|random forest\w*|lstm|cnn|xgboost|"
    r"support vector|svm|bert|transformer model\w*|gradient boost\w*|decision tree\w*|"
    r"artificial neural)\b",
    re.I,
)
# ...unless the title also points at people, institutions, or effects.
SOCIETAL = re.compile(
    r"\b(societ\w*|social|impact\w*|effect\w*|labou?r|work\w*|employ\w*|job\w*|trust\w*|bias\w*|"
    r"fair\w*|ethic\w*|govern\w*|polic\w*|public|attitude\w*|percep\w*|adopt\w*|human\w*|user\w*|"
    r"student\w*|worker\w*|risk\w*|harm\w*|inequal\w*|persua\w*|misinformation|disinformation|"
    r"democra\w*|institution\w*|organi[sz]ation\w*|depend\w*|relian\w*|assistant\w*|chatbot\w*|chatgpt)\b",
    re.I,
)


def reason_to_drop(row):
    title = (row.get("title") or "").strip()
    if not title or len(title.split()) < 3:
        return "no usable title"
    if JUNK_TITLE.search(title):
        return "front matter or notice"
    year = row.get("year")
    if not year or not year.isdigit() or not 2015 <= int(year) <= 2026:
        return "year missing or out of range"
    fields = {f.strip() for f in (row.get("fieldsOfStudy", "") + ";" + row.get("s2FieldsOfStudy", "")).split(";")}
    if not fields & SOCIAL_FIELDS:
        return "no social science field"
    if TOOL_TASK.search(title) and TOOL_TECH.search(title) and not SOCIETAL.search(title):
        return "AI used as a research tool"
    return None


def main():
    reasons = Counter()
    kept = 0
    with open(SRC, newline="", encoding="utf-8") as f, \
            open(DATA / "candidates.csv", "w", newline="", encoding="utf-8") as keep, \
            open(DATA / "filtered_out.csv", "w", newline="", encoding="utf-8") as drop:
        reader = csv.DictReader(f)
        kw = csv.DictWriter(keep, fieldnames=reader.fieldnames)
        dw = csv.DictWriter(drop, fieldnames=["reason"] + reader.fieldnames)
        kw.writeheader()
        dw.writeheader()
        for row in reader:
            why = reason_to_drop(row)
            if why:
                reasons[why] += 1
                dw.writerow({"reason": why, **row})
            else:
                kept += 1
                kw.writerow(row)
    total = kept + sum(reasons.values())
    print(f"Census papers: {total:,}")
    for why, n in reasons.most_common():
        print(f"  removed, {why}: {n:,}")
    print(f"Candidates for classification: {kept:,} ({kept / total:.0%})")


if __name__ == "__main__":
    main()
