# SIRA literature census

A census of social science research on the societal impacts of AI, 2015 to October 2026. It
backs the growth figures on [SIRA's evidence page](https://sira-project.org/evidence/).

The census searches [Semantic Scholar](https://www.semanticscholar.org/) for papers in social
science fields that pair AI terms with societal-impact terms, then has Claude classify each paper
against a written definition. A random sample was hand-labeled to measure how well the
classifier agrees with a person.

## Results (November 2026)

| | |
|---|---|
| Candidate papers classified | 138,938 |
| Papers meeting the definition | 104,300 (about 101,200 adjusted for measured precision and recall) |
| Papers per year, 2019 to 2025 | 1,837 to 31,151 (17×) |
| Share of all social science papers | about 1 in 1,300 (2019) to 1 in 50 (2025) |
| Empirical share, 2019 to 2025 | 19% to 43% |
| Agreement with hand labels | 85% on 183 papers (Cohen's kappa 0.61; precision 88%, recall 91%) |

By threat model:

| Threat model | Papers | Growth, 2019 to 2025 | Empirical | Experiments |
|---|---:|---:|---:|---:|
| Institutional and organizational | 48,914 | 27× | 46% | 8% |
| Governance and policy | 18,817 | 10× | 9% | 2% |
| Human-AI interaction | 17,128 | 24× | 69% | 30% |
| Economic and labor | 14,136 | 10× | 35% | 4% |
| Other societal | 3,008 | 7× | 16% | 9% |
| Persuasion and manipulation | 2,289 | 19× | 47% | 30% |

Summary files are in [`results/`](results/): `by_year.csv`, `by_threat_model.csv`, and
`analysis.json`. Threat models and research designs are Claude's labels and have not yet been
checked against hand labels; the yes/no decision has.

**Treat the counts as a lower bound.** Keyword search only reaches titles and abstracts, so it
misses papers indexed without an abstract. As a check, the census found about a third of the
academic papers in SIRA's hand-picked list of studies; most of the misses are working papers or
lack an abstract in Semantic Scholar.

## Definition

A paper meets the definition if it studies how AI affects people, organizations, institutions,
economies, or politics, or how people and institutions perceive, use, or govern AI. It does not
if it only uses AI as a research tool, is a technical paper about building AI systems, is a
how-to guide, or is not a research paper. AI must be a central subject. The full instructions
Claude receives are `SYSTEM` in [`classify.py`](classify.py).

## Pipeline

Requires Python 3.10+ and `pip install -r requirements.txt`.

| Step | Script | Cost | What it does |
|---|---|---|---|
| 1 | `census.py` | free | Semantic Scholar bulk search: one query per category and year, 2015 to 2026, in social science fields. Resumable. |
| 2 | `filter.py` | free | Drops front matter, retractions, and titles where AI is only a prediction tool. |
| 3 | `abstracts.py` | free | Fetches abstracts, 500 papers per request. |
| 4 | `classify.py label-sheet` | free | 200 random papers for hand labeling. |
| 5 | `classify.py trial` | cents | Claude labels the same 200 and projects the full cost. |
| 6 | `classify.py validate` | free | Agreement with the hand labels and an estimated total. |
| 7 | `classify.py submit --yes`, `collect` | about $15 | Classifies every candidate with Claude Haiku 5.5 on the Message Batches API. |
| 8 | `baseline.py` | free | All papers in the same fields per year, for the share-of-field figure. |
| 9 | `analyze.py` | free | Writes `data/analysis.json` and the summaries in `results/`. |

```zsh
python3 census.py --dry-run            # match counts per category and year
caffeinate -i python3 census.py --start-year 2015
python3 filter.py
python3 abstracts.py
python3 classify.py label-sheet        # then fill in data/hand_labels.csv
python3 classify.py trial --model haiku
python3 classify.py validate
python3 classify.py submit --yes && python3 classify.py collect
python3 baseline.py
python3 analyze.py
```

The website reads `data/analysis.json` through `tools/census_data.py` in the
[site repository](https://github.com/sira-fellowship/sira-fellowship.github.io), which writes
`evidence/census.js`.

### Keys

Never commit keys. The Semantic Scholar key is read from `SEMANTIC_SCHOLAR_API_KEY`:

```zsh
read -s "key?Paste your Semantic Scholar API key: "; echo
echo "export SEMANTIC_SCHOLAR_API_KEY='$key'" >> ~/.zshrc; unset key; source ~/.zshrc
```

The Anthropic key is read from `ANTHROPIC_API_KEY`, or from a git-ignored `.anthropic_key` file:

```zsh
read -s "key?Paste your Anthropic API key: "; echo
printf '%s' "$key" > .anthropic_key; unset key; chmod 600 .anthropic_key
```

## How the search works

Each query is (AI terms) AND (one category's terms), run once per year, with Semantic Scholar's
field filter set to Sociology, Economics, Psychology, Political Science, Business, Law,
Education, and Philosophy. Terms live in `AI_TERMS` and `CATEGORIES` in `census.py`.

Things learned the hard way:

- Semantic Scholar ignores `*` inside quotes: `"persuasi*"` matches only the literal text. Use
  unquoted wildcards for single words and list plural phrases explicitly. The first run had this
  bug; the corrected search added about 5,000 papers, mostly in persuasion and manipulation.
- "machine learning" and "algorithmic" are left out of the AI terms: in social science fields
  they mostly match papers that use ML as a method.
- Requests are spaced 1.5 s apart and HTTP 429s are retried with backoff (up to about 40
  minutes). If a key is throttled, `SEMANTIC_SCHOLAR_API_KEY= python3 census.py ...` uses the
  shared public limit instead.
- Bulk search does not return TL;DRs; they come from the paper batch endpoint.
- Semantic Scholar's total for social science fields dips in 2021 and 2022, likely an indexing
  change, so read the share-of-field ratio loosely.

## Layout

- `data/` (not committed): the census, candidates, abstracts, classifications, hand labels, and
  checkpoints. Several of these files are over GitHub's size limit.
- `results/`: small summary files, safe to share.
