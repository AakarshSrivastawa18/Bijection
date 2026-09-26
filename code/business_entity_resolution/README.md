# Bijection — Business Entity Resolution

End-to-end pipeline for the ML Challenge 2026 business entity resolution task:
given noisy business records from three independent sources, find every Source-2 /
Source-3 record that refers to the same real-world business as each Source-1 entity.

The name refers to the structural property the solution exploits. Across all
7,638,365 matched pairs in the training ground truth, **every Source-2/3 record
belongs to exactly one Source-1 entity — zero are reused.** The match relation is a
bijection onto its matched subset, and enforcing that globally is the pipeline's
main precision lever.

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export BIJECTION_DATA_DIR=/path/to/dataset     # must contain train/ and test/
export PYTHONPATH=src

# 1. mine normalization tables from the training ground truth   (~60s)
python -m bijection.mine_rules

# 2. build labelled training matrices                           (~9 min each)
python -m bijection.make_dataset --country US    --n 80000
python -m bijection.make_dataset --country India --n 60000

# 3. train the matcher and tune the threshold for macro-F0.5    (~2 min)
python -m bijection.train --train work/ds_US_80000.npz --ood work/ds_India_60000.npz

# 4. run inference over the test set -> output/*.tsv
python -m bijection.predict --split test
```

Outputs land in `output/`:

* `candidate_pairs.tsv` — the blocking output, exactly the pairs the model scores
* `matching_results.tsv` — the final matches

Validate before submitting:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

## Reproducing the measurements

```bash
python -m bijection.validate_blocking --country US --n 5000     # recall / reduction curve
python -m bijection.tune_blocking     --country US --n 5000     # ranking-formula comparison
python -m pytest tests/ -q                                      # 53 tests
```

---

## Pipeline

```
   test_source{1,2,3}.tsv
            |
   [0] country partition ............ verified: 0 of 7,638,365 GT pairs cross a country
            |
   [1] normalize ................... unicode fold, homoglyph skeleton, mined rule tables
            |
   [2] block ....................... conjunctive inverted index -> top-20 per entity
            |                        >>> candidate_pairs.tsv is written here <<<
   [3] score ....................... LightGBM over 34 string/overlap/listwise features
            |
   [4] decide ...................... F0.5-tuned threshold + global one-to-one assignment
            |
   matching_results.tsv
```

### `src/bijection/`

| module | role |
|---|---|
| `config.py` | paths and tunables (`TOP_K`, caps, posting limits) |
| `io_utils.py` | streaming TSV readers/writers — nothing loads a whole source file |
| `metrics.py` | the exact competition metric + blocking diagnostics |
| `normalize.py` | Phase 1 text normalization (conservative + skeleton forms) |
| `mine_rules.py` | mines all substitution tables from the training ground truth |
| `blocking.py` | CSR record store, key generation, sorted inverted index |
| `candidates.py` | shard construction and probing |
| `features.py` | 34 pairwise features |
| `dataset.py` | batched (X, y) construction |
| `model.py` | LightGBM training / persistence |
| `decide.py` | threshold, one-to-one assignment, per-source caps |
| `predict.py` | end-to-end test inference |
| `validate_blocking.py`, `tune_blocking.py`, `make_dataset.py`, `train.py` | measurement + training drivers |

---

## Design notes

**No external data.** The challenge forbids external lookup, so there is no
hardcoded gazetteer anywhere. US state codes, street-type abbreviations, French
department names and Indic transliteration are all *mined* from aligned ground-truth
pairs by `mine_rules.py`. This is also why the pipeline generalizes to France, which
has zero training rows — the same procedure runs unchanged on an unseen country, and
per-country IDF suppresses its legal suffixes (SARL/SAS/EURL) without supervision.

**Expansion, not canonicalization.** The mined table contains `st → {street, saint}`
because both are real (St.-Herblain is a French city). Collapsing tokens to one
canonical form would corrupt addresses, so alternates are *expanded* at probe time
instead.

**Nine scripts, not one.** Source-2/3 India records appear in Devanagari, Telugu,
Kannada, Tamil, Bengali, Gujarati, Malayalam, Oriya and Gurmukhi. The transliteration
table covers all of them (1,222 entries). The data also contains 14,079 invisible
U+200C characters, which are stripped — they silently break token equality.

**Blocking is ranked separately, so it is measured separately.**
`validate_blocking.py` reports pair completeness, reduction ratio, and the ceiling
macro-F0.5 the candidates allow. `tune_blocking.py` exists because the first
intuition was wrong: a second-stage overlap reranker (cosine / dice / coverage, with
and without length penalties, plus character-level similarity) was tested against the
plain IDF-weighted key score and **none of them beat it** — every variant landed
within 0.001 of 0.9542 recall@20 and most were 1–2 points worse. That stage was
removed rather than shipped.

**Validation simulates the country shift.** The model trains on US and is validated
on India as well as on a held-out US slice. The test set's France records have no
training analogue, so the India number — not the in-domain US one — is the honest
generalization estimate.
