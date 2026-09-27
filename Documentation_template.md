# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Bijection
**Team Members:** Aakarsh Srivastava
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

A four-stage pipeline — mined normalization → country-partitioned conjunctive blocking →
LightGBM over string/overlap/**listwise** features → F0.5-tuned decision layer — reaching
**macro-F0.5 0.963 (US) and 0.950 (India)** on held-out validation while emitting only
**24.6 candidates per Source-1 entity**, a ~4 × 10⁻⁶ reduction ratio.

Two things distinguish it. First, **every normalization table is mined from the training
ground truth rather than hardcoded** — US state codes, street abbreviations, and a
1,222-entry transliteration table covering nine Indic scripts — which satisfies the
no-external-data rule and is also why the pipeline transfers to France, a country with
zero training rows. Second, **listwise features dominate the model**: a candidate's score
z-scored within its own Source-1 group carries 54.7% of total model gain, more than every
string-similarity feature combined.

---

## 2. Methodology

### 2.1 Problem Analysis

All figures below are measured from the provided data, not assumed. Where a claim rests on
a sample rather than a full scan, that is stated.

**Scale.** Test is 1,732,544 Source-1 entities against 9,969,589 Source-2/3 records —
**1.73 × 10¹³ brute-force pairs**, roughly 200 CPU-days at 1 µs/pair.

**Two structural properties (both established by exhaustive scan, not sampling):**

| Property | Evidence |
|---|---|
| Country is a **hard partition** | All 7,638,365 GT matched pairs checked: **0 cross a country boundary**, 0 missing |
| The match relation is a **bijection onto its matched subset** | All 7,638,365 matched IDs exploded and counted: every Source-2/3 record is claimed by **exactly one** Source-1 entity; zero reuse |

The first is worth a free 3× reduction and lets the pipeline shard by country with no
recall loss. The second is genuinely true but, as §5 reports honestly, turned out to be
worth almost nothing once the model was well calibrated.

**Match-count distribution** (full scan of 2,206,821 GT rows): mean 3.46 matches, max 11,
with per-entity caps of 5 Source-2 and 6 Source-3. Only **5.58% are singletons**, so
"predict nothing" scores 0.056 — the tip about singletons is a trap if over-read.

**Blocking signal ceilings** (146,209 true pairs sampled from GT):

| Shared between a Source-1 entity and its true match | Recall ceiling |
|---|---|
| Any address word | 95.17% |
| Any name token (legal suffixes stripped) | 84.97% |
| Any address number | 79.93% |
| Union of all three | **99.99%** |

**Noise patterns observed in real matched clusters:**

* Source-2 is UPPERCASE and abbreviated (`ST`/`AVE`/`RD`), 2-letter state; Source-3 is title
  case with full state names (`Texas`, `New York`) and frequently reorders components.
* Homoglyph substitution: `Jania 5herman`, `pvt. pr0ud hotels ltd.`, `Vinayak (lndia)`.
* Injected diacritics on US names (`BÍNGAMAN`), legitimate accents in France (`Àmicale`).
* Legal-suffix drift, DBA prefixes (`Zephveo doing business as Pacific League`), `M/s`,
  appended junk (`Services #45303`), word transposition (`MD, Jania Sherman,`).
* **4.55% of true matches have a completely empty address** — any address-mandatory rule
  caps recall at 95.5%.

**Two findings that corrected earlier assumptions:**

1. **Nine Indic scripts, not one.** Source-2/3 India records appear in Devanagari, Telugu,
   Kannada, Tamil, Bengali, Gujarati, Malayalam, Oriya and Gurmukhi. An initial pass that
   looked only for Devanagari understated the transliteration problem by roughly 8×
   (153 mined entries vs 1,222 once all scripts were handled).
2. **14,079 invisible U+200C characters** (zero-width non-joiner) in Source-2/3 fields.
   Legitimate Indic orthography, but it silently breaks token equality, so it is stripped.

**France** (15% of test entities, zero training rows) follows the same generative template:
`SARL/SAS/SASU/EURL/SCI` suffixes, `R.`/`AV` abbreviations, injected accents, ~3% empty
addresses in S2/S3 versus 0% in S1. One caution stated plainly: with no France ground
truth, *no* France matching behaviour is directly verifiable by anyone. Region↔department
variation (`Lille` co-occurring with both `Hauts-de-France` and `Nord`) is established from
label co-occurrence in the test data itself, not from outside geography knowledge.

### 2.2 Solution Strategy

**Approach Type:** Blocking + gradient-boosted pairwise classifier + constrained assignment

**Core Innovation:** Every substitution table is *learned from aligned ground-truth pairs*
rather than authored. The same procedure runs unchanged on an unseen country, which is what
makes France tractable. Paired with this, the matcher is made *listwise* — each candidate is
scored in the context of its competitors — which turned out to matter far more than any
string metric.

```
   test_source{1,2,3}.tsv
            |
   [0] country partition ......... verified: 0 of 7,638,365 GT pairs cross a country
            |
   [1] normalize ................. unicode fold, homoglyph skeleton, mined rule tables
            |
   [2] block ..................... conjunctive inverted index -> top-25 per entity
            |                      >>> candidate_pairs.tsv written here <<<
   [3] score ..................... LightGBM over 36 features
            |
   [4] decide .................... F0.5-tuned threshold + one-to-one assignment
            |
   matching_results.tsv
```

---

## 3. Candidate Generation (Blocking)

**Blocking keys used.** Country is a hard partition. Within a shard, every key is a
*conjunction* of two signals, which keeps posting lists at single/double digits — a single
shared token (`pacific`, `bordeaux`) has a posting list in the tens of thousands and is
useless to walk.

| Family | Key | Weight |
|---|---|---|
| `K_NUM_ADDR` | house number + address word | 3.5 |
| `K_NAME_ADDR` | name token + address word | 3.0 |
| `K_NAME_NUM` | name token + house number | 3.0 |
| `K_NAME_PAIR` | two name tokens | 2.0 |
| `K_EXACT_NAME` | whole normalized name (rescues the 4.55% with empty addresses) | 4.0 |

Tokens entering a key are the **top-3 by IDF** on each side. Candidates are ranked by
accumulated key weight × IDF of the matched tokens. Posting lists longer than 400 are
dropped at build time. Mined alternates are expanded **at probe time only** — they are
symmetric, so probing `{st, street, saint}` against a raw-token index catches the match in
either direction at half the index size.

**Candidate pairs generated:** 25 per Source-1 entity (24.6 mean). Over the full test set
that is ~42.7M pairs against a 1.73 × 10¹³ brute-force space.

**LSH typo family (added after leaderboard round 1):** 4 bands × 2 rows of MinHash over
char-3grams of the whole skeletonised name. This is the only key family that can retrieve
"Bsigaman" for "Bingaman" — every other family needs an exact shared token. Collision
probability ≈ 0.83 at 3gram-Jaccard 0.6 (a one-typo name), near zero for unrelated names.

**Measured blocking quality** (full-pool evaluation, ground truth held out):

| Country | Recall@20 | Avg cand/entity | Reduction ratio | Ceiling macro-F0.5 |
|---|---|---|---|---|
| US | **0.95541** | 24.63 | 3.98 × 10⁻⁶ | 0.98627 |
| India | **0.92841** | 24.65 | 5.96 × 10⁻⁶ | 0.97563 |

(First leaderboard round ran K=20 without the LSH family: US 0.95297 / India 0.92431.
The loss decomposition after that round showed blocking misses were 4.7% / 7.6% of true
matches — a hard cap no model change can recover — which motivated both additions.)

**How true matches were kept:** the key union was chosen against the measured 99.99%
ceiling in §2.1; the skeleton normalization recovers homoglyph noise that token-exact keys
miss; the whole-name key rescues empty-address records; IDF-weighted ranking (rather than
raw key counts) recovered ~0.5 points of recall@20.

**A negative result, reported because it shaped the design.** A second-stage reranker on the
shortlist was built and then removed. Measured on 5,000 US entities against the full
6.19M-record pool:

| Ranking formula | Recall@20 |
|---|---|
| **Crude IDF key score (shipped)** | **0.9542** |
| + character similarity | 0.9546 – 0.9553 |
| + coverage blend | 0.9540 – 0.9545 |
| Dice | 0.9415 |
| Cosine-normalised overlap | 0.9413 |
| Asymmetric coverage | 0.9387 |
| Coverage + length penalty | 0.9080 – 0.9298 |
| Character similarity alone | 0.8398 |

Nothing beat the plain score by more than noise (~±0.15% on 17,300 pairs), and the
normalized variants were *actively worse* — they penalise candidates carrying extra tokens,
but the generator **adds** tokens to true matches. The reranker cost ~40% of probe time for
no gain and was deleted. Token-level ordering signal is saturated; discrimination beyond it
is the matcher's job.

**Why K=20 and not 50.** Uncapped key recall is 97.6% (US) / 96.8% (India), so K=50 would buy
roughly +0.8 points. But the ceiling at K=20 is already macro-F0.5 0.985/0.974, far above
what the matcher achieves (0.959/0.946) — blocking is not the binding constraint. Since
candidate-set size is explicitly ranked, spending 2.5× more candidates for headroom that
goes unused is the wrong trade.

---

## 4. Matching Model

**Features used** (36 total, all lexical/structural — **no feature encodes country**, since
France would make any such feature undefined):

* **Name string:** `ratio`, `token_sort_ratio`, `token_set_ratio`, `partial_ratio`,
  Jaro-Winkler, OSA (Levenshtein + adjacent transposition — added in round 2 for the
  transposition-typo noise class none of the others measures directly; also on address). Each fails differently — `token_sort` survives transposition, `partial`
  survives DBA prefixes, plain `ratio` catches character noise — so the tree picks.
* **Address string:** `ratio`, `token_sort`, `token_set`, `partial`.
* **Structured overlap:** IDF-weighted name/address intersection mass, asymmetric coverage
  in both directions, Jaccard, shared-number count and Jaccard.
* **Shape/missingness:** empty-address flag, name lengths, token counts, `is_s2`.
* **Listwise (within the Source-1 candidate group):** blocking score, rank,
  score-ratio-to-top, score-gap-to-top, **score z-score within group**, group size.

**Model type:** LightGBM (MIT), binary objective, 160 leaves, lr 0.05, early-stopped at 2056 rounds (validation logloss was still falling at the round-1 cap of 700 — the model was under-trained).
Chosen over a neural cross-encoder because the test set requires ~34.6M inferences on
CPU-only 8-core/16 GB hardware, and the noise is lexical rather than semantic.

**Feature importance (% of total gain):**

| Feature | Gain % |
|---|---|
| `score_z` (listwise) | **54.68** |
| `block_score` (listwise) | 6.49 |
| `num_jaccard` | 4.80 |
| `rank` (listwise) | 3.78 |
| `addr_token_set` | 2.92 |
| `name_token_sort` | 2.92 |
| `name_partial` | 2.89 |
| `name_cover_cand` | 2.84 |

The listwise block contributes ~67% of total gain. This is the single most valuable design
decision in the matcher: knowing a candidate is "the 7th best of 20, 2.1 standard deviations
below the leader" is worth more than any pairwise string comparison.

**Threshold selection:** macro-F0.5 sweep on a pooled cross-country validation slice, not
accuracy or AUC. The optimum is **0.68** — well above 0.5, exactly as the metric's asymmetry
predicts. For a 3-match entity, one false positive costs 0.211 while one false negative costs
0.091, a 2.3× penalty.

**Training data:** the model trains on the blocker's own output rather than random negatives,
so the negatives it learns to reject are the near-misses it will actually face. 2.07M training
rows, 16.5% positive.

---

## 5. Results & Error Analysis

**F_0.5 Score (macro), held-out validation, threshold 0.70:**

| Slice | macro-F0.5 | Precision | Recall | Exact sets | % of blocking ceiling |
|---|---|---|---|---|---|
| US held-out | **0.96274** | 0.9819 | 0.9216 | 73.0% | 97.6% |
| India held-out | **0.95039** | 0.9737 | 0.9037 | 69.1% | 97.4% |
| Pooled | **0.95715** | 0.9782 | 0.9135 | — | — |

(Leaderboard round 1 — K=20, 34 features, 96 leaves × 700 rounds — scored **0.945** with
pooled validation 0.95343; the validation-to-leaderboard gap was ~0.008. Round 2 adds the
LSH blocking family, K=25, two OSA character-distance features, 160 leaves × 2056 rounds
(early-stopped), threshold 0.68.)

**Leave-one-country-out — the France proxy.** France has zero training rows, so the in-domain
numbers above would flatter it. Training on one country and testing on the other:

| | macro-F0.5 |
|---|---|
| train[India] → test[US] | 0.91009 |
| train[US] → test[India] | 0.88648 |

France should be expected in the **0.89–0.91** band, not 0.95. Weighting the three countries
by their test entity counts (US 38.3%, India 46.7%, France 15.0%) gives an expected test score
around **0.94**.

**A decision-layer idea the sweep rejected.** A "rescue" rule (give an empty entity its
single best sub-threshold candidate) was added in round 2 and tuned on validation — and the
sweep chose to DISABLE it (no-rescue 0.95715 vs best rescue 0.95673). With the deeper,
better-calibrated model, sub-threshold candidates are genuinely below the precision bar.
The machinery ships (tests included) but the tuned configuration turns it off.

**An ablation that did not go as predicted.** The one-to-one assignment step — the property
the team is named after — contributes **+0.00008 (US) and +0.00005 (India)**. The structural
property is real and exhaustively verified; the inference that exploiting it would move the
score was wrong. It only bites when two entities claim the same record above threshold, and at
precision 0.977 that is rare — a well-tuned threshold already does the work. The stage is kept
because it costs nothing measurable and can only ever remove a false positive, never add one.

**Common false positives (wrong merges):** distinct businesses sharing a street number and
city whose names differ only in a low-IDF token — the `K_NUM_ADDR` key fires strongly and the
name evidence is too thin to override it. Chains and franchises at nearby addresses are the
worst case.

**Common false negatives (missed matches):** dominated by blocking, not by the model — the
matcher already extracts ~96% of what the candidate set contains. The residual misses are
(a) heavy character corruption that skeletonization cannot fold (`Bsigaman`/`Bingaman`,
`Rti`/`Reit` are 2-edit and break token equality), (b) records with an empty address *and* a
corrupted name, leaving no usable key, and (c) India addresses reduced to a landmark or to
city+state only, where the house number — the strongest key — is absent. India's lower score
traces directly to (c) plus transliteration.

**Verification:** 71 automated tests, including a synthetic end-to-end run that asserts every
rule the official validator enforces, and regression tests for four real defects found during
development (a `\w+` tokenizer that shredded Devanagari into characters; city names leaking
into the droppable-suffix table; word-order loss that broke acronym mining; `dr↔ltd` junk from
unfiltered co-occurrence).

---

## 6. Conclusion

Mining normalization rules from the ground truth instead of hardcoding gazetteers satisfies
the fair-play constraint and is the mechanism that carries the pipeline to an unseen country;
making the matcher listwise contributed ~65% of model gain. The most useful discipline was
measuring every intuition rather than shipping it — two ideas that seemed obviously right (a
second-stage overlap reranker, and exploiting the one-to-one constraint) were built, measured
at no better than neutral, and are reported as such here. The remaining headroom is in
blocking recall, not in the matcher: at 97% of the candidate-set ceiling, a better model has
almost nothing left to find.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` — `src/bijection/` (15 modules), `tests/` (71 tests),
`README.md`, `requirements.txt`.

Reproduce end-to-end:

```bash
export BIJECTION_DATA_DIR=/path/to/dataset
export PYTHONPATH=src
python -m bijection.mine_rules                                   # mine tables    (~60s)
python -m bijection.make_dataset --country US    --n 80000       # build matrix   (~11min)
python -m bijection.make_dataset --country India --n 60000       #                (~8min)
python -m bijection.train --train work/ds_US_80000.npz work/ds_India_60000.npz --loco
python -m bijection.predict --split test                         # -> output/*.tsv
```

Measurement tools (not on the inference path): `validate_blocking.py` (recall/reduction
curve), `tune_blocking.py` (ranking-formula comparison — reproduces the §3 table).

### B. Additional Results

**Threshold sweep, pooled validation** — the precision-heavy optimum at 0.70:

| Threshold | macro-F0.5 | Precision | Recall |
|---|---|---|---|
| 0.30 | 0.93954 | 0.9523 | 0.9294 |
| 0.50 | 0.95106 | 0.9693 | 0.9186 |
| 0.65 | 0.95340 | 0.9758 | 0.9077 |
| **0.70** | **0.95343** | 0.9775 | 0.9031 |
| 0.80 | 0.95105 | 0.9794 | 0.8895 |
| 0.95 | 0.93059 | 0.9753 | 0.8356 |

**Mined table sizes:** 89 droppable name tokens, 267 token expansions, 14 multi-word
initialisms, 1,222 transliteration entries across nine scripts.
