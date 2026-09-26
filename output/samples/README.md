# Output samples

**These are excerpts, not submittable files.**

The two real outputs are produced by `python -m bijection.predict --split test` and
written to `output/`. They are not committed:

| File | Full size | Why not in git |
|---|---|---|
| `candidate_pairs.tsv` | ~400 MB | Over GitHub's 100 MB per-file hard limit |
| `matching_results.tsv` | ~87 MB | Under the limit but large and fully regenerable |

Both regenerate deterministically from the committed model (`work/model.txt`), the
committed mined rule tables (`work/rules/`), and the committed threshold
(`work/threshold.txt`). Nothing about the result depends on anything excluded here.

## What is in this folder

* `matching_results.sample.tsv` — first 200 data rows of the final matches
* `candidate_pairs.sample.tsv` — first 50 data rows of the blocking output

Both samples are taken from the **France** block of the run. France is the country
with **zero training rows**, so these rows are the most informative ones to eyeball:
they show the pipeline generalising to an unseen country, unaided by any
country-specific rule.

## Reading the samples

Every row is `source1_entity_id <TAB> comma,separated,ids`. An empty second column is a
predicted singleton — worth a full 1.0 on the metric when correct, so it is a real
prediction, not a failure.

Each entity's matched ids are a subset of its candidate ids; a match that never appeared
as a candidate would signal a pipeline bug, and `tests/test_pipeline.py` asserts the
subset property end to end.

## Reproducing the full files

```bash
export BIJECTION_DATA_DIR=/path/to/dataset
export PYTHONPATH=code/business_entity_resolution/src
python -m bijection.predict --split test        # ~2h on 8 cores / 16 GB
```

Then validate before submitting:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
