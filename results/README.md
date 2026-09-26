# Measured results

Raw logs from the runs that produced the numbers quoted in `Documentation_template.md`.
Kept so the claims can be checked rather than taken on trust.

## `blocking_India.log`

Blocking quality on India, 5,000 held-out Source-1 entities against the **full**
4,133,346-record pool (the pool is never subsampled — shrinking it would inflate recall
and understate candidate counts).

Reports the recall/candidate-count curve from K=5 to K=50, plus the uncapped
"KEYS-ONLY" ceiling that no top-K choice can exceed.

## `dataset_build_and_blocking.log`

The US and India training-matrix builds. Contains the blocking scorecard for each
country at the shipped K=20:

| Country | Recall@20 | Avg cand/entity | Reduction ratio | Ceiling macro-F0.5 |
|---|---|---|---|---|
| US | 0.95297 | 19.72 | 3.19e-06 | 0.98544 |
| India | 0.92431 | 19.71 | 4.77e-06 | 0.97411 |

## `throughput.log`

Measured inference throughput on the test set. Recorded because the first ETA was
wrong: France ran at ~1,050 entities/sec and India at 222/sec. That is not a
regression — India's pool is 3.3x larger (4.72M vs 1.43M records), so every probe walks
proportionally longer posting lists, and 1050/3.3 ≈ 318 brackets the observed rate.

## Not reproduced here

The model's own validation numbers (US 0.95926, India 0.94561, LOCO 0.886–0.910) come
from `python -m bijection.train --loco`, which prints them to stdout. Re-run it against
the committed `work/model.txt` to reproduce.
