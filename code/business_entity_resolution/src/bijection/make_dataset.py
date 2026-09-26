"""Build and cache a labelled feature matrix for one country.

    python -m bijection.make_dataset --country US --n 80000

Shard construction costs ~400s regardless of how many Source-1 entities are
sampled (the pool is always the whole country), so it is worth sampling
generously in a single pass and splitting afterwards.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from . import config
from .candidates import build_shard
from .dataset import build_matrix
from .metrics import score_blocking
from .mine_rules import RuleSet
from .validate_blocking import load_country_slice


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="US")
    ap.add_argument("--n", type=int, default=80000)
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rules = RuleSet.load()
    t0 = time.time()
    s1, pool, gt = load_country_slice(args.country, args.n)
    print(f"[load]  S1={len(s1):,}  pool={len(pool):,}  ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    shard = build_shard(args.country, s1, pool, rules)
    pool_n = len(pool)
    del pool, s1
    print(f"[index] vocab={len(shard.vocab):,}  postings={shard.index.keys.size:,}  "
          f"({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    X, y, row_s1, row_cand = build_matrix(shard, rules, args.top_k, gt)
    print(f"[feat]  X={X.shape}  positives={int(y.sum()):,} "
          f"({100*y.mean():.2f}%)  ({time.time()-t0:.1f}s)", flush=True)

    # Report the recall ceiling this matrix inherits - the model can never exceed it.
    cand = {}
    for s, c in zip(row_s1, row_cand):
        cand.setdefault(s, []).append(c)
    b = score_blocking(cand, gt, pool_size=pool_n)
    print(f"[block] {b}")

    out = args.out or (config.WORK_DIR / f"ds_{args.country}_{args.n}.npz")
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, X=X, y=y,
        row_s1=np.asarray(row_s1), row_cand=np.asarray(row_cand),
        s1_ids=np.asarray(shard.s1.ids), pool_n=pool_n,
        gt_keys=np.asarray(list(gt.keys())),
        gt_vals=np.asarray([",".join(v) for v in gt.values()]),
    )
    print(f"[save]  {out}")


if __name__ == "__main__":  # pragma: no cover
    main()
