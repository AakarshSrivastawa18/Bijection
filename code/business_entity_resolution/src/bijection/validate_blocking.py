"""Measure blocking quality on a held-out slice of the training data.

Usage:
    python -m bijection.validate_blocking --country US --n 20000 --top-k 20

Reports pair completeness (the recall ceiling every downstream stage inherits),
average candidates per Source-1 entity, reduction ratio, and the best macro-F0.5
any matcher could reach given these candidates.
"""
from __future__ import annotations

import argparse
import random
import time
from typing import Dict, List, Set, Tuple

from . import config
from .candidates import build_shard, generate
from .io_utils import load_ground_truth, read_records
from .metrics import score_blocking
from .mine_rules import RuleSet


def load_country_slice(country: str, n_s1: int, seed: int = config.RANDOM_SEED):
    """Sample n_s1 Source-1 entities of one country, plus the FULL pool for it.

    The pool is never subsampled: shrinking it would inflate recall and understate
    the candidate counts, which is exactly the number under review.
    """
    rng = random.Random(seed)
    gt = load_ground_truth(config.TRAIN["gt"])

    s1_all = [(e, n, a) for e, n, a, c in read_records(config.TRAIN["s1"], {country})]
    if n_s1 and n_s1 < len(s1_all):
        s1_all = rng.sample(s1_all, n_s1)
    keep: Set[str] = {e for e, _, _ in s1_all}
    sub_gt: Dict[str, List[str]] = {e: gt.get(e, []) for e in keep}

    pool: List[Tuple[str, str, str]] = []
    for key in ("s2", "s3"):
        for e, n, a, c in read_records(config.TRAIN[key], {country}):
            pool.append((e, n, a))
    return s1_all, pool, sub_gt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="US")
    ap.add_argument("--n", type=int, default=20000, help="Source-1 entities to sample")
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--max-posting", type=int, default=config.MAX_POSTING)
    ap.add_argument("--cache", action="store_true", help="cache/reuse the built shard")
    ap.add_argument("--rerank-pool", type=int, default=200,
                    help="crude-score shortlist size fed to the overlap reranker")
    args = ap.parse_args()

    rules = RuleSet.load()

    # Building a full-pool shard costs ~400s, which makes scorer iteration painful.
    # Cache it so ranking experiments are seconds, not minutes.
    import pickle
    cache = config.WORK_DIR / f"shard_{args.country}_{args.n}_{args.max_posting}.pkl"
    if args.cache and cache.exists():
        t0 = time.time()
        with open(cache, "rb") as fh:
            shard, gt, pool_n = pickle.load(fh)
        print(f"[cache] loaded {cache.name} ({time.time()-t0:.1f}s)")
    else:
        t0 = time.time()
        s1, pool, gt = load_country_slice(args.country, args.n)
        pool_n = len(pool)
        print(f"[load]  S1={len(s1):,}  pool={pool_n:,}  ({time.time()-t0:.1f}s)")
        t0 = time.time()
        shard = build_shard(args.country, s1, pool, rules, max_posting=args.max_posting)
        print(f"[index] vocab={len(shard.vocab):,}  postings={shard.index.keys.size:,}  "
              f"({time.time()-t0:.1f}s)")
        del pool
        if args.cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "wb") as fh:
                pickle.dump((shard, gt, pool_n), fh, protocol=5)
            print(f"[cache] wrote {cache.name}")


    # Probe once with no cap, then evaluate every K off the same ranked lists. This
    # separates the two independent failure modes: keys that never retrieved a true
    # match at all (raw recall < 1) versus true matches retrieved but ranked below
    # the cut (raw recall high, recall@K low). They have opposite fixes.
    import statistics

    from .candidates import Prober
    prober = Prober(shard, rules)
    pool_ids = shard.pool.ids
    rp = args.rerank_pool

    # Uncapped crude pass: the recall ceiling the KEYS alone can deliver.
    t0 = time.time()
    raw: Dict[str, List[str]] = {}
    raw_counts: List[int] = []
    for i, eid in enumerate(shard.s1.ids):
        scored = prober.candidates(i, 10 ** 9, rerank_pool=0)
        raw_counts.append(len(scored))
        raw[eid] = [pool_ids[j] for j, _ in scored]
    print(f"[raw]   {time.time()-t0:.1f}s  candidates/entity: "
          f"mean={statistics.mean(raw_counts):.1f} "
          f"median={statistics.median(raw_counts):.0f} max={max(raw_counts)}")

    # Reranked pass: the ordering the pipeline actually emits.
    t0 = time.time()
    ranked: Dict[str, List[str]] = {}
    for i, eid in enumerate(shard.s1.ids):
        scored = prober.candidates(i, rp, rerank_pool=rp)
        ranked[eid] = [pool_ids[j] for j, _ in scored]
    dt = time.time() - t0
    print(f"[rerank] {len(ranked):,} entities in {dt:.1f}s "
          f"({len(ranked)/max(dt,1e-9):,.0f} entities/s)")

    b_raw = score_blocking(raw, gt, pool_size=pool_n)
    print(f"\n=== BLOCKING  country={args.country}  (rerank_pool={rp}) ===")
    print(f"  {'KEYS-ONLY':11s} recall={b_raw.pair_completeness:.4f}  "
          f"avg_cand={b_raw.avg_candidates:7.2f}   <- ceiling the keys allow")
    for k in (5, 10, 15, 20, 25, 30, 50):
        cut = {e: v[:k] for e, v in ranked.items()}
        b = score_blocking(cut, gt, pool_size=pool_n)
        print(f"  {'K=%d' % k:11s} recall={b.pair_completeness:.4f}  "
              f"avg_cand={b.avg_candidates:7.2f}  "
              f"ceiling_F0.5={b.ceiling_f05:.4f}  rr={b.reduction_ratio:.2e}")


if __name__ == "__main__":  # pragma: no cover
    main()
