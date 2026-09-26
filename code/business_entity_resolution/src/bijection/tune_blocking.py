"""Compare candidate-ranking formulas against a cached shard.

Ranking, not key design, turned out to be the binding constraint on blocking recall:
the keys retrieve 97.6% of true matches but a poor ordering leaves 3 points on the
floor at K=20. This script computes the crude key scores once, then scores the same
shortlist with several formulas so they can be compared in seconds.

    python -m bijection.tune_blocking --country US --n 5000
"""
from __future__ import annotations

import argparse
import heapq
import math
import pickle
import time
from typing import Callable, Dict, List, Tuple

import numpy as np

from . import config
from .candidates import Prober
from .metrics import score_blocking
from .mine_rules import RuleSet

SHORTLIST = 200


def _sets(store, i):
    return (set(store.names(i).tolist()),
            set(store.words(i).tolist()),
            set(store.nums(i).tolist()))


def build_variants() -> Dict[str, Callable]:
    """Each variant maps per-field (overlap, s1_mass, cand_mass) triples -> score."""

    def field(i, b, c, mode, pen):
        if b <= 0 or c <= 0 or i <= 0:
            return 0.0
        if mode == "cover":       # how much of the Source-1 side is accounted for
            v = i / b
        elif mode == "cosine":
            v = i / math.sqrt(b * c)
        elif mode == "dice":
            v = 2.0 * i / (b + c)
        else:
            raise ValueError(mode)
        if pen:
            v /= c ** pen
        return v

    def make(mode, pen, wn, ww, wd, crude_w):
        def f(parts, crude):
            (i_n, b_n, c_n), (i_w, b_w, c_w), (i_d, b_d, c_d) = parts
            return (wn * field(i_n, b_n, c_n, mode, pen)
                    + ww * field(i_w, b_w, c_w, mode, pen)
                    + wd * field(i_d, b_d, c_d, mode, pen)
                    + crude_w * crude)
        return f

    return {
        "crude-only":            lambda parts, crude: crude,
        "cosine(current)":       make("cosine", 0.0, 3.0, 2.0, 2.5, 1e-6),
        "dice":                  make("dice", 0.0, 3.0, 2.0, 2.5, 1e-6),
        "coverage":              make("cover", 0.0, 3.0, 2.0, 2.5, 1e-6),
        "coverage+pen0.15":      make("cover", 0.15, 3.0, 2.0, 2.5, 1e-6),
        "coverage+pen0.25":      make("cover", 0.25, 3.0, 2.0, 2.5, 1e-6),
        "coverage+pen0.35":      make("cover", 0.35, 3.0, 2.0, 2.5, 1e-6),
        "coverage+crude":        make("cover", 0.15, 3.0, 2.0, 2.5, 0.02),
        "coverage+crude(hi)":    make("cover", 0.15, 3.0, 2.0, 2.5, 0.08),
        "cover addr-heavy":      make("cover", 0.15, 2.0, 3.0, 3.0, 0.02),
        "cover name-heavy":      make("cover", 0.15, 4.0, 1.5, 2.0, 0.02),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--country", default="US")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--max-posting", type=int, default=config.MAX_POSTING)
    ap.add_argument("--k", type=int, default=20)
    args = ap.parse_args()

    cache = config.WORK_DIR / f"shard_{args.country}_{args.n}_{args.max_posting}.pkl"
    with open(cache, "rb") as fh:
        shard, gt, pool_n = pickle.load(fh)
    print(f"[cache] {cache.name}  S1={len(shard.s1):,} pool={pool_n:,}")

    rules = RuleSet.load()
    prober = Prober(shard, rules)
    idf = prober.idf

    def mass(ids):
        return float(sum(idf[t] for t in ids)) if ids else 0.0

    # Character-level similarity is the one signal qualitatively different from token
    # overlap: it can see "Bsigaman" ~ "Bingaman" and "Rti" ~ "Reit", which no
    # token-set measure can. Names are reconstructed from skeletonised token ids.
    from rapidfuzz import fuzz
    inv = prober.inv

    def name_str(store, i):
        return " ".join(sorted(inv.get(t, "") for t in store.names(i).tolist()))

    # ---- one crude pass, shortlist kept in memory with precomputed IDF parts
    t0 = time.time()
    rows: List[Tuple[str, List[Tuple[int, float, tuple]]]] = []
    for i, eid in enumerate(shard.s1.ids):
        crude = prober.candidates(i, 10 ** 9, rerank_pool=0)
        short = heapq.nlargest(SHORTLIST, crude, key=lambda kv: kv[1])
        s1n, s1w, s1d = _sets(shard.s1, i)
        b_n, b_w, b_d = mass(s1n), mass(s1w), mass(s1d)
        q_name = name_str(shard.s1, i)
        prepared = []
        for ci, cs in short:
            cn, cw, cd = _sets(shard.pool, ci)
            csim = fuzz.token_sort_ratio(q_name, name_str(shard.pool, ci)) / 100.0
            prepared.append((ci, cs, (
                (mass(s1n & cn), b_n, mass(cn)),
                (mass(s1w & cw), b_w, mass(cw)),
                (mass(s1d & cd), b_d, mass(cd)),
            ), csim))
        rows.append((eid, prepared))
    print(f"[prep]  {time.time()-t0:.1f}s\n")

    pool_ids = shard.pool.ids
    print(f"{'variant':22s} {'recall@%d' % args.k:>10s} {'ceilingF0.5':>12s}")
    print("-" * 48)
    variants = build_variants()
    # char-similarity blends, added on top of the crude key score
    for w in (0.5, 1.0, 2.0, 4.0, 8.0):
        variants[f"crude+{w}*charsim"] = (
            lambda parts, crude, csim, _w=w: crude + _w * csim
        )
    variants["charsim-only"] = lambda parts, crude, csim: csim

    results = []
    for name, fn in variants.items():
        pred: Dict[str, List[str]] = {}
        for eid, prepared in rows:
            try:
                scored = [(ci, fn(parts, cs, csim)) for ci, cs, parts, csim in prepared]
            except TypeError:
                scored = [(ci, fn(parts, cs)) for ci, cs, parts, csim in prepared]
            top = heapq.nlargest(args.k, scored, key=lambda kv: kv[1])
            top.sort(key=lambda kv: -kv[1])
            pred[eid] = [pool_ids[ci] for ci, _ in top]
        b = score_blocking(pred, gt, pool_size=pool_n)
        results.append((b.pair_completeness, b.ceiling_f05, name))
        print(f"{name:22s} {b.pair_completeness:10.4f} {b.ceiling_f05:12.4f}")
    best = max(results)
    print("-" * 48)
    print(f"BEST: {best[2]}  recall={best[0]:.4f}  ceiling={best[1]:.4f}")


if __name__ == "__main__":  # pragma: no cover
    main()
