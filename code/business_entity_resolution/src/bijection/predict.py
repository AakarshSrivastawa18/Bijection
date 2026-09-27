"""Phase 5: end-to-end inference over the test set.

    python -m bijection.predict --split test

Writes `output/candidate_pairs.tsv` (blocking output) and
`output/matching_results.tsv` (final matches).

Processed one country at a time. That is not just a memory convenience - country is
a verified hard partition of the ground truth, so no candidate and no one-to-one
contention ever crosses a shard boundary, and per-country processing is exactly
equivalent to doing it globally.

Memory: the full test set is 34.6M candidate pairs. Candidate rows are written to
disk as each batch is scored rather than accumulated, and only pairs at or above the
retention threshold are held for the assignment step.
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

from . import config, model as M
from .candidates import build_shard
from .dataset import iter_batches
from .decide import assign
from .io_utils import list_countries, read_records
from .mine_rules import RuleSet


def _sources(split: str):
    return config.TEST if split == "test" else config.TRAIN


def _open_outputs(out_dir: Path) -> Tuple:
    out_dir.mkdir(parents=True, exist_ok=True)
    cf = open(out_dir / "candidate_pairs.tsv", "w", encoding="utf-8", newline="")
    mf = open(out_dir / "matching_results.tsv", "w", encoding="utf-8", newline="")
    cf.write("source1_entity_id\tcandidate_entity_ids\n")
    mf.write("source1_entity_id\tmatched_entity_ids\n")
    return cf, mf


def run(split: str = "test", top_k: int = config.TOP_K,
        threshold: float | None = None, out_dir: Path | None = None,
        countries: Sequence[str] | None = None, one_to_one: bool = True,
        limit: int = 0, rescue: float | None = None) -> None:
    src = _sources(split)
    out_dir = out_dir or config.OUTPUT_DIR
    rules = RuleSet.load()
    booster = M.load()
    if threshold is None:
        tp = config.WORK_DIR / "threshold.txt"
        if tp.exists():
            lines = tp.read_text().splitlines()
            threshold = float(lines[0])
            if rescue is None and len(lines) > 1 and lines[1].strip():
                rescue = float(lines[1])
        else:
            threshold = 0.5
    print(f"[cfg] split={split} top_k={top_k} threshold={threshold:.3f} "
          f"rescue={rescue} one_to_one={one_to_one}")

    todo = list(countries) if countries else sorted(list_countries(src["s1"]))
    print(f"[cfg] countries={todo}")

    cf, mf = _open_outputs(out_dir)
    n_entities = n_cand = n_match = 0
    try:
        for country in todo:
            t0 = time.time()
            s1 = [(e, n, a) for e, n, a, c in read_records(src["s1"], {country})]
            if limit:
                # Smoke-test mode. The POOL is never truncated - shrinking it would
                # make retrieval artificially easy and prove nothing about the real run.
                s1 = s1[:limit]
            pool = [(e, n, a) for key in ("s2", "s3")
                    for e, n, a, c in read_records(src[key], {country})]
            print(f"\n[{country}] S1={len(s1):,} pool={len(pool):,} "
                  f"(load {time.time()-t0:.0f}s)", flush=True)

            t0 = time.time()
            shard = build_shard(country, s1, pool, rules)
            del pool, s1
            print(f"[{country}] index postings={shard.index.keys.size:,} "
                  f"({time.time()-t0:.0f}s)", flush=True)

            t0 = time.time()
            kept: List[Tuple[str, str, float]] = []
            seen_entities: Set[str] = set()
            done = 0
            for X, _y, row_s1, row_cand in iter_batches(shard, rules, top_k):
                if len(X):
                    scores = booster.predict(X, num_iteration=booster.best_iteration)
                else:
                    scores = np.zeros(0)
                # candidate_pairs: write immediately, never accumulate
                groups: Dict[str, List[str]] = defaultdict(list)
                for s, c in zip(row_s1, row_cand):
                    groups[s].append(c)
                for s, cs in groups.items():
                    cf.write(f"{s}\t{','.join(cs)}\n")
                    n_cand += len(cs)
                seen_entities.update(groups)
                floor = threshold if rescue is None else min(threshold, rescue)
                for s, c, p in zip(row_s1, row_cand, scores):
                    if p >= floor:
                        kept.append((s, c, float(p)))
                done += len(groups)
                if done and done % 200_000 < 20_000:
                    print(f"[{country}]   scored ~{done:,} entities", flush=True)

            # Entities whose blocking returned nothing still need a row in BOTH files.
            all_ids = shard.s1.ids
            for e in all_ids:
                if e not in seen_entities:
                    cf.write(f"{e}\t\n")
            print(f"[{country}] scored {len(all_ids):,} entities, "
                  f"{len(kept):,} pairs >= threshold ({time.time()-t0:.0f}s)", flush=True)

            matches = assign(kept, threshold, all_ids, one_to_one=one_to_one,
                             rescue=rescue)
            for e in all_ids:
                ids = matches.get(e, [])
                mf.write(f"{e}\t{','.join(ids)}\n")
                n_match += len(ids)
            n_entities += len(all_ids)
            del shard, kept, matches
    finally:
        cf.close()
        mf.close()

    print(f"\n[done] entities={n_entities:,}  candidates={n_cand:,} "
          f"({n_cand/max(1,n_entities):.2f}/entity)  matches={n_match:,} "
          f"({n_match/max(1,n_entities):.2f}/entity)")
    print(f"[out] {out_dir/'candidate_pairs.tsv'}")
    print(f"[out] {out_dir/'matching_results.tsv'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--top-k", type=int, default=config.TOP_K)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--countries", nargs="*", default=None)
    ap.add_argument("--no-one-to-one", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke test: cap Source-1 entities per country (pool stays full)")
    a = ap.parse_args()
    run(split=a.split, top_k=a.top_k, threshold=a.threshold,
        out_dir=Path(a.out) if a.out else None,
        countries=a.countries, one_to_one=not a.no_one_to_one, limit=a.limit)


if __name__ == "__main__":  # pragma: no cover
    main()
