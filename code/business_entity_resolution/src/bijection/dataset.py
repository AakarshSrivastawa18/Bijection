"""Build (X, y) feature batches from a country shard.

One row per (Source-1 entity, blocking candidate) pair - exactly the pairs written
to `candidate_pairs.tsv` and exactly the pairs the model scores at inference.
Training on the blocker's own output rather than on random negatives is what makes
the learned threshold meaningful: the negatives seen in training are the near-misses
the model will actually have to reject.

Batched by design. The full test set is 1.73M entities x 20 candidates = 34.6M rows;
at 34 float32 features that is 4.7 GB materialised, which does not fit alongside a
shard on a 16 GB machine. Callers consume batches and either accumulate a sample
(training) or predict and discard (inference).
"""
from __future__ import annotations

from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

from .candidates import Prober, Shard
from .features import N_FEATURES, group_stats, pair_features
from .mine_rules import RuleSet

Batch = Tuple[np.ndarray, np.ndarray, List[str], List[str]]


def iter_batches(shard: Shard, rules: RuleSet, top_k: int,
                 ground_truth: Optional[Dict[str, Iterable[str]]] = None,
                 batch_entities: int = 20_000,
                 entity_filter: Optional[set] = None) -> Iterator[Batch]:
    """Yield (X, y, s1_ids, cand_ids) for successive blocks of Source-1 entities.

    `y` is all zeros when ground_truth is None (inference mode). Entities whose
    blocking returned nothing produce no rows but still appear in the caller's
    output - they become empty predictions, which for a true singleton scores 1.0.
    """
    prober = Prober(shard, rules)
    idf = prober.idf
    s1, pool = shard.s1, shard.pool

    rows: List[List[float]] = []
    labels: List[int] = []
    row_s1: List[str] = []
    row_cand: List[str] = []
    n_ent = 0

    def flush() -> Batch:
        X = (np.asarray(rows, dtype=np.float32) if rows
             else np.zeros((0, N_FEATURES), dtype=np.float32))
        return X, np.asarray(labels, dtype=np.int8), list(row_s1), list(row_cand)

    for i, eid in enumerate(s1.ids):
        if entity_filter is not None and eid not in entity_filter:
            continue
        n_ent += 1
        cands = prober.candidates(i, top_k)
        if cands:
            truth = set(ground_truth.get(eid, ())) if ground_truth is not None else set()
            scores = [s for _, s in cands]
            top, mean, std = group_stats(scores)
            gsize = len(cands)

            s1_names = set(s1.names(i).tolist())
            s1_words = set(s1.words(i).tolist())
            s1_nums = set(s1.nums(i).tolist())
            s1_nm, s1_ad = s1.norm_name[i], s1.norm_addr[i]

            for rank, (ci, sc) in enumerate(cands):
                cid = pool.ids[ci]
                rows.append(pair_features(
                    s1_nm, s1_ad, s1_names, s1_words, s1_nums,
                    pool.norm_name[ci], pool.norm_addr[ci],
                    set(pool.names(ci).tolist()), set(pool.words(ci).tolist()),
                    set(pool.nums(ci).tolist()),
                    idf, cid.startswith("S2-"),
                    sc, rank, top, mean, std, gsize,
                ))
                labels.append(1 if cid in truth else 0)
                row_s1.append(eid)
                row_cand.append(cid)

        if n_ent >= batch_entities:
            yield flush()
            rows.clear(); labels.clear(); row_s1.clear(); row_cand.clear()
            n_ent = 0

    if rows or n_ent:
        yield flush()


def build_matrix(shard: Shard, rules: RuleSet, top_k: int,
                 ground_truth: Optional[Dict[str, Iterable[str]]] = None,
                 entity_filter: Optional[set] = None,
                 ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Materialise every batch at once. Only safe for training-sized samples."""
    Xs, ys, s1s, cs = [], [], [], []
    for X, y, a, b in iter_batches(shard, rules, top_k, ground_truth,
                                   entity_filter=entity_filter):
        Xs.append(X); ys.append(y); s1s.extend(a); cs.extend(b)
    if not Xs:
        return np.zeros((0, N_FEATURES), dtype=np.float32), np.zeros(0, np.int8), [], []
    return np.vstack(Xs), np.concatenate(ys), s1s, cs


__all__ = ["iter_batches", "build_matrix", "Batch"]
