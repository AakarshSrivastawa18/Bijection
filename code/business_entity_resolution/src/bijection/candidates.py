"""Phase 2 driver: build one index per country shard and probe it.

Produces, for every Source-1 entity, the top-K Source-2/3 records ranked by
accumulated blocking-key weight. That ranked list is exactly what is written to
`candidate_pairs.tsv` and exactly what the matching model scores - there is no
further filtering stage between the two.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from . import config
from .blocking import (
    KEY_WEIGHT, MAX_ADDR_WORDS, MAX_NAME_TOKENS, MAX_NUMS, BlockIndex, RecordStore,
    Vocab, _top_by_idf, build_store, record_keys,
)
from .mine_rules import RuleSet


@dataclass
class Shard:
    """One country's records plus the structures built over them."""

    country: str
    s1: RecordStore
    pool: RecordStore
    pool_df: np.ndarray
    index: BlockIndex
    s1_full: np.ndarray
    vocab: Vocab
    name_vocab: Vocab

    @property
    def pool_size(self) -> int:
        return len(self.pool)


def build_shard(country: str,
                s1_records: Iterable[Tuple[str, str, str]],
                pool_records: Iterable[Tuple[str, str, str]],
                rules: RuleSet,
                max_posting: int = config.MAX_POSTING) -> Shard:
    vocab = Vocab()
    name_vocab = Vocab()
    # Pool first so that document-frequency reflects the searchable corpus; the
    # Source-1 side then reuses the same vocab and the same IDF ranking.
    pool, pool_df, pool_full = build_store(pool_records, rules, vocab, name_vocab)
    s1, _, s1_full = build_store(s1_records, rules, vocab, name_vocab)

    # DF grew as the S1 pass added new tokens; pad so indexing never runs off the end.
    if pool_df.size < len(vocab):
        pool_df = np.concatenate([pool_df, np.zeros(len(vocab) - pool_df.size, dtype=np.int64)])

    index = BlockIndex.build(pool, pool_df, vocab, pool_full, max_posting)
    return Shard(country=country, s1=s1, pool=pool, pool_df=pool_df, index=index,
                 s1_full=s1_full, vocab=vocab, name_vocab=name_vocab)


class Prober:
    """Probes one shard, reusing an inverted string table for expansion lookups."""

    def __init__(self, shard: Shard, rules: RuleSet) -> None:
        self.shard = shard
        self.rules = rules
        # int id -> surface string, needed to consult the mined expansion table.
        self.inv: Dict[int, str] = {v: k for k, v in shard.vocab._d.items()}
        # Smoothed IDF over the searchable pool. A key matched on a rare token is
        # worth far more than one matched on "pacific" or "bordeaux"; weighting by
        # this is what moves true matches into the top of the ranked list.
        n = max(1, shard.pool_size)
        df = shard.pool_df.astype(np.float64)
        self.idf = np.log(1.0 + n / (1.0 + df))

    def _expand_ids(self, ids: Sequence[int]) -> List[int]:
        out: List[int] = []
        seen = set()
        for i in ids:
            if i in seen:
                continue
            seen.add(i)
            out.append(i)
            surf = self.inv.get(i)
            if surf is None:
                continue
            for alt in self.rules.expand(surf):
                j = self.shard.vocab.get(alt)
                if j >= 0 and j not in seen:
                    seen.add(j)
                    out.append(j)
        return out

    def candidates(self, i: int, top_k: int, rerank_pool: int = 0
                   ) -> List[Tuple[int, float]]:
        """Top-k pool indices for Source-1 row `i`, ranked by IDF-weighted key score.

        `rerank_pool` is retained for the tuning harness but is a no-op: a measured
        comparison of overlap rerankers (cosine / dice / coverage, with and without
        length penalties, plus character-level similarity) found NONE of them beat
        this score. See tune_blocking.py - every variant landed within 0.001 of
        0.9542 recall@20, and most were 1-2 points worse. The ordering signal
        available at token level is saturated; discriminating further is the
        matching model's job, not the blocker's.
        """
        sh = self.shard
        df = sh.pool_df
        names = _top_by_idf(sh.s1.names(i), df, MAX_NAME_TOKENS)
        words = _top_by_idf(sh.s1.words(i), df, MAX_ADDR_WORDS)
        nums = [int(x) for x in np.unique(sh.s1.nums(i))[:MAX_NUMS]]

        # Expand address words only. Mined alternates are overwhelmingly street
        # types and state initialisms; expanding name tokens as well tripled the
        # key count and dragged in unrelated records without adding recall.
        words = self._expand_ids(words)[: MAX_ADDR_WORDS * 3]

        scores: Dict[int, float] = {}
        keys = record_keys(names, words, nums, sh.s1_full[i], sh.s1.lsh(i))
        idx = sh.index
        idf = self.idf
        nidf = float(idf.max()) if idf.size else 1.0
        for key, fam, ta, tb in keys:
            lo, hi = idx.lookup(np.uint64(key))
            if lo == hi:
                continue
            if ta < 0:                      # whole-name key: no constituent tokens
                w = KEY_WEIGHT[fam] * nidf
            else:
                w = KEY_WEIGHT[fam] * (float(idf[ta]) + float(idf[tb]))
            for v in idx.vals[lo:hi]:
                vi = int(v)
                scores[vi] = scores.get(vi, 0.0) + w
        if not scores:
            return []
        if len(scores) <= top_k:
            return sorted(scores.items(), key=lambda kv: -kv[1])
        ranked = heapq.nlargest(top_k, scores.items(), key=lambda kv: kv[1])
        ranked.sort(key=lambda kv: -kv[1])
        return ranked


def generate(shard: Shard, rules: RuleSet, top_k: int = config.TOP_K
             ) -> Dict[str, List[str]]:
    """entity_id -> ranked candidate entity_ids for every Source-1 row in the shard."""
    prober = Prober(shard, rules)
    out: Dict[str, List[str]] = {}
    pool_ids = shard.pool.ids
    for i, eid in enumerate(shard.s1.ids):
        out[eid] = [pool_ids[j] for j, _ in prober.candidates(i, top_k)]
    return out


__all__ = ["Shard", "build_shard", "Prober", "generate"]
