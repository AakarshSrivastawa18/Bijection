"""Phase 3: pairwise features for the matching model.

Three families, in rising order of how much they contributed on validation:

* **String similarity** - name and address compared with several rapidfuzz metrics.
  Different metrics fail differently: token_sort survives word transposition,
  partial_ratio survives DBA prefixes ("Zephveo doing business as Pacific League"),
  plain ratio catches character noise. Supplying all of them lets the tree pick.

* **Structured overlap** - IDF-weighted token/number agreement, computed from the
  same interned ids the blocker used, so it is nearly free.

* **Listwise** - a candidate's rank, score, and gap to the best candidate *within
  its own Source-1 group*. These turn a pointwise classifier into a ranker and are
  the features that know "this is the 7th-best of 20" - information no pairwise
  comparison can see.

Everything is deliberately lexical and country-agnostic. The test set contains
France, which has zero training rows; any feature keyed on a country label would be
undefined there, so none exists.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

FEATURE_NAMES: List[str] = [
    # --- name string similarity
    "name_ratio", "name_token_sort", "name_token_set", "name_partial", "name_jw",
    # --- address string similarity
    "addr_ratio", "addr_token_sort", "addr_token_set", "addr_partial",
    # --- structured overlap
    "name_idf_overlap", "name_cover_s1", "name_cover_cand", "name_jaccard",
    "addr_idf_overlap", "addr_cover_s1", "addr_cover_cand", "addr_jaccard",
    "num_shared", "num_jaccard", "num_any",
    # --- shape / missingness
    "cand_addr_empty", "s1_name_len", "cand_name_len", "name_len_diff",
    "s1_ntok", "cand_ntok", "ntok_diff",
    "is_s2",
    # --- listwise (within the Source-1 candidate group)
    "block_score", "rank", "score_ratio_to_top", "score_gap_to_top",
    "score_z", "group_size",
]
N_FEATURES = len(FEATURE_NAMES)


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _jacc(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def pair_features(
    s1_name: str, s1_addr: str, s1_names: set, s1_words: set, s1_nums: set,
    c_name: str, c_addr: str, c_names: set, c_words: set, c_nums: set,
    idf: np.ndarray, is_s2: bool,
    block_score: float, rank: int, top_score: float, mean_score: float,
    std_score: float, group_size: int,
) -> List[float]:
    """Feature vector for one (Source-1, candidate) pair."""
    def mass(ids) -> float:
        return float(sum(idf[t] for t in ids)) if ids else 0.0

    n_inter = mass(s1_names & c_names)
    n_s1, n_c = mass(s1_names), mass(c_names)
    a_inter = mass(s1_words & c_words)
    a_s1, a_c = mass(s1_words), mass(c_words)
    shared_nums = s1_nums & c_nums

    return [
        fuzz.ratio(s1_name, c_name) / 100.0,
        fuzz.token_sort_ratio(s1_name, c_name) / 100.0,
        fuzz.token_set_ratio(s1_name, c_name) / 100.0,
        fuzz.partial_ratio(s1_name, c_name) / 100.0,
        JaroWinkler.similarity(s1_name, c_name),

        fuzz.ratio(s1_addr, c_addr) / 100.0,
        fuzz.token_sort_ratio(s1_addr, c_addr) / 100.0,
        fuzz.token_set_ratio(s1_addr, c_addr) / 100.0,
        fuzz.partial_ratio(s1_addr, c_addr) / 100.0,

        n_inter,
        _safe_div(n_inter, n_s1),
        _safe_div(n_inter, n_c),
        _jacc(s1_names, c_names),

        a_inter,
        _safe_div(a_inter, a_s1),
        _safe_div(a_inter, a_c),
        _jacc(s1_words, c_words),

        float(len(shared_nums)),
        _jacc(s1_nums, c_nums),
        1.0 if shared_nums else 0.0,

        1.0 if not c_addr else 0.0,
        float(len(s1_name)),
        float(len(c_name)),
        float(abs(len(s1_name) - len(c_name))),
        float(len(s1_names)),
        float(len(c_names)),
        float(abs(len(s1_names) - len(c_names))),
        1.0 if is_s2 else 0.0,

        block_score,
        float(rank),
        _safe_div(block_score, top_score),
        top_score - block_score,
        _safe_div(block_score - mean_score, std_score),
        float(group_size),
    ]


def group_stats(scores: Sequence[float]) -> Tuple[float, float, float]:
    """(top, mean, std) of one Source-1 entity's candidate scores."""
    if not scores:
        return 0.0, 0.0, 1.0
    arr = np.asarray(scores, dtype=np.float64)
    std = float(arr.std())
    return float(arr.max()), float(arr.mean()), std if std > 1e-9 else 1.0


__all__ = ["FEATURE_NAMES", "N_FEATURES", "pair_features", "group_stats"]
