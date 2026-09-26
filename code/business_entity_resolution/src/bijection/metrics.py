"""Scoring harness: the exact competition metric plus blocking diagnostics.

The leaderboard metric is macro-averaged F-beta with beta=0.5, computed per
Source-1 entity and then averaged over every Source-1 entity in the evaluation
set - singletons included. A singleton scores 1.0 when you correctly predict an
empty list and 0.0 when you predict anything.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set

BETA2 = 0.25  # beta^2 for beta = 0.5


def fbeta_half(pred: Set[str], truth: Set[str]) -> float:
    """F_0.5 for a single entity, with the competition's singleton convention."""
    if not truth and not pred:
        return 1.0
    if not truth or not pred:
        return 0.0
    tp = len(pred & truth)
    if tp == 0:
        return 0.0
    precision = tp / len(pred)
    recall = tp / len(truth)
    denom = BETA2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.0 + BETA2) * precision * recall / denom


@dataclass
class MatchScore:
    macro_f05: float
    n_entities: int
    mean_precision: float
    mean_recall: float
    exact_sets: int          # entities predicted perfectly
    singleton_total: int
    singleton_correct: int
    predicted_ids: int
    true_ids: int

    def __str__(self) -> str:
        return (
            f"macro_F0.5={self.macro_f05:.5f}  n={self.n_entities}\n"
            f"  mean_precision={self.mean_precision:.5f}  mean_recall={self.mean_recall:.5f}\n"
            f"  exact_set_matches={self.exact_sets} ({100*self.exact_sets/max(1,self.n_entities):.2f}%)\n"
            f"  singletons: {self.singleton_correct}/{self.singleton_total} correct\n"
            f"  ids predicted={self.predicted_ids}  ids in truth={self.true_ids}"
        )


def score_matching(
    predictions: Mapping[str, Iterable[str]],
    ground_truth: Mapping[str, Iterable[str]],
) -> MatchScore:
    """Macro F_0.5 over every entity in `ground_truth`.

    An entity present in ground_truth but missing from predictions is scored as an
    empty prediction - that mirrors the grader, which requires a row per entity.
    """
    total = 0.0
    p_sum = 0.0
    r_sum = 0.0
    exact = 0
    singles = 0
    singles_ok = 0
    n_pred = 0
    n_true = 0

    for s1, truth_ids in ground_truth.items():
        truth = set(truth_ids)
        pred = set(predictions.get(s1, ()))
        n_pred += len(pred)
        n_true += len(truth)
        total += fbeta_half(pred, truth)
        if not truth:
            singles += 1
            if not pred:
                singles_ok += 1
        if pred == truth:
            exact += 1
        # Diagnostics only (the scored metric is `total` above). Conventions:
        # precision is 1.0 when nothing was predicted and nothing was true;
        # recall is 1.0 whenever there is nothing to find.
        tp = len(pred & truth)
        if pred:
            p_sum += tp / len(pred)
        elif not truth:
            p_sum += 1.0
        r_sum += (tp / len(truth)) if truth else 1.0

    n = len(ground_truth)
    return MatchScore(
        macro_f05=total / n if n else 0.0,
        n_entities=n,
        mean_precision=p_sum / n if n else 0.0,
        mean_recall=r_sum / n if n else 0.0,
        exact_sets=exact,
        singleton_total=singles,
        singleton_correct=singles_ok,
        predicted_ids=n_pred,
        true_ids=n_true,
    )


@dataclass
class BlockingScore:
    pair_completeness: float   # recall of true pairs inside the candidate set
    entities_full_recall: float
    avg_candidates: float
    max_candidates: int
    total_candidates: int
    search_space: int
    reduction_ratio: float
    ceiling_f05: float         # best macro F0.5 any downstream model could reach

    def __str__(self) -> str:
        return (
            f"pair_completeness={self.pair_completeness:.5f}  "
            f"entities_with_all_matches={self.entities_full_recall:.5f}\n"
            f"  avg_candidates/S1={self.avg_candidates:.2f}  max={self.max_candidates}  "
            f"total={self.total_candidates:,}\n"
            f"  search_space={self.search_space:,}  reduction_ratio={self.reduction_ratio:.3e}\n"
            f"  CEILING macro_F0.5={self.ceiling_f05:.5f}"
        )


def score_blocking(
    candidates: Mapping[str, Sequence[str]],
    ground_truth: Mapping[str, Iterable[str]],
    pool_size: int,
) -> BlockingScore:
    """Blocking quality.

    `pool_size` is the number of Source-2 + Source-3 records that were searchable,
    i.e. the denominator of the brute-force comparison space.
    """
    found = 0
    total_pairs = 0
    full = 0
    n_cand = 0
    mx = 0
    ceiling = 0.0

    for s1, truth_ids in ground_truth.items():
        truth = set(truth_ids)
        cand = set(candidates.get(s1, ()))
        n_cand += len(cand)
        mx = max(mx, len(cand))
        hit = truth & cand
        found += len(hit)
        total_pairs += len(truth)
        if hit == truth:
            full += 1
        # best case downstream: predict exactly the reachable true matches
        ceiling += fbeta_half(hit, truth)

    n = len(ground_truth)
    space = n * pool_size
    return BlockingScore(
        pair_completeness=found / total_pairs if total_pairs else 1.0,
        entities_full_recall=full / n if n else 0.0,
        avg_candidates=n_cand / n if n else 0.0,
        max_candidates=mx,
        total_candidates=n_cand,
        search_space=space,
        reduction_ratio=(n_cand / space) if space else 0.0,
        ceiling_f05=ceiling / n if n else 0.0,
    )


__all__ = ["fbeta_half", "score_matching", "score_blocking", "MatchScore", "BlockingScore"]
