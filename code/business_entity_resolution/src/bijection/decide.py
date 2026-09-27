"""Phase 4: turn pairwise scores into final match sets.

This is where F_0.5 is won or lost. Three mechanisms, in order of contribution:

1. **Threshold**, tuned directly for macro-F_0.5 on validation rather than for
   accuracy or AUC. For a typical entity with 3 true matches, one false positive
   costs 0.211 of that entity's score while one false negative costs 0.091 - a
   2.3x asymmetry that pushes the optimum well above 0.5.

2. **Global one-to-one assignment.** Verified exhaustively over all 7,638,365
   training ground-truth pairs: every Source-2/3 record belongs to exactly ONE
   Source-1 entity, with zero reuse. So when two entities both claim a record, at
   most one can be right. Resolving that greedily by score can only remove false
   positives, never add them - which on a precision-weighted metric is close to
   free score. Most pipelines treat pair scoring as independent and leave this.

3. **Per-source caps.** Training ground truth never exceeds 5 Source-2 or 6
   Source-3 records for one entity; the caps here carry one slot of headroom.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from . import config
from .metrics import score_matching

Triple = Tuple[str, str, float]   # (source1_id, candidate_id, score)


def assign(triples: Iterable[Triple], threshold: float,
           all_s1: Sequence[str],
           max_per_source: Mapping[str, int] | None = None,
           one_to_one: bool = True,
           rescue: float | None = None) -> Dict[str, List[str]]:
    """Greedy score-descending assignment under the one-to-one and cap constraints.

    Greedy is an approximation to maximum-weight bipartite b-matching, but with
    scores this separated the two coincide almost everywhere, and greedy is O(n log n)
    rather than cubic - which matters at 34.6M candidate pairs.

    `rescue`: entities that end up EMPTY after the main pass get their single best
    candidate if it scores >= rescue (< threshold). Rationale: only 5.6% of entities
    are true singletons, so an empty prediction is usually a miss, and for an entity
    with n true matches one correct rescue moves its F0.5 from 0 to 1.25/(0.25+n) -
    worth it whenever the rescued candidate is right more than ~n/5 of the time.
    """
    caps = dict(max_per_source or config.MAX_PER_SOURCE)
    floor = threshold if rescue is None else min(threshold, rescue)
    kept = [t for t in triples if t[2] >= floor]
    kept.sort(key=lambda t: -t[2])

    out: Dict[str, List[str]] = {s: [] for s in all_s1}
    taken: set = set()
    per_src: Dict[Tuple[str, str], int] = defaultdict(int)

    for s1, cid, score in kept:
        if score < threshold:
            break                         # sorted: everything below is rescue-only
        if one_to_one and cid in taken:
            continue
        src = cid[:2]
        cap = caps.get(src)
        if cap is not None and per_src[(s1, src)] >= cap:
            continue
        lst = out.get(s1)
        if lst is None:
            continue                      # candidate for an entity we were not asked about
        lst.append(cid)
        per_src[(s1, src)] += 1
        if one_to_one:
            taken.add(cid)

    if rescue is not None and rescue < threshold:
        # second pass, still score-descending, only for entities that stayed empty
        for s1, cid, score in kept:
            if score >= threshold:
                continue
            lst = out.get(s1)
            if lst is None or lst:
                continue
            if one_to_one and cid in taken:
                continue
            lst.append(cid)
            if one_to_one:
                taken.add(cid)
    return out


@dataclass
class ThresholdSweep:
    best_threshold: float
    best_f05: float
    curve: List[Tuple[float, float, float, float]]  # (thr, F0.5, precision, recall)

    def report(self) -> str:
        lines = [f"{'thr':>6s} {'macroF0.5':>10s} {'meanP':>8s} {'meanR':>8s}"]
        for thr, f, p, r in self.curve:
            mark = "  <-- best" if abs(thr - self.best_threshold) < 1e-12 else ""
            lines.append(f"{thr:6.2f} {f:10.5f} {p:8.4f} {r:8.4f}{mark}")
        return "\n".join(lines)


def tune_threshold(triples: Sequence[Triple], ground_truth: Mapping[str, Iterable[str]],
                   all_s1: Sequence[str], one_to_one: bool = True,
                   grid: Sequence[float] | None = None,
                   max_per_source: Mapping[str, int] | None = None) -> ThresholdSweep:
    grid = grid if grid is not None else [i / 100 for i in range(5, 100, 5)]
    curve: List[Tuple[float, float, float, float]] = []
    best = (-1.0, 0.5)
    for thr in grid:
        pred = assign(triples, thr, all_s1, max_per_source, one_to_one)
        s = score_matching(pred, ground_truth)
        curve.append((thr, s.macro_f05, s.mean_precision, s.mean_recall))
        if s.macro_f05 > best[0]:
            best = (s.macro_f05, thr)
    return ThresholdSweep(best_threshold=best[1], best_f05=best[0], curve=curve)


__all__ = ["Triple", "assign", "tune_threshold", "ThresholdSweep"]
