"""Phase 0 tests - the scorer must reproduce the competition's worked example."""
from __future__ import annotations

import math

from bijection.metrics import fbeta_half, score_blocking, score_matching


def test_worked_example_from_problem_statement():
    # Problem statement: predicted 3, truth 2, 2 correct -> P=2/3, R=1.0, F0.5=0.714
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    truth = {"S2-00047", "S3-00812"}
    assert math.isclose(fbeta_half(pred, truth), 0.7142857142857143, rel_tol=1e-9)


def test_singleton_conventions():
    assert fbeta_half(set(), set()) == 1.0            # correct empty -> full credit
    assert fbeta_half({"S2-1"}, set()) == 0.0         # false merge on a singleton
    assert fbeta_half(set(), {"S2-1"}) == 0.0         # missed everything


def test_perfect_and_disjoint():
    assert fbeta_half({"a", "b"}, {"a", "b"}) == 1.0
    assert fbeta_half({"a"}, {"b"}) == 0.0


def test_precision_weighted_twice_as_hard_as_recall():
    """The asymmetry that drives every threshold decision: for 3 true matches,
    one false positive must cost more than one false negative."""
    truth = {"a", "b", "c"}
    fp_cost = 1.0 - fbeta_half({"a", "b", "c", "d"}, truth)
    fn_cost = 1.0 - fbeta_half({"a", "b"}, truth)
    assert fp_cost > fn_cost
    assert math.isclose(fp_cost / fn_cost, 2.3158, rel_tol=1e-3)


def test_score_matching_macro_average_includes_singletons():
    gt = {"S1-1": ["S2-1"], "S1-2": [], "S1-3": ["S2-3", "S3-3"]}
    pred = {"S1-1": ["S2-1"], "S1-2": [], "S1-3": ["S2-3"]}
    s = score_matching(pred, gt)
    # 1.0 + 1.0 + F0.5(P=1,R=0.5)
    third = fbeta_half({"S2-3"}, {"S2-3", "S3-3"})
    assert math.isclose(s.macro_f05, (1.0 + 1.0 + third) / 3, rel_tol=1e-9)
    assert s.singleton_total == 1 and s.singleton_correct == 1
    assert s.n_entities == 3


def test_missing_entity_scored_as_empty_prediction():
    gt = {"S1-1": ["S2-1"], "S1-2": []}
    s = score_matching({}, gt)          # predicted nothing at all
    assert math.isclose(s.macro_f05, 0.5)   # 0.0 for S1-1, 1.0 for the singleton


def test_blocking_ceiling_and_reduction():
    gt = {"S1-1": ["S2-1", "S3-1"], "S1-2": ["S2-2"], "S1-3": []}
    cand = {"S1-1": ["S2-1", "S2-9"], "S1-2": ["S2-2"], "S1-3": ["S2-7"]}
    b = score_blocking(cand, gt, pool_size=100)
    assert math.isclose(b.pair_completeness, 2 / 3)      # S3-1 was never retrieved
    assert b.total_candidates == 4
    assert b.search_space == 300
    assert math.isclose(b.avg_candidates, 4 / 3)
    # ceiling: S1-1 can reach at best P=1,R=0.5; S1-2 perfect; S1-3 has a candidate
    # but an ideal model would still emit empty, so it scores 1.0.
    expected = (fbeta_half({"S2-1"}, {"S2-1", "S3-1"}) + 1.0 + 1.0) / 3
    assert math.isclose(b.ceiling_f05, expected, rel_tol=1e-9)
