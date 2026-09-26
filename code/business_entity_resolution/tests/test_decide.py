"""Phase 4 tests: threshold, the one-to-one constraint, and per-source caps."""
from __future__ import annotations

from bijection.decide import assign, tune_threshold


def test_threshold_filters_below_cutoff():
    tri = [("S1-1", "S2-a", 0.9), ("S1-1", "S2-b", 0.3)]
    out = assign(tri, 0.5, ["S1-1"])
    assert out["S1-1"] == ["S2-a"]


def test_every_requested_entity_gets_a_row_even_with_no_candidates():
    """The grader rejects a submission with a missing Source-1 entity."""
    out = assign([("S1-1", "S2-a", 0.9)], 0.5, ["S1-1", "S1-2", "S1-3"])
    assert set(out) == {"S1-1", "S1-2", "S1-3"}
    assert out["S1-2"] == [] and out["S1-3"] == []


def test_one_to_one_resolves_contention_by_score():
    """Verified over all 7,638,365 training pairs: an S2/S3 record belongs to
    exactly one Source-1 entity. When two entities claim it, only the stronger wins."""
    tri = [("S1-1", "S2-x", 0.95), ("S1-2", "S2-x", 0.80)]
    out = assign(tri, 0.5, ["S1-1", "S1-2"], one_to_one=True)
    assert out["S1-1"] == ["S2-x"]
    assert out["S1-2"] == []


def test_independent_mode_allows_the_duplicate():
    tri = [("S1-1", "S2-x", 0.95), ("S1-2", "S2-x", 0.80)]
    out = assign(tri, 0.5, ["S1-1", "S1-2"], one_to_one=False)
    assert out["S1-1"] == ["S2-x"] and out["S1-2"] == ["S2-x"]


def test_one_to_one_only_ever_removes_predictions():
    """It can never introduce a false positive - that is why it is safe on F0.5."""
    tri = [("S1-1", "S2-x", 0.9), ("S1-2", "S2-x", 0.8), ("S1-2", "S3-y", 0.7)]
    free = assign(tri, 0.5, ["S1-1", "S1-2"], one_to_one=False)
    strict = assign(tri, 0.5, ["S1-1", "S1-2"], one_to_one=True)
    for k in free:
        assert set(strict[k]) <= set(free[k])


def test_per_source_caps_are_enforced_separately():
    tri = [("S1-1", f"S2-{i}", 0.9 - i / 100) for i in range(8)]
    tri += [("S1-1", f"S3-{i}", 0.9 - i / 100) for i in range(9)]
    out = assign(tri, 0.5, ["S1-1"], max_per_source={"S2": 2, "S3": 3})
    assert sum(c.startswith("S2-") for c in out["S1-1"]) == 2
    assert sum(c.startswith("S3-") for c in out["S1-1"]) == 3


def test_caps_keep_the_highest_scoring_candidates():
    tri = [("S1-1", "S2-lo", 0.6), ("S1-1", "S2-hi", 0.99), ("S1-1", "S2-mid", 0.8)]
    out = assign(tri, 0.5, ["S1-1"], max_per_source={"S2": 2, "S3": 2})
    assert out["S1-1"] == ["S2-hi", "S2-mid"]


def test_no_duplicate_ids_within_a_list():
    tri = [("S1-1", "S2-a", 0.9), ("S1-1", "S2-a", 0.8)]
    out = assign(tri, 0.5, ["S1-1"], one_to_one=True)
    assert out["S1-1"] == ["S2-a"]


def test_candidates_for_unknown_entities_are_dropped():
    out = assign([("S1-ghost", "S2-a", 0.9)], 0.5, ["S1-1"])
    assert out == {"S1-1": []}


def test_tune_threshold_prefers_precision():
    """With one true match and one plausible impostor, F0.5 should choose the
    threshold that excludes the impostor rather than the one that catches both."""
    gt = {"S1-1": ["S2-good"]}
    tri = [("S1-1", "S2-good", 0.90), ("S1-1", "S2-bad", 0.55)]
    sweep = tune_threshold(tri, gt, ["S1-1"], grid=[0.5, 0.6, 0.7, 0.8])
    assert sweep.best_threshold >= 0.6
    assert sweep.best_f05 == 1.0


def test_tune_threshold_rewards_correct_singletons():
    gt = {"S1-1": [], "S1-2": ["S2-x"]}
    tri = [("S1-1", "S2-junk", 0.55), ("S1-2", "S2-x", 0.95)]
    sweep = tune_threshold(tri, gt, ["S1-1", "S1-2"], grid=[0.5, 0.9])
    # 0.9 keeps only the true match and correctly leaves S1-1 empty -> 1.0 average
    assert sweep.best_threshold == 0.9 and sweep.best_f05 == 1.0
