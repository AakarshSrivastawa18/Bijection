"""Phase 3 tests: feature vector shape, semantics, and country-agnosticism."""
from __future__ import annotations

import numpy as np

from bijection.features import FEATURE_NAMES, N_FEATURES, group_stats, pair_features

IDF = np.array([0.0, 5.0, 4.0, 3.0, 2.0, 1.0, 6.0, 6.0], dtype=np.float64)


def _fv(s1_name, s1_addr, c_name, c_addr, s1n=(1, 2), s1w=(3,), s1d=(4,),
        cn=(1, 2), cw=(3,), cd=(4,), is_s2=True, score=10.0, rank=0,
        top=10.0, mean=5.0, std=2.0, gsize=20):
    return pair_features(s1_name, s1_addr, set(s1n), set(s1w), set(s1d),
                         c_name, c_addr, set(cn), set(cw), set(cd),
                         IDF, is_s2, score, rank, top, mean, std, gsize)


def test_vector_length_matches_declared_names():
    v = _fv("alpha beta", "10 main st", "alpha beta", "10 main st")
    assert len(v) == N_FEATURES == len(FEATURE_NAMES)
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)   # no duplicate names
    assert all(isinstance(x, float) for x in v)
    assert not any(np.isnan(x) or np.isinf(x) for x in v)


def test_identical_records_score_at_the_top():
    v = _fv("rapid reit", "6519 17th ave seattle wa", "rapid reit",
            "6519 17th ave seattle wa")
    d = dict(zip(FEATURE_NAMES, v))
    assert d["name_ratio"] == 1.0
    assert d["addr_ratio"] == 1.0
    assert d["name_jw"] == 1.0
    assert d["num_any"] == 1.0


def test_unrelated_records_score_low():
    v = _fv("rapid reit", "6519 17th ave seattle",
            "zzz different qqq", "999 nowhere ln fargo",
            cn=(6, 7), cw=(5,), cd=(0,))
    d = dict(zip(FEATURE_NAMES, v))
    assert d["name_ratio"] < 0.5
    assert d["name_idf_overlap"] == 0.0
    assert d["num_any"] == 0.0


def test_empty_candidate_address_is_flagged_not_crashed():
    """4.55% of true matches have a blank address; they must stay scorable."""
    v = _fv("rapid reit llc", "6519 17th ave", "rapid reit llc", "",
            cw=(), cd=())
    d = dict(zip(FEATURE_NAMES, v))
    assert d["cand_addr_empty"] == 1.0
    assert d["name_ratio"] == 1.0            # name evidence still intact
    assert d["addr_cover_s1"] == 0.0
    assert not any(np.isnan(x) for x in v)


def test_token_sort_survives_word_transposition():
    """Observed: "MD, Jania Sherman," vs "Jania Sherman, MD"."""
    v = _fv("jania sherman md", "x", "md jania sherman", "x")
    d = dict(zip(FEATURE_NAMES, v))
    assert d["name_token_sort"] == 1.0
    assert d["name_ratio"] < 1.0             # plain ratio does not


def test_partial_ratio_survives_dba_prefix():
    """Observed: "Zephveo doing business as Pacific League"."""
    v = _fv("pacific league", "x", "zephveo doing business as pacific league", "x")
    d = dict(zip(FEATURE_NAMES, v))
    assert d["name_partial"] == 1.0
    assert d["name_ratio"] < 0.8


def test_coverage_features_are_asymmetric():
    v = _fv("a", "b", "c", "d", s1n=(1,), cn=(1, 2, 3))
    d = dict(zip(FEATURE_NAMES, v))
    assert d["name_cover_s1"] == 1.0         # all of S1 accounted for
    assert d["name_cover_cand"] < 1.0        # candidate has extra material


def test_listwise_features_reflect_position_in_group():
    top = _fv("a", "b", "a", "b", score=10.0, rank=0, top=10.0)
    low = _fv("a", "b", "a", "b", score=2.0, rank=17, top=10.0)
    dt, dl = dict(zip(FEATURE_NAMES, top)), dict(zip(FEATURE_NAMES, low))
    assert dt["score_ratio_to_top"] == 1.0 and dt["score_gap_to_top"] == 0.0
    assert dl["rank"] == 17.0
    assert dl["score_ratio_to_top"] < dt["score_ratio_to_top"]
    assert dl["score_gap_to_top"] > 0


def test_is_s2_flag():
    assert dict(zip(FEATURE_NAMES, _fv("a", "b", "a", "b", is_s2=True)))["is_s2"] == 1.0
    assert dict(zip(FEATURE_NAMES, _fv("a", "b", "a", "b", is_s2=False)))["is_s2"] == 0.0


def test_no_feature_encodes_country():
    """France has zero training rows, so a country-keyed feature would be undefined
    there. The feature set must stay purely lexical/structural."""
    for n in FEATURE_NAMES:
        assert "country" not in n and "france" not in n.lower()


def test_group_stats_handles_degenerate_inputs():
    assert group_stats([]) == (0.0, 0.0, 1.0)
    top, mean, std = group_stats([5.0, 5.0, 5.0])
    assert top == 5.0 and mean == 5.0 and std == 1.0    # zero variance -> safe divisor
    top, mean, std = group_stats([1.0, 3.0])
    assert top == 3.0 and mean == 2.0 and std > 0


def test_division_guards_on_all_empty_sides():
    v = _fv("", "", "", "", s1n=(), s1w=(), s1d=(), cn=(), cw=(), cd=(),
            score=0.0, top=0.0, mean=0.0, std=0.0)
    assert not any(np.isnan(x) or np.isinf(x) for x in v)
