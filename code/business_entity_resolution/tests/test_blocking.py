"""Phase 2 tests: key symmetry, index mechanics, and end-to-end retrieval.

The single most important property is that `record_keys` produces IDENTICAL hashes
for the index side and the probe side - if those ever diverge, blocking silently
returns nothing and the whole pipeline scores zero while every unit test on the
pieces still passes.
"""
from __future__ import annotations

import numpy as np

from bijection.blocking import (
    K_EXACT_NAME, K_NAME_ADDR, BlockIndex, Vocab, _mix, build_store, prepare,
    record_keys,
)
from bijection.candidates import Prober, build_shard
from bijection.mine_rules import RuleSet

RULES = RuleSet(
    droppable={"llc", "inc", "ltd", "limited", "pvt", "private", "corp"},
    expansions={"st": ["street"], "street": ["st"], "ave": ["avenue"],
                "avenue": ["ave"], "rd": ["road"], "road": ["rd"]},
    phrases={"new york": "ny"},
    deva={"लिमिटेड": "limited"},
)


def test_mix_is_deterministic_and_order_sensitive():
    assert _mix(1, 2, 3) == _mix(1, 2, 3)
    assert _mix(1, 2, 3) != _mix(1, 3, 2)
    assert _mix(1, 2, 3) != _mix(2, 2, 3)      # family is part of the key
    assert 0 <= _mix(7, 11, 13) < (1 << 63)


def test_mix_accepts_numpy_ints_regression():
    """numpy int64 * a 64-bit constant raises OverflowError instead of wrapping."""
    assert _mix(K_EXACT_NAME, np.int64(123456789), 0) == _mix(K_EXACT_NAME, 123456789, 0)


def test_keys_identical_for_index_and_probe_sides():
    a = record_keys([5, 9], [2], [77], 42)
    b = record_keys([5, 9], [2], [77], 42)
    assert a == b
    assert {fam for _, fam, _, _ in a} >= {K_NAME_ADDR, K_EXACT_NAME}


def test_name_pair_key_is_order_independent():
    """Word-order transposition is a documented noise pattern; the pair key must
    survive it ("MD, Jania Sherman," vs "Jania Sherman, MD")."""
    k1 = {k for k, f, _, _ in record_keys([3, 8], [], [], -1)}
    k2 = {k for k, f, _, _ in record_keys([8, 3], [], [], -1)}
    assert k1 == k2


def test_no_exact_name_key_when_name_empty():
    fams = {f for _, f, _, _ in record_keys([], [1], [2], -1)}
    assert K_EXACT_NAME not in fams


def test_prepare_applies_full_chain():
    n, w, d, nn, na = prepare("Rapid Reit LLC", "6519 17th Avenue, Seattle, WA", RULES)
    assert "llc" not in n                      # droppable stripped
    assert d == ["6519"]                       # house number isolated
    # "seattle" -> "seattie": the l->i confusable fold is lossy but applied to BOTH
    # sides, so the two records still meet. What matters is convergence, not fidelity.
    assert "seattie" in w
    # conservative forms preserved untouched for the matcher's similarity features
    assert nn == "rapid reit llc"
    assert na == "6519 17th avenue, seattle, wa".replace(",", "")


def test_skeleton_fold_is_lossy_but_convergent():
    """Differently-cased/abbreviated forms of one address must land on one key set."""
    _, w1, d1, _, _ = prepare("Rapid Reit LLC", "6519 17th Avenue, Seattle, WA", RULES)
    _, w2, d2, _, _ = prepare("RAPID REIT", "6519 17TH AVENUE, SEATTLE, WA", RULES)
    assert set(w1) == set(w2) and d1 == d2


def test_prepare_survives_empty_address():
    n, w, d, nn, na = prepare("Rapid Reit LLC", "", RULES)
    assert n and w == [] and d == []
    assert nn == "rapid reit llc" and na == ""


def test_build_store_csr_roundtrip():
    vocab, nv = Vocab(), Vocab()
    recs = [("S2-1", "Alpha Beta LLC", "10 Main St, Springfield"),
            ("S2-2", "Gamma", "")]
    store, df, full = build_store(recs, RULES, vocab, nv)
    assert len(store) == 2
    assert store.ids == ["S2-1", "S2-2"]
    assert store.nums(0).tolist() == [vocab.get("10")]
    assert store.words(1).size == 0            # empty address -> no words
    assert full[1] >= 0                        # still gets a whole-name key
    assert df.sum() > 0


def test_index_drops_oversized_postings():
    keys = np.array([1, 1, 1, 1, 2], dtype=np.uint64)
    vals = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    fams = np.array([1, 1, 1, 1, 1], dtype=np.int8)
    idx = BlockIndex(keys, vals, fams)
    idx._drop_oversized(max_posting=3)
    assert idx.keys.tolist() == [2]            # the 4-long posting list is gone
    lo, hi = idx.lookup(np.uint64(2))
    assert (lo, hi) == (0, 1)


def test_lookup_miss_returns_empty_range():
    idx = BlockIndex(np.array([5], dtype=np.uint64), np.array([0], dtype=np.int32),
                     np.array([1], dtype=np.int8))
    lo, hi = idx.lookup(np.uint64(99))
    assert lo == hi


# ---------------------------------------------------------------- end to end

def _shard():
    s1 = [("S1-1", "Rapid Reit LLC", "6519 17th Avenue, Seattle, WA"),
          ("S1-2", "Proud Hotels Pvt Ltd", "Mig 151, Sreesailam, Ernakulam"),
          ("S1-3", "Zephay Labs Inc", "2621 Cotten Road, Tyler, TX")]
    pool = [
        ("S2-A", "RAPID REIT", "6519 17RD AVE, SEATTLE, WA"),           # -> S1-1
        ("S2-B", "Rapid Rti LLC", "6519 17RD AVE, SEATTLE, WA"),        # -> S1-1 (typo)
        ("S3-C", "Rapid Reit LLC", ""),                                 # -> S1-1 (no addr)
        ("S3-D", "pvt. pr0ud hotels ltd.", "Mig 151, Ernakulam"),       # -> S1-2 (homoglyph)
        ("S2-E", "Zephay Labs", "2621 COTTEN RD, TYLER, TX"),           # -> S1-3
        ("S2-F", "Completely Different Co", "99 Nowhere Ln, Fargo, ND"),  # distractor
    ]
    return build_shard("US", s1, pool, RULES, max_posting=1000)


def test_end_to_end_retrieval_finds_noisy_true_matches():
    sh = _shard()
    p = Prober(sh, RULES)
    got = {sh.s1.ids[i]: {sh.pool.ids[j] for j, _ in p.candidates(i, 10)}
           for i in range(len(sh.s1))}
    assert {"S2-A", "S2-B"} <= got["S1-1"]     # abbreviation + typo variants
    assert "S3-D" in got["S1-2"]               # homoglyph "pr0ud" -> "proud"
    assert "S2-E" in got["S1-3"]


def test_empty_address_record_is_reachable_by_name_alone():
    sh = _shard()
    p = Prober(sh, RULES)
    i = sh.s1.ids.index("S1-1")
    assert "S3-C" in {sh.pool.ids[j] for j, _ in p.candidates(i, 10)}


def test_top_k_is_respected_and_ranked():
    sh = _shard()
    p = Prober(sh, RULES)
    res = p.candidates(0, 2)
    assert len(res) <= 2
    scores = [s for _, s in res]
    assert scores == sorted(scores, reverse=True)


def test_candidates_never_returns_duplicates():
    sh = _shard()
    p = Prober(sh, RULES)
    for i in range(len(sh.s1)):
        idxs = [j for j, _ in p.candidates(i, 50)]
        assert len(idxs) == len(set(idxs))
