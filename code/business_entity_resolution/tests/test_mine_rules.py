"""Phase 1b tests - the mined-rule filters, and regressions for four real defects.

Each *_regression test below encodes a bad table entry that the first mining pass
actually produced against the real training data.
"""
from __future__ import annotations

from bijection.mine_rules import RuleSet, _diff, _is_acronym, _is_subsequence, _related


def test_subsequence_accepts_real_abbreviations():
    for short, long in [("st", "street"), ("st", "saint"), ("tx", "texas"),
                        ("rd", "road"), ("mh", "maharashtra"), ("ave", "avenue")]:
        assert _is_subsequence(short, long), (short, long)


def test_related_regression_dr_ltd():
    """First pass mined dr <-> ltd from bare co-occurrence; it must be rejected."""
    assert not _related("dr", "ltd")
    assert _related("dr", "drive")


def test_related_accepts_generator_typos():
    # The generator injects these; mining recovers them as free typo rules.
    assert _related("road", "raod")
    assert _related("road", "roda")


def test_related_rejects_short_unrelated_pairs():
    assert not _related("cat", "dog")
    assert not _related("llc", "inc")


def test_diff_preserves_order_regression():
    """Sorting the diff produced 'pradesh uttar -> up' and broke every acronym."""
    a, b = _diff(["uttar", "pradesh", "road"], ["road"])
    assert a == ["uttar", "pradesh"]      # original order, not alphabetical
    assert b == []


def test_diff_deduplicates():
    a, b = _diff(["x", "x", "y"], ["y"])
    assert a == ["x"] and b == []


def test_acronym_filter():
    assert _is_acronym("ny", ("new", "york"))
    assert _is_acronym("up", ("uttar", "pradesh"))
    assert not _is_acronym("llc", ("c", "l"))        # regression: wrong length
    assert not _is_acronym("smt", ("ltd", "pvt"))    # regression: not initials
    assert not _is_acronym("arizona", ("az", "unit"))


def test_ruleset_expand_keeps_both_senses_of_st():
    rs = RuleSet(droppable=set(), expansions={"st": ["street", "saint"]},
                 phrases={}, deva={})
    assert rs.expand("st") == {"st", "street", "saint"}
    assert rs.expand("unknown") == {"unknown"}


def test_ruleset_strip_droppable_never_empties_a_name():
    rs = RuleSet(droppable={"llc", "inc"}, expansions={}, phrases={}, deva={})
    assert rs.strip_droppable(["rapid", "reit", "llc"]) == ["rapid", "reit"]
    # A name made entirely of suffixes must keep its tokens, not vanish.
    assert rs.strip_droppable(["llc", "inc"]) == ["llc", "inc"]


def test_ruleset_phrase_collapse():
    rs = RuleSet(droppable=set(), expansions={}, phrases={"new york": "ny"}, deva={})
    assert rs.apply_phrases(["123", "main", "new", "york"]) == ["123", "main", "ny"]
    assert rs.apply_phrases(["new"]) == ["new"]


def test_ruleset_transliterates_multiple_scripts():
    rs = RuleSet(droppable=set(), expansions={}, phrases={},
                 deva={"लिमिटेड": "limited", "রাম": "ram", "లక్ష్మీ": "lakshmi"})
    assert rs.transliterate(["शिवा", "लिमिटेड"]) == ["शिवा", "limited"]
    assert rs.transliterate(["রাম"]) == ["ram"]
    assert rs.transliterate(["లక్ష్మీ"]) == ["lakshmi"]
    assert rs.transliterate(["already", "latin"]) == ["already", "latin"]
