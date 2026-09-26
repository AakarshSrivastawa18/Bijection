"""Phase 1 tests. Every case here is a real string observed in the dataset."""
from __future__ import annotations

from bijection.normalize import (
    fold_unicode, is_devanagari, norm_text, skeletonize_token, split_numeric, tokenize,
)


def test_accent_stripping_real_cases():
    # S2 injects diacritics into US names; France uses them legitimately.
    assert fold_unicode("BÍNGAMAN") == "BINGAMAN"
    assert fold_unicode("Àmicale") == "Amicale"
    assert fold_unicode("Établissements Dëleves") == "Etablissements Deleves"
    assert fold_unicode("çompagnie") == "compagnie"


def test_ampersand_and_punctuation_converge():
    assert norm_text("Harper, Bingaman & Guallpa LLC") == "harper bingaman and guallpa llc"
    assert norm_text("Harper Bingaman and Guallpa LLC") == "harper bingaman and guallpa llc"
    assert norm_text("Thermal & Fils SASU") == norm_text("Thermal and Fils SASU")


def test_skeleton_recovers_homoglyph_noise():
    # observed: "Jania 5herman, MD", "pvt. pr0ud hotels ltd.", "Vinayak (lndia)"
    assert skeletonize_token("5herman") == skeletonize_token("sherman")
    assert skeletonize_token("pr0ud") == skeletonize_token("proud")
    assert skeletonize_token("lndia") == skeletonize_token("india")


def test_skeleton_does_not_destroy_ordinals_or_house_numbers():
    # "6519 17th Avenue" must keep 17th recognisable and 6519 untouched.
    assert skeletonize_token("17th") == "17th"
    assert skeletonize_token("6519") == "6519"
    assert skeletonize_token("3337") == "3337"


def test_skeleton_leaves_distinct_letters_distinct():
    # q/g and u/v are NOT collapsed - over-collapsing invents false merges.
    assert skeletonize_token("quality") != skeletonize_token("guality")
    assert skeletonize_token("value") != skeletonize_token("valve")


def test_split_numeric_on_real_addresses():
    words, nums = split_numeric(tokenize(norm_text("6519 17RD AVE, PMB 9349, SEATTLE, WA")))
    assert nums == ["6519", "9349"]
    assert "ave" in words and "seattle" in words and "17rd" in words


def test_municipal_numbering_splits_both_halves():
    _, nums = split_numeric(tokenize(norm_text("MIG #151, SREESAILAM, 38/412")))
    assert nums == ["151", "38", "412"]


def test_devanagari_survives_tokenization():
    name = "राम मार्केटिंग प्राइवेट लिमिटेड"
    assert is_devanagari(name)
    assert len(tokenize(name)) == 4          # not shredded into characters
    assert is_devanagari(norm_text(name))    # and not stripped by folding
    assert not is_devanagari("Ram Marketing Private Limited")


def test_empty_and_whitespace_inputs():
    assert norm_text("") == ""
    assert norm_text("   ") == ""
    assert tokenize("") == []
    assert split_numeric([]) == ([], [])
