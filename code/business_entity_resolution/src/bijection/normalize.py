"""Phase 1: text normalization.

Two normalized forms are produced for every field, because they serve different jobs:

* ``norm``     - conservative. Unicode-folded, lowercased, depunctuated. Used for
                 human-readable output and for the string-similarity features, where
                 destroying information would hurt.
* ``skeleton`` - aggressive. Additionally collapses visually-confusable characters
                 into a single representative (i/l/1/j -> i, o/0 -> o, s/5 -> s, ...).
                 Used for blocking keys and as a match feature, where recall matters
                 more than fidelity.

The skeleton form is what recovers the generator's homoglyph noise: "5herman" and
"Sherman" both skeletonise to "sherman"; "lndia" and "India" both to "india".

No external gazetteer is used anywhere in this module - every substitution table
applied downstream is mined from the training ground truth (see mine_rules.py).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Sequence, Set, Tuple

# ---------------------------------------------------------------- character folding

# Confusable classes. Each character on the left maps to the representative on the
# right. Digits that commonly stand in for letters are included; this is only ever
# applied to tokens that are predominantly alphabetic, so street numbers survive.
# Only digit<->letter and i/l/j shapes are collapsed. Letter pairs that are merely
# similar (q/g, u/v, m/rn) are deliberately left alone: the business-name vocabulary
# is small (69k distinct tokens over 500k Source-1 names), so over-collapsing
# manufactures collisions between genuinely different businesses.
_CONFUSABLE = {
    "l": "i", "1": "i", "j": "i",
    "0": "o",
    "5": "s",
    "3": "e",
    "4": "a",
    "8": "b",
    "9": "g", "6": "g",
    "7": "t",
    "2": "z",
}
_SKEL_TABLE = str.maketrans(_CONFUSABLE)

# Punctuation -> space. Ampersand is spelled out first (see _AMP) so "&" and "and"
# converge; the generator swaps them freely.
_PUNCT = "".join(chr(c) for c in range(32, 127) if not chr(c).isalnum())
# Common non-ASCII punctuation that survives NFKD and would otherwise stay glued
# to a token (French quotes/dashes, Devanagari danda).
_PUNCT += "‐‑‒–—―‘’“”«»·•।॥"
_PUNCT_TABLE = str.maketrans({c: " " for c in _PUNCT})

_WS = re.compile(r"\s+")
_AMP = re.compile(r"\s*&\s*")


# Invisible characters. U+200C (ZERO WIDTH NON-JOINER) occurs 14,079 times in the
# training Source-2/3 fields; it is legitimate Indic orthography but it silently
# breaks token equality, so it is removed rather than folded.
_INVISIBLE = dict.fromkeys([0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF, 0x00AD])


def fold_unicode(s: str) -> str:
    """Strip accents, compatibility forms and invisibles; keep non-Latin scripts.

    Indic scripts are left as-is here - their combining vowel signs carry meaning and
    must survive to reach the mined transliteration table. Only Latin diacritics
    (the generator's injected noise, plus genuine French accents) are flattened.
    """
    if s.isascii():
        return s
    s = unicodedata.normalize("NFKD", s).translate(_INVISIBLE)
    # U+0300-U+036F is the Combining Diacritical Marks block: exactly the Latin
    # accents NFKD just detached. Indic matras live at U+0900+ and are untouched.
    return "".join(c for c in s if not 0x0300 <= ord(c) <= 0x036F)


def norm_text(s: str) -> str:
    """Conservative normalization: fold, expand '&', depunctuate, lowercase, collapse."""
    if not s:
        return ""
    s = _AMP.sub(" and ", s)
    s = fold_unicode(s).translate(_PUNCT_TABLE).lower()
    return _WS.sub(" ", s).strip()


def _alpha_ratio(tok: str) -> float:
    if not tok:
        return 0.0
    return sum(c.isalpha() for c in tok) / len(tok)


def skeletonize_token(tok: str) -> str:
    """Collapse confusables, but only for tokens that are mostly letters.

    The guard is what stops "17th" becoming "itth" while still turning "pr0ud" into
    "proud". Pure-numeric tokens (house numbers, PIN codes) are never touched.
    """
    if not tok or tok.isdigit():
        return tok
    if _alpha_ratio(tok) < 0.66:
        return tok
    return tok.translate(_SKEL_TABLE)


def skeletonize(tokens: Iterable[str]) -> List[str]:
    return [skeletonize_token(t) for t in tokens]


# ---------------------------------------------------------------- tokenization

def tokenize(s: str) -> List[str]:
    """Split on punctuation and whitespace.

    Deliberately NOT a ``\\w+`` regex: Devanagari vowel signs (matras) are Unicode
    marks, not word characters, so ``[^\\W_]+`` shreds "राम" into "र" + "म" and
    silently destroys every Hindi name. Punctuation-to-space plus ``str.split`` is
    both correct here and faster - it runs ~23M times per full pass.
    """
    if not s:
        return []
    return s.translate(_PUNCT_TABLE).split()


def split_numeric(tokens: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Partition tokens into (word tokens, numeric tokens).

    Numeric tokens are the strongest address signal in this dataset - 79.9% of true
    matched pairs share at least one - so they are indexed separately from words.
    """
    # "3337-3339" ranges and "38/412" municipal numbers are already split on the
    # punctuation, so both of their halves land in `nums`. Ordinals like "17th"
    # keep their digits and stay in `words`.
    words: List[str] = []
    nums: List[str] = []
    for t in tokens:
        (nums if t.isdigit() else words).append(t)
    return words, nums


def is_non_latin(s: str) -> bool:
    """True if the string carries any non-Latin letter.

    The training data contains NINE Indic scripts, not just Devanagari: Telugu,
    Kannada, Tamil, Bengali, Gujarati, Malayalam, Oriya and Gurmukhi all appear in
    Source-2/3 India records. The test set's France records are pure Latin, so this
    predicate is what routes a record to the mined transliteration table.
    """
    return any(ord(c) >= 0x0900 and c.isalpha() for c in s)


# Retained under the old name for readability at call sites that specifically mean
# Hindi; behaviourally identical to is_non_latin for this dataset.
is_devanagari = is_non_latin


__all__ = [
    "fold_unicode", "norm_text", "skeletonize", "skeletonize_token",
    "tokenize", "split_numeric", "is_devanagari",
]
