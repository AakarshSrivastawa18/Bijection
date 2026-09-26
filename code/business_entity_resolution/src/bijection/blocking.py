"""Phase 2: candidate generation.

The brute-force space is 1,732,544 Source-1 test entities x 9,969,589 Source-2/3
records = 1.73e13 pairs. This module reduces that to ~20 candidates per entity.

Design
------
1. **Country is a hard partition.** Verified exhaustively: across all 7,638,365
   ground-truth matched pairs, zero cross a country boundary. Sharding by country
   is therefore free recall and cuts the space ~3x before anything else runs.

2. **Conjunctive keys, not single tokens.** A single shared token ("pacific",
   "bordeaux") has a posting list in the tens of thousands - walking it per entity
   is hopeless. Every key here is a *pair* of signals (name token + address word,
   house number + address word, ...), which keeps posting lists at single or double
   digits while preserving the 99.99% union recall ceiling measured on real pairs.

3. **Expansion at probe time only.** Mined alternates are symmetric, so expanding
   just the query side catches both directions ("st" indexed / "street" probed and
   vice versa) at half the index size.

4. **Score during the walk.** Candidates are ranked by accumulated key weight as the
   posting lists are traversed - no per-pair set intersection, which at ~100 raw
   candidates x 1.73M entities would dominate runtime.

Memory
------
Records are held as CSR-style flat numpy arrays of interned token ids rather than
Python lists of strings: ~40 bytes/record instead of ~300, which is what makes a
4.7M-record shard fit alongside everything else in 16 GB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .mine_rules import RuleSet
from .normalize import norm_text, skeletonize, split_numeric, tokenize

_MASK = (1 << 63) - 1
# Distinct odd multipliers per key slot so that ("a","b") and ("b","a") differ and
# key families never collide with one another.
_P_KIND = 0x9E3779B185EBCA87
_P_A = 0xC2B2AE3D27D4EB4F
_P_B = 0x165667B19E3779F9


def _mix(kind: int, a: int, b: int) -> int:
    """Deterministic 63-bit key hash.

    Inputs are cast with int() because token ids arrive as numpy int64, and
    numpy int64 * a 64-bit Python constant raises OverflowError rather than
    wrapping. Every product is masked to keep the arithmetic in fixed width.
    """
    h = ((int(kind) * _P_KIND) & _MASK) ^ ((int(a) * _P_A) & _MASK) ^ ((int(b) * _P_B) & _MASK)
    h ^= (h >> 31)
    h = (h * 0xFF51AFD7ED558CCD) & _MASK
    h ^= (h >> 29)
    return h & _MASK


# key families
K_NAME_ADDR = 1     # name token + address word
K_NUM_ADDR = 2      # house number + address word
K_NAME_NUM = 3      # name token + house number
K_NAME_PAIR = 4     # two name tokens
K_EXACT_NAME = 5    # whole normalized name (rescues empty-address records)

# Relative trust of each family, used to rank candidates during the probe walk.
KEY_WEIGHT = {
    K_NAME_ADDR: 3.0,
    K_NUM_ADDR: 3.5,     # house number + street word is the most specific signal
    K_NAME_NUM: 3.0,
    K_NAME_PAIR: 2.0,
    K_EXACT_NAME: 4.0,
}

MAX_NAME_TOKENS = 3
MAX_ADDR_WORDS = 3
MAX_NUMS = 3


class Vocab:
    """String -> int interning table."""

    def __init__(self) -> None:
        self._d: Dict[str, int] = {}

    def add(self, tok: str) -> int:
        i = self._d.get(tok)
        if i is None:
            i = len(self._d)
            self._d[tok] = i
        return i

    def get(self, tok: str) -> int:
        return self._d.get(tok, -1)

    def __len__(self) -> int:
        return len(self._d)


@dataclass
class RecordStore:
    """CSR-packed token ids for one shard."""

    ids: List[str] = field(default_factory=list)
    # Conservative normalized text (accents folded, depunctuated, lowercased) kept
    # for the string-similarity features. Deliberately NOT the skeleton form: the
    # confusable fold is lossy, and the matcher needs the residual character
    # evidence the blocker was happy to throw away.
    norm_name: List[str] = field(default_factory=list)
    norm_addr: List[str] = field(default_factory=list)
    name_tok: np.ndarray = field(default=None)
    name_off: np.ndarray = field(default=None)
    word_tok: np.ndarray = field(default=None)
    word_off: np.ndarray = field(default=None)
    num_tok: np.ndarray = field(default=None)
    num_off: np.ndarray = field(default=None)

    def __len__(self) -> int:
        return len(self.ids)

    def names(self, i: int) -> np.ndarray:
        return self.name_tok[self.name_off[i]:self.name_off[i + 1]]

    def words(self, i: int) -> np.ndarray:
        return self.word_tok[self.word_off[i]:self.word_off[i + 1]]

    def nums(self, i: int) -> np.ndarray:
        return self.num_tok[self.num_off[i]:self.num_off[i + 1]]


def prepare(name: str, addr: str, rules: RuleSet
            ) -> Tuple[List[str], List[str], List[str], str, str]:
    """Full Phase-1 chain for one record.

    Returns (name tokens, address words, address numbers, norm name, norm address).
    The first three are skeletonised and feed blocking; the last two are the
    conservative forms and feed the matcher's similarity features.
    """
    nn = norm_text(name)
    na = norm_text(addr)

    n = tokenize(nn)
    n = rules.transliterate(n)          # nine Indic scripts -> Latin
    n = skeletonize(n)                  # homoglyph noise -> canonical shape
    n = rules.strip_droppable(n)        # legal suffixes, DBA markers

    a = tokenize(na)
    a = rules.apply_phrases(a)          # "new york" -> "ny"
    a = skeletonize(a)
    words, nums = split_numeric(a)
    # Transliterated names are more useful to the matcher than raw Indic script,
    # which shares no characters with the Latin Source-1 side.
    if n and nn and any(ord(c) >= 0x0900 for c in nn):
        nn = " ".join(n)
    return n, words, nums, nn, na


def build_store(records: Iterable[Tuple[str, str, str]], rules: RuleSet,
                vocab: Vocab, name_vocab: Vocab
                ) -> Tuple[RecordStore, np.ndarray, np.ndarray]:
    """Tokenize a shard into CSR arrays.

    Returns (store, token document-frequency, whole-name ids). `name_vocab` is kept
    separate from the token vocab so that full-name strings never pollute the
    document-frequency statistics used for IDF ranking.
    """
    ids: List[str] = []
    nt: List[int] = []
    no: List[int] = [0]
    wt: List[int] = []
    wo: List[int] = [0]
    dt: List[int] = []
    do: List[int] = [0]
    df: Dict[int, int] = {}
    full: List[int] = []
    nnames: List[str] = []
    naddrs: List[str] = []

    for eid, name, addr in records:
        n, w, d, nn, na = prepare(name, addr, rules)
        ids.append(eid)
        nnames.append(nn)
        naddrs.append(na)
        full.append(name_vocab.add(" ".join(sorted(n))) if n else -1)
        seen = set()
        for tok in n:
            i = vocab.add(tok)
            nt.append(i)
            seen.add(i)
        for tok in w:
            i = vocab.add(tok)
            wt.append(i)
            seen.add(i)
        for tok in d:
            i = vocab.add(tok)
            dt.append(i)
            seen.add(i)
        for i in seen:
            df[i] = df.get(i, 0) + 1
        no.append(len(nt))
        wo.append(len(wt))
        do.append(len(dt))

    store = RecordStore(
        ids=ids, norm_name=nnames, norm_addr=naddrs,
        name_tok=np.asarray(nt, dtype=np.int32), name_off=np.asarray(no, dtype=np.int64),
        word_tok=np.asarray(wt, dtype=np.int32), word_off=np.asarray(wo, dtype=np.int64),
        num_tok=np.asarray(dt, dtype=np.int32), num_off=np.asarray(do, dtype=np.int64),
    )
    dfa = np.zeros(len(vocab), dtype=np.int64)
    for i, c in df.items():
        dfa[i] = c
    return store, dfa, np.asarray(full, dtype=np.int64)


def _top_by_idf(toks: np.ndarray, df: np.ndarray, k: int) -> List[int]:
    """Most discriminative k tokens (lowest document frequency), deduplicated."""
    if toks.size == 0:
        return []
    uniq = np.unique(toks)
    if uniq.size <= k:
        return [int(x) for x in uniq]
    d = np.where(uniq < df.size, df[np.minimum(uniq, df.size - 1)], 1)
    order = np.argsort(d, kind="stable")[:k]
    return [int(uniq[i]) for i in order]


def record_keys(names: Sequence[int], words: Sequence[int], nums: Sequence[int],
                name_full: int) -> List[Tuple[int, int, int, int]]:
    """(key_hash, family, token_a, token_b) for one record.

    The constituent token ids are returned alongside the hash so the probe side can
    weight each key by the IDF of what it actually matched on. Ranking purely by
    key family measured 95.0% recall@20 against 97.8% uncapped - i.e. nearly three
    points were being lost to bad ordering rather than to missing candidates.
    """
    out: List[Tuple[int, int, int, int]] = []
    for n in names:
        for w in words:
            out.append((_mix(K_NAME_ADDR, n, w), K_NAME_ADDR, n, w))
        for d in nums:
            out.append((_mix(K_NAME_NUM, n, d), K_NAME_NUM, n, d))
    for d in nums:
        for w in words:
            out.append((_mix(K_NUM_ADDR, d, w), K_NUM_ADDR, d, w))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = sorted((names[i], names[j]))
            out.append((_mix(K_NAME_PAIR, a, b), K_NAME_PAIR, a, b))
    if name_full >= 0:
        out.append((_mix(K_EXACT_NAME, name_full, 0), K_EXACT_NAME, -1, -1))
    return out


class BlockIndex:
    """Sorted (key -> record index) table supporting binary-search probes."""

    def __init__(self, keys: np.ndarray, vals: np.ndarray, fams: np.ndarray) -> None:
        order = np.argsort(keys, kind="stable")
        self.keys = keys[order]
        self.vals = vals[order]
        self.fams = fams[order]

    @classmethod
    def build(cls, store: RecordStore, df: np.ndarray, vocab: Vocab,
              full_name_ids: Sequence[int], max_posting: int) -> "BlockIndex":
        keys: List[int] = []
        vals: List[int] = []
        fams: List[int] = []
        for i in range(len(store)):
            names = _top_by_idf(store.names(i), df, MAX_NAME_TOKENS)
            words = _top_by_idf(store.words(i), df, MAX_ADDR_WORDS)
            nums = [int(x) for x in np.unique(store.nums(i))[:MAX_NUMS]]
            for k, fam, _a, _b in record_keys(names, words, nums, full_name_ids[i]):
                keys.append(k)
                vals.append(i)
                fams.append(fam)
        idx = cls(np.asarray(keys, dtype=np.uint64),
                  np.asarray(vals, dtype=np.int32),
                  np.asarray(fams, dtype=np.int8))
        idx._drop_oversized(max_posting)
        return idx

    def _drop_oversized(self, max_posting: int) -> None:
        """Discard keys whose posting list is so long they carry no information.

        These are near-stopword conjunctions; walking them costs far more than the
        recall they add, and they are exactly the keys that blow up candidate counts.
        """
        if self.keys.size == 0:
            return
        uniq, counts = np.unique(self.keys, return_counts=True)
        bad = uniq[counts > max_posting]
        if bad.size:
            keep = ~np.isin(self.keys, bad)
            self.keys = self.keys[keep]
            self.vals = self.vals[keep]
            self.fams = self.fams[keep]

    def lookup(self, key: int) -> Tuple[int, int]:
        lo = int(np.searchsorted(self.keys, key, side="left"))
        hi = int(np.searchsorted(self.keys, key, side="right"))
        return lo, hi


__all__ = [
    "Vocab", "RecordStore", "BlockIndex", "prepare", "build_store", "record_keys",
    "KEY_WEIGHT", "K_NAME_ADDR", "K_NUM_ADDR", "K_NAME_NUM", "K_NAME_PAIR",
    "K_EXACT_NAME", "MAX_NAME_TOKENS", "MAX_ADDR_WORDS", "MAX_NUMS",
]
