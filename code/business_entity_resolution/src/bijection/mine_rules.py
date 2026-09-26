"""Phase 1b: mine every substitution table from the training ground truth.

The challenge forbids external data lookup, which rules out off-the-shelf gazetteers
for US state codes, street-type abbreviations, French department names, and Hindi
transliteration. That constraint is also the right engineering call here: the tables
this module learns are specific to *this generator's* noise, and the same procedure
runs unchanged on a country it has never seen.

Four tables are produced, all from aligned (Source-1, matched Source-2/3) pairs:

``droppable``   tokens whose presence on one side only still left a true match -
                legal suffixes (llc, ltd, pvt, sarl, sasu) and generic filler.
``expansions``  token -> alternative surface forms (st -> street, saint; ny -> new york).
                Applied as *expansion* at blocking time, never as rewriting: "st"
                genuinely means both "street" and "saint" (St.-Herblain), so collapsing
                it to one canonical form would lose matches.
``phrases``     multi-token form -> single-token form (("new","york") -> "ny").
``deva``        Devanagari token -> Latin token, aligned positionally.

Nothing here is hand-authored. Run ``python -m bijection.mine_rules`` to rebuild.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from . import config
from .io_utils import read_ground_truth, read_records
from .normalize import is_non_latin, norm_text, tokenize


def _is_subsequence(short: str, long: str) -> bool:
    """True if every character of `short` appears in `long` in order.

    This is the defining property of an abbreviation: st<=street, st<=saint,
    tx<=texas, rd<=road, mh<=maharashtra. It correctly rejects the junk pairs that
    a bare co-occurrence count produces, such as dr<->ltd.
    """
    it = iter(long)
    return all(c in it for c in short)


def _related(a: str, b: str) -> bool:
    """Keep a mined 1<->1 substitution only if the two tokens are plausibly variants."""
    if a == b:
        return False
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    if _is_subsequence(short, long):
        return True
    # transposition/typo variants of similar length (road/raod/roda)
    if abs(len(a) - len(b)) <= 2:
        from rapidfuzz.distance import Levenshtein
        return Levenshtein.distance(a, b) <= 2 and min(len(a), len(b)) >= 4
    return False


def _is_acronym(single: str, multi: Tuple[str, ...]) -> bool:
    """True if `single` is the initialism of `multi` (new york -> ny)."""
    if len(single) != len(multi):
        return False
    return all(w.startswith(c) for c, w in zip(single, multi))

# A substitution must be seen this many times before it is trusted. The generator
# applies abbreviations systematically, so genuine rules appear thousands of times
# while coincidences appear once or twice.
MIN_SUBST_COUNT = 40
MIN_DROP_COUNT = 200
MIN_DEVA_COUNT = 15
SAMPLE_PAIRS = 300_000


def _load_sample(seed: int = config.RANDOM_SEED) -> Tuple[Dict[str, tuple], List[Tuple[str, List[str]]]]:
    rng = random.Random(seed)
    rows = [(s1, ids) for s1, ids in read_ground_truth(config.TRAIN["gt"]) if ids]
    if len(rows) > SAMPLE_PAIRS:
        rows = rng.sample(rows, SAMPLE_PAIRS)
    need: Set[str] = set()
    for s1, ids in rows:
        need.add(s1)
        need.update(ids)
    recs: Dict[str, tuple] = {}
    for key in ("s1", "s2", "s3"):
        for eid, name, addr, country in read_records(config.TRAIN[key]):
            if eid in need:
                recs[eid] = (name, addr, country)
    return recs, rows


def _diff(a: List[str], b: List[str]) -> Tuple[List[str], List[str]]:
    """Tokens unique to each side, in their ORIGINAL order.

    Order matters: sorting here turned "uttar pradesh -> up" into the unusable
    "pradesh uttar -> up" and broke every acronym rule downstream.
    """
    sa, sb = set(a), set(b)
    seen_a, seen_b = [], []
    for t in a:
        if t not in sb and t not in seen_a:
            seen_a.append(t)
    for t in b:
        if t not in sa and t not in seen_b:
            seen_b.append(t)
    return seen_a, seen_b


def mine(out_dir: Path | None = None, seed: int = config.RANDOM_SEED) -> Dict[str, object]:
    out_dir = out_dir or (config.WORK_DIR / "rules")
    out_dir.mkdir(parents=True, exist_ok=True)
    recs, rows = _load_sample(seed)

    subst: Counter = Counter()        # (tok_a, tok_b) sorted -> count
    phrase: Counter = Counter()       # (single, (w1, w2)) -> count
    drop: Counter = Counter()         # NAME token -> times it was a sole extra token
    seen_tok: Counter = Counter()     # name token -> total occurrences (drop ratio)
    deva: defaultdict = defaultdict(Counter)

    for s1, ids in rows:
        if s1 not in recs:
            continue
        n1, a1, _ = recs[s1]
        n1t, a1t = tokenize(norm_text(n1)), tokenize(norm_text(a1))
        for t in set(n1t):
            seen_tok[t] += 1
        for x in ids:
            if x not in recs:
                continue
            n2, a2, _ = recs[x]
            # ---- transliteration alignment (name only, positional, equal length).
            # Covers all nine Indic scripts present in the data, not just Devanagari.
            if is_non_latin(n2):
                d_toks = tokenize(norm_text(n2))
                if len(d_toks) == len(n1t) and d_toks:
                    for d, l in zip(d_toks, n1t):
                        if is_non_latin(d) and not is_non_latin(l):
                            deva[d][l] += 1
                continue
            n2t, a2t = tokenize(norm_text(n2)), tokenize(norm_text(a2))
            for field_a, field_b, is_name in ((n1t, n2t, True), (a1t, a2t, False)):
                if not field_a or not field_b:
                    continue
                only_a, only_b = _diff(field_a, field_b)
                # 1 <-> 1 : abbreviation or spelling variant
                if len(only_a) == 1 and len(only_b) == 1:
                    x_, y_ = sorted((only_a[0], only_b[0]))
                    subst[(x_, y_)] += 1
                # 1 <-> 2 : short form vs multi-word form (new york <-> ny)
                elif len(only_a) == 1 and len(only_b) == 2:
                    phrase[(only_a[0], tuple(only_b))] += 1
                elif len(only_b) == 1 and len(only_a) == 2:
                    phrase[(only_b[0], tuple(only_a))] += 1
                # Pure additions on one side are droppable - but ONLY in the name
                # field. Mining this from addresses harvested city names (bangalore,
                # ernakulam, hyderabad...) because Source-3 often truncates an address
                # down to city+state; dropping the city would destroy the single
                # strongest geo anchor the blocker has.
                elif is_name and not only_a and only_b:
                    for t in only_b:
                        drop[t] += 1
                elif is_name and not only_b and only_a:
                    for t in only_a:
                        drop[t] += 1

    # ---- build expansion table (no transitive closure)
    # Union-find would chain "st -> street" with "st -> saint" into one class and
    # merge Street with Saint. Each token instead keeps an explicit set of alternates.
    expansions: defaultdict = defaultdict(set)
    for (x_, y_), c in subst.items():
        if c < MIN_SUBST_COUNT:
            continue
        if x_.isdigit() or y_.isdigit():
            continue          # never equate two different house numbers
        # Co-occurrence alone produced nonsense like dr<->ltd, which then dragged
        # unrelated records into the same block. Require an abbreviation or
        # typo relationship.
        if not _related(x_, y_):
            continue
        expansions[x_].add(y_)
        expansions[y_].add(x_)

    phrases: Dict[str, str] = {}
    for (single, multi), c in phrase.items():
        if c < MIN_SUBST_COUNT or single.isdigit():
            continue
        if any(w.isdigit() or is_non_latin(w) for w in multi):
            continue
        if is_non_latin(single):
            continue
        # Only genuine initialisms. Without this the table filled with artefacts
        # ("c l -> llc", "ltd pvt -> smt", "az unit -> arizona").
        if not _is_acronym(single, multi):
            continue
        phrases[" ".join(multi)] = single

    droppable = {
        t for t, c in drop.items()
        if c >= MIN_DROP_COUNT and not t.isdigit() and c / max(1, seen_tok[t]) > 0.02
    }

    deva_map = {
        d: cnt.most_common(1)[0][0]
        for d, cnt in deva.items()
        if cnt.most_common(1)[0][1] >= MIN_DEVA_COUNT
    }

    tables = {
        "droppable": sorted(droppable),
        "expansions": {k: sorted(v) for k, v in sorted(expansions.items())},
        "phrases": phrases,
        "deva": deva_map,
    }
    for name, obj in tables.items():
        (out_dir / f"{name}.json").write_text(
            json.dumps(obj, ensure_ascii=False, indent=0), encoding="utf-8"
        )
    return tables


# ---------------------------------------------------------------- apply side

class RuleSet:
    """Loaded tables plus the helpers that apply them."""

    def __init__(self, droppable: Set[str], expansions: Dict[str, List[str]],
                 phrases: Dict[str, str], deva: Dict[str, str]) -> None:
        self.droppable = droppable
        self.expansions = expansions
        self.phrases = phrases
        self.deva = deva

    @classmethod
    def load(cls, rule_dir: Path | None = None) -> "RuleSet":
        rule_dir = rule_dir or (config.WORK_DIR / "rules")
        def _read(n, default):
            p = rule_dir / f"{n}.json"
            return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default
        return cls(
            droppable=set(_read("droppable", [])),
            expansions=_read("expansions", {}),
            phrases=_read("phrases", {}),
            deva=_read("deva", {}),
        )

    def transliterate(self, tokens: List[str]) -> List[str]:
        """Replace Devanagari tokens with their mined Latin equivalents."""
        if not any(is_non_latin(t) for t in tokens):
            return tokens
        return [self.deva.get(t, t) for t in tokens]

    def strip_droppable(self, tokens: List[str]) -> List[str]:
        """Remove legal suffixes / filler. Never returns empty if input was non-empty."""
        kept = [t for t in tokens if t not in self.droppable]
        return kept if kept else tokens

    def expand(self, token: str) -> Set[str]:
        """Token plus every mined alternative surface form."""
        alts = self.expansions.get(token)
        return {token, *alts} if alts else {token}

    def apply_phrases(self, tokens: List[str]) -> List[str]:
        """Collapse mined multi-word forms ("new york" -> "ny") into the short form."""
        if not self.phrases or len(tokens) < 2:
            return tokens
        out: List[str] = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens):
                bg = tokens[i] + " " + tokens[i + 1]
                repl = self.phrases.get(bg)
                if repl is not None:
                    out.append(repl)
                    i += 2
                    continue
            out.append(tokens[i])
            i += 1
        return out


if __name__ == "__main__":  # pragma: no cover
    t = mine()
    print(f"droppable : {len(t['droppable'])}")
    print(f"expansions: {len(t['expansions'])}")
    print(f"phrases   : {len(t['phrases'])}")
    print(f"deva      : {len(t['deva'])}")
