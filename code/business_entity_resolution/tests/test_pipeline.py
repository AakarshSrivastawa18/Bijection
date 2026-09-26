"""End-to-end integration test on a synthetic dataset.

Runs the real pipeline — mine rules, block, featurise, train, score, decide, write —
against a tiny generated corpus, then checks the two output files against every rule
the official validator enforces. This is what catches format and plumbing bugs
before committing to a multi-hour run on the full test set.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import pytest

from bijection import config, model as M
from bijection.candidates import build_shard
from bijection.dataset import build_matrix
from bijection.decide import assign
from bijection.io_utils import read_records, write_id_lists
from bijection.metrics import score_matching
from bijection.mine_rules import RuleSet, mine

# --- generator mimicking the observed noise: case, abbreviation, homoglyph,
# --- suffix drift, token addition, empty address.
SUFFIX = ["LLC", "Inc", "Corp", "Ltd", "Pvt Ltd", "LLP"]
WORDS = ["rapid", "pacific", "zephay", "orelee", "vinayak", "proud", "harper",
         "alpha", "summit", "crescent", "vertex", "lumen", "arbor", "quill"]
STREET = ["Avenue", "Street", "Road", "Lane", "Boulevard"]
ABBR = {"Avenue": "AVE", "Street": "ST", "Road": "RD", "Lane": "LN",
        "Boulevard": "BLVD"}
CITY = ["Seattle", "Tyler", "Fargo", "Amarillo", "Maywood", "Canandaigua"]
LEET = {"s": "5", "o": "0", "i": "l", "a": "4"}


def _noisy(s: str, rng: random.Random) -> str:
    if rng.random() < 0.3:
        for a, b in LEET.items():
            if a in s.lower() and rng.random() < 0.4:
                i = s.lower().index(a)
                s = s[:i] + b + s[i + 1:]
                break
    return s


def _make_corpus(tmp: Path, n: int = 260, seed: int = 5):
    rng = random.Random(seed)
    s1, s2, s3, gt = [], [], [], []
    for i in range(n):
        base = f"{rng.choice(WORDS).title()} {rng.choice(WORDS).title()}"
        suf = rng.choice(SUFFIX)
        num = rng.randint(100, 9999)
        st = rng.choice(STREET)
        city = rng.choice(CITY)
        name = f"{base} {suf}"
        addr = f"{num} {rng.choice(WORDS).title()} {st}, {city}, WA"
        sid = f"S1-{i:05d}"
        s1.append((sid, name, addr, "US"))

        matches = []
        if rng.random() < 0.88:                      # ~12% singletons
            for j in range(rng.randint(1, 3)):
                mid = f"S2-{i:05d}{j}"
                s2.append((mid, _noisy(name.upper(), rng),
                           f"{num} {rng.choice(WORDS).upper()} {ABBR[st]}, "
                           f"{city.upper()}, WA", "US"))
                matches.append(mid)
            for j in range(rng.randint(0, 2)):
                mid = f"S3-{i:05d}{j}"
                blank = rng.random() < 0.12
                s3.append((mid, _noisy(f"{base} {rng.choice(SUFFIX)}", rng),
                           "" if blank else f"{num} {st}, {city}, Washington", "US"))
                matches.append(mid)
        gt.append((sid, ",".join(matches)))

    # distractors that share a city but nothing else
    for k in range(n // 2):
        s2.append((f"S2-D{k:05d}", f"{rng.choice(WORDS).title()} Holdings",
                   f"{rng.randint(100,9999)} Other St, {rng.choice(CITY)}, WA", "US"))

    d = tmp / "train"
    d.mkdir(parents=True, exist_ok=True)
    for fn, rows in (("train_source1.tsv", s1), ("train_source2.tsv", s2),
                     ("train_source3.tsv", s3)):
        with open(d / fn, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(["entity_id", "business_name", "business_address", "country"])
            w.writerows(rows)
    with open(d / "train_ground_truth.tsv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["source1_entity_id", "matched_entity_ids"])
        w.writerows(gt)
    return dict(gt), s1, s2, s3


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bij")
    gt_raw, s1, s2, s3 = _make_corpus(tmp)
    gt = {k: ([] if not v else v.split(",")) for k, v in gt_raw.items()}

    # point the package at the synthetic corpus
    old_data, old_work = config.DATA_DIR, config.WORK_DIR
    config.DATA_DIR = tmp
    config.WORK_DIR = tmp / "work"
    config.TRAIN.update({
        "s1": tmp / "train" / "train_source1.tsv",
        "s2": tmp / "train" / "train_source2.tsv",
        "s3": tmp / "train" / "train_source3.tsv",
        "gt": tmp / "train" / "train_ground_truth.tsv",
    })
    try:
        tables = mine(out_dir=config.WORK_DIR / "rules")
        rules = RuleSet.load(config.WORK_DIR / "rules")
        shard = build_shard("US", [(a, b, c) for a, b, c, _ in s1],
                            [(a, b, c) for a, b, c, _ in s2 + s3], rules,
                            max_posting=5000)
        X, y, row_s1, row_cand = build_matrix(shard, rules, config.TOP_K, gt)
        yield dict(tmp=tmp, gt=gt, shard=shard, rules=rules, X=X, y=y,
                   row_s1=row_s1, row_cand=row_cand, tables=tables,
                   s1=s1, s2=s2, s3=s3)
    finally:
        config.DATA_DIR, config.WORK_DIR = old_data, old_work


def test_rule_mining_runs_on_a_fresh_corpus(built):
    """Mining must not require hand-authored tables to bootstrap."""
    t = built["tables"]
    assert isinstance(t["droppable"], list)
    assert isinstance(t["expansions"], dict)


def test_blocking_reaches_most_true_matches(built):
    cand = {}
    for s, c in zip(built["row_s1"], built["row_cand"]):
        cand.setdefault(s, []).append(c)
    total = sum(len(v) for v in built["gt"].values())
    found = sum(len(set(v) & set(cand.get(k, []))) for k, v in built["gt"].items())
    assert total > 0
    assert found / total > 0.80


def test_feature_matrix_is_finite_and_labelled(built):
    X, y = built["X"], built["y"]
    assert X.shape[0] == len(y) > 0
    assert np.isfinite(X).all()
    assert 0 < y.sum() < len(y)          # both classes present


def test_model_trains_and_separates_classes(built):
    X, y = built["X"], built["y"]
    b = M.train(X, y, num_round=60)
    p = b.predict(X)
    assert p[y == 1].mean() > p[y == 0].mean() + 0.2


def test_end_to_end_outputs_pass_validator_rules(built, tmp_path):
    """Every rule utils/validate_submission.py enforces, checked here."""
    X, y = built["X"], built["y"]
    booster = M.train(X, y, num_round=60)
    scores = booster.predict(X)
    triples = list(zip(built["row_s1"], built["row_cand"], scores.tolist()))

    all_s1 = [r[0] for r in built["s1"]]
    matches = assign(triples, 0.5, all_s1, one_to_one=True)

    cand_map = {}
    for s, c in zip(built["row_s1"], built["row_cand"]):
        cand_map.setdefault(s, []).append(c)

    out = tmp_path / "out"
    write_id_lists(out / "matching_results.tsv",
                   ((e, matches.get(e, [])) for e in all_s1),
                   ("source1_entity_id", "matched_entity_ids"))
    write_id_lists(out / "candidate_pairs.tsv",
                   ((e, cand_map.get(e, [])) for e in all_s1),
                   ("source1_entity_id", "candidate_entity_ids"))

    valid_ids = {r[0] for r in built["s2"]} | {r[0] for r in built["s3"]}
    required = set(all_s1)

    for fn, col in (("matching_results.tsv", "matched_entity_ids"),
                    ("candidate_pairs.tsv", "candidate_entity_ids")):
        with open(out / fn, encoding="utf-8") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            assert header == ["source1_entity_id", col]
            seen = set()
            rows = {}
            for line in fh:
                s1id, tab, rest = line.partition("\t")
                assert tab, "every row must contain a tab"
                assert s1id not in seen, "duplicate source1_entity_id row"
                seen.add(s1id)
                ids = rest.rstrip("\n").split(",") if rest.strip() else []
                assert len(ids) == len(set(ids)), "duplicate id inside a list"
                for i in ids:
                    assert i.startswith(("S2-", "S3-")), f"bad prefix {i}"
                    assert i in valid_ids, f"id not in test set: {i}"
                rows[s1id] = set(ids)
            assert seen == required, "one row per Source-1 entity, no extras"
            built[fn] = rows

    # matches must be a subset of candidates - a match that was never a candidate
    # signals a pipeline bug, and the official validator warns about it.
    for e, ids in built["matching_results.tsv"].items():
        assert ids <= built["candidate_pairs.tsv"][e]


def test_scores_beat_predicting_everything_and_nothing(built):
    """Sanity floor: the tuned pipeline must beat both trivial baselines."""
    X, y = built["X"], built["y"]
    booster = M.train(X, y, num_round=60)
    scores = booster.predict(X)
    triples = list(zip(built["row_s1"], built["row_cand"], scores.tolist()))
    all_s1 = [r[0] for r in built["s1"]]
    gt = built["gt"]

    empty = score_matching({e: [] for e in all_s1}, gt).macro_f05
    everything = score_matching(
        {e: [c for s, c in zip(built["row_s1"], built["row_cand"]) if s == e]
         for e in all_s1}, gt).macro_f05
    tuned = score_matching(assign(triples, 0.5, all_s1), gt).macro_f05
    assert tuned > empty and tuned > everything
