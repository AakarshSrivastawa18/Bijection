"""Train the matcher and evaluate it under the competition metric.

    python -m bijection.train --train work/ds_US_80000.npz work/ds_India_60000.npz

Two things are reported, and they answer different questions:

* **Held-out validation per country** - entities are split by a deterministic hash,
  so the same entity always lands on the same side. This is the in-domain number.

* **Leave-one-country-out (LOCO)** - train on US only, score India (and vice versa).
  The test set contains France, which has ZERO training rows, so LOCO is the only
  honest estimate of what happens on it. The in-domain number would flatter the
  pipeline badly.

The final model trains on every available country, because India alone is 47% of
the test Source-1 entities and leaving it out of training costs far more than the
LOCO diagnostic is worth.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import config, model as M
from .decide import Triple, assign, tune_threshold
from .metrics import score_matching  # noqa: F401  (used in _eval and rescue sweep)


class Split:
    """One country's feature matrix plus its ground truth."""

    def __init__(self, path: str) -> None:
        z = np.load(path, allow_pickle=False)
        self.name = Path(path).stem.replace("ds_", "").split("_")[0]
        self.X = z["X"]
        self.y = z["y"]
        self.row_s1 = z["row_s1"].tolist()
        self.row_cand = z["row_cand"].tolist()
        self.s1_ids = z["s1_ids"].tolist()
        self.gt = {k: ([] if not v else v.split(","))
                   for k, v in zip(z["gt_keys"].tolist(), z["gt_vals"].tolist())}
        self.tr_ent, self.va_ent = _split_entities(self.s1_ids)
        self.tr_mask = np.fromiter((s in self.tr_ent for s in self.row_s1),
                                   bool, len(self.row_s1))
        self.va_mask = ~self.tr_mask

    def rows(self, mask) -> Tuple[List[str], List[str]]:
        return ([s for s, m in zip(self.row_s1, mask) if m],
                [c for c, m in zip(self.row_cand, mask) if m])


def _split_entities(ids: Sequence[str], frac: float = 0.25) -> Tuple[set, set]:
    val = {e for e in ids
           if int(hashlib.blake2b(e.encode(), digest_size=8).hexdigest(), 16) % 100
           < frac * 100}
    return set(ids) - val, val


def _eval(booster, sp: Split, mask, ent: set, thr: float, label: str,
          ablate: bool = False) -> float:
    if not mask.any():
        return 0.0
    scores = booster.predict(sp.X[mask], num_iteration=booster.best_iteration)
    s1s, cds = sp.rows(mask)
    tri: List[Triple] = [(a, b, float(p)) for a, b, p in zip(s1s, cds, scores)]
    sub_gt = {e: sp.gt.get(e, []) for e in ent}
    ids = sorted(ent)

    pred = assign(tri, thr, ids, one_to_one=True)
    s = score_matching(pred, sub_gt)
    line = (f"  {label:22s} F0.5={s.macro_f05:.5f}  P={s.mean_precision:.4f} "
            f"R={s.mean_recall:.4f}  exact={100*s.exact_sets/max(1,s.n_entities):.1f}%")
    if ablate:
        base = score_matching(assign(tri, thr, ids, one_to_one=False), sub_gt)
        line += f"   [one-to-one delta: {s.macro_f05 - base.macro_f05:+.5f}]"
    print(line)
    return s.macro_f05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--rounds", type=int, default=700)
    ap.add_argument("--loco", action="store_true",
                    help="also run leave-one-country-out (the France proxy)")
    args = ap.parse_args()

    splits = [Split(p) for p in args.train]
    for sp in splits:
        print(f"[data] {sp.name:8s} rows={len(sp.y):,} pos={100*sp.y.mean():.2f}%  "
              f"train_ent={len(sp.tr_ent):,} val_ent={len(sp.va_ent):,}")

    # ---------------------------------------------------------------- LOCO first
    if args.loco and len(splits) > 1:
        print("\n=== LEAVE-ONE-COUNTRY-OUT (proxy for France, which has no training data) ===")
        for held in splits:
            others = [s for s in splits if s is not held]
            Xtr = np.vstack([s.X for s in others])
            ytr = np.concatenate([s.y for s in others])
            b = M.train(Xtr, ytr, num_round=args.rounds // 2, early_stopping=0)
            sc = b.predict(held.X, num_iteration=b.best_iteration)
            tri = [(a, c, float(p)) for a, c, p in zip(held.row_s1, held.row_cand, sc)]
            sw = tune_threshold(tri, held.gt, held.s1_ids)
            trained_on = "+".join(s.name for s in others)
            print(f"  train[{trained_on}] -> test[{held.name}]: "
                  f"F0.5={sw.best_f05:.5f} at its own best threshold {sw.best_threshold:.2f}")
            del Xtr, ytr, b

    # ---------------------------------------------------------------- final model
    print("\n=== FINAL MODEL (all countries) ===")
    Xtr = np.vstack([sp.X[sp.tr_mask] for sp in splits])
    ytr = np.concatenate([sp.y[sp.tr_mask] for sp in splits])
    Xva = np.vstack([sp.X[sp.va_mask] for sp in splits])
    yva = np.concatenate([sp.y[sp.va_mask] for sp in splits])
    print(f"[fit] train rows={len(ytr):,}  val rows={len(yva):,}")
    booster = M.train(Xtr, ytr, Xva, yva, num_round=args.rounds)
    path = M.save(booster)
    print(f"[model] best_iter={booster.best_iteration}  -> {path}\n")
    print(M.importances(booster, top=16))
    del Xtr, ytr, Xva, yva

    # ---- threshold tuned on the pooled validation slice across ALL countries, so
    # ---- it is not fitted to one country's idiosyncrasies.
    tri_all: List[Triple] = []
    gt_all: Dict[str, List[str]] = {}
    ids_all: List[str] = []
    for sp in splits:
        sc = booster.predict(sp.X[sp.va_mask], num_iteration=booster.best_iteration)
        s1s, cds = sp.rows(sp.va_mask)
        tri_all += [(a, b, float(p)) for a, b, p in zip(s1s, cds, sc)]
        gt_all.update({e: sp.gt.get(e, []) for e in sp.va_ent})
        ids_all += sorted(sp.va_ent)
    # coarse sweep, then refine around the winner at 0.01 resolution
    sweep = tune_threshold(tri_all, gt_all, ids_all)
    t0 = sweep.best_threshold
    fine = [round(t0 + d, 2) for d in
            (-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04)]
    sweep2 = tune_threshold(tri_all, gt_all, ids_all, grid=fine)
    print(f"\n[threshold sweep - pooled validation]\n{sweep.report()}")
    print(f"[fine sweep]\n{sweep2.report()}")
    thr = sweep2.best_threshold

    # rescue sweep: give empty entities their best sub-threshold candidate
    print(f"\n[rescue sweep @ threshold={thr:.2f}]")
    best_r, best_f = None, sweep2.best_f05
    for r in (0.20, 0.30, 0.40, 0.50, 0.60):
        pred = assign(tri_all, thr, ids_all, one_to_one=True, rescue=r)
        f = score_matching(pred, gt_all).macro_f05
        mark = ""
        if f > best_f:
            best_f, best_r, mark = f, r, "  <-- best"
        print(f"  rescue={r:.2f}  F0.5={f:.5f}{mark}")
    print(f"  no rescue    F0.5={sweep2.best_f05:.5f}")

    print(f"\n=== VALIDATION (threshold={thr:.2f}, rescue={best_r}) ===")
    for sp in splits:
        _eval(booster, sp, sp.va_mask, sp.va_ent, thr, f"{sp.name} held-out", ablate=True)
    print(f"  {'POOLED (final)':22s} F0.5={best_f:.5f}")

    (config.WORK_DIR / "threshold.txt").write_text(
        f"{thr}\n{'' if best_r is None else best_r}\n")
    print(f"\n[saved] threshold={thr} rescue={best_r} -> {config.WORK_DIR/'threshold.txt'}")


if __name__ == "__main__":  # pragma: no cover
    main()
