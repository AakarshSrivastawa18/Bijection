"""Phase 3: the matching model.

LightGBM (MIT licence) over engineered string/overlap/listwise features, rather than
a neural cross-encoder. At 1.73M Source-1 entities x 20 candidates the model runs
~35M inferences; a transformer at that scale costs GPU-hours for a gain that the
feature set already largely captures, and the challenge caps models at 8B params
under MIT/Apache anyway. A GBDT also gives directly inspectable feature importances,
which is what the methodology write-up needs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import lightgbm as lgb
import numpy as np

from . import config
from .features import FEATURE_NAMES

PARAMS = {
    "objective": "binary",
    "metric": ["binary_logloss", "auc"],
    "learning_rate": 0.06,
    "num_leaves": 96,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "max_bin": 127,
    "num_threads": 0,
    "verbosity": -1,
    "seed": config.RANDOM_SEED,
}


def train(X: np.ndarray, y: np.ndarray,
          X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
          num_round: int = 700, early_stopping: int = 50) -> lgb.Booster:
    dtrain = lgb.Dataset(X, label=y, feature_name=FEATURE_NAMES, free_raw_data=False)
    valid, names = [dtrain], ["train"]
    if X_val is not None and len(X_val):
        dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_NAMES,
                           reference=dtrain, free_raw_data=False)
        valid.append(dval)
        names.append("valid")
    cbs = [lgb.log_evaluation(period=100)]
    if len(valid) > 1:
        cbs.append(lgb.early_stopping(early_stopping, verbose=False))
    return lgb.train(PARAMS, dtrain, num_boost_round=num_round,
                     valid_sets=valid, valid_names=names, callbacks=cbs)


def save(booster: lgb.Booster, path: Optional[Path] = None) -> Path:
    path = path or (config.WORK_DIR / "model.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))
    return path


def load(path: Optional[Path] = None) -> lgb.Booster:
    path = path or (config.WORK_DIR / "model.txt")
    return lgb.Booster(model_file=str(path))


def importances(booster: lgb.Booster, top: int = 20) -> str:
    gain = booster.feature_importance(importance_type="gain")
    order = np.argsort(gain)[::-1][:top]
    total = gain.sum() or 1.0
    lines = [f"{'feature':24s} {'gain%':>8s}"]
    for i in order:
        lines.append(f"{FEATURE_NAMES[i]:24s} {100*gain[i]/total:8.2f}")
    return "\n".join(lines)


__all__ = ["train", "save", "load", "importances", "PARAMS"]
