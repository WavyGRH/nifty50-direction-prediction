"""Expanding-window walk-forward validation for time-series classification.

The default uses non-overlapping monthly test blocks. Earlier versions used a
63-day test block with a 21-day step, which caused the same dates to appear in
multiple out-of-fold metric calculations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    initial_train_days: int = 252
    test_days: int = 21
    step_days: int = 21
    embargo_days: int = 1


@dataclass(frozen=True)
class FoldInfo:
    fold_idx: int
    train_start: int
    train_end: int  # exclusive
    test_start: int
    test_end: int   # exclusive


def walk_forward_splits(
    n_samples: int | pd.Index,
    cfg: WalkForwardConfig | None = None,
) -> list[FoldInfo]:
    if not isinstance(n_samples, int):
        n_samples = len(n_samples)
    cfg = cfg or WalkForwardConfig()
    folds: list[FoldInfo] = []
    fold_idx = 0
    train_end = cfg.initial_train_days

    while True:
        test_start = train_end + cfg.embargo_days
        test_end = test_start + cfg.test_days
        if test_end > n_samples:
            break
        folds.append(FoldInfo(
            fold_idx=fold_idx,
            train_start=0,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        ))
        fold_idx += 1
        train_end += cfg.step_days

    return folds


def iter_fold_arrays(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    folds: list[FoldInfo],
) -> Iterator[tuple[FoldInfo, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    for fold in folds:
        X_train = X_arr[fold.train_start : fold.train_end]
        y_train = y_arr[fold.train_start : fold.train_end]
        X_test = X_arr[fold.test_start : fold.test_end]
        y_test = y_arr[fold.test_start : fold.test_end]
        yield fold, X_train, y_train, X_test, y_test


def describe_splits(folds: list[FoldInfo], dates: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for f in folds:
        row = {
            "fold": f.fold_idx,
            "train_size": f.train_end - f.train_start,
            "test_size": f.test_end - f.test_start,
        }
        if dates is not None:
            row["train_start"] = dates.iloc[f.train_start]
            row["train_end"] = dates.iloc[f.train_end - 1]
            row["test_start"] = dates.iloc[f.test_start]
            row["test_end"] = dates.iloc[f.test_end - 1]
        rows.append(row)
    return pd.DataFrame(rows)
