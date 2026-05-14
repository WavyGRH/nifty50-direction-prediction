"""Baseline classifiers for next-day NIFTY 50 direction prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


class MajorityClassBaseline:
    """Always predicts the majority class observed during fit."""

    def __init__(self) -> None:
        self.majority_class_: int | None = None
        self.class_prior_: float | None = None

    def fit(self, y_train: ArrayLike) -> "MajorityClassBaseline":
        y = np.asarray(y_train)
        self.class_prior_ = y.mean()
        self.majority_class_ = int(self.class_prior_ >= 0.5)
        return self

    def predict(self, n: int) -> np.ndarray:
        return np.full(n, self.majority_class_, dtype=int)

    def __repr__(self) -> str:
        return f"MajorityClassBaseline(majority={self.majority_class_})"


class RandomBaseline:
    """Predicts 1 with probability equal to the training-set class prior."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.p_up_: float | None = None

    def fit(self, y_train: ArrayLike) -> "RandomBaseline":
        self.p_up_ = float(np.asarray(y_train).mean())
        return self

    def predict(self, n: int) -> np.ndarray:
        return (self.rng.random(n) < self.p_up_).astype(int)

    def __repr__(self) -> str:
        return f"RandomBaseline(p_up={self.p_up_:.3f})"


class PersistenceBaseline:
    """Predicts tomorrow's direction = today's observed direction."""

    def fit(self, y_train: ArrayLike) -> "PersistenceBaseline":
        return self

    def predict(self, y_true: ArrayLike, initial_label: int | None = None) -> np.ndarray:
        """Shift actuals by 1: prediction[t] = y_true[t-1].

        The first prediction uses the last observed training label when
        available. Falling back to y_true[0] is only for standalone smoke tests.
        """
        y = np.asarray(y_true)
        first = y[0] if initial_label is None else int(initial_label)
        return np.concatenate([[first], y[:-1]])

    def __repr__(self) -> str:
        return "PersistenceBaseline()"


def evaluate_baselines(
    y_train: ArrayLike, y_test: ArrayLike
) -> dict[str, dict[str, float]]:
    """Run all baselines on a train/test split and return accuracy metrics."""
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)
    n = len(y_test)

    results = {}

    majority = MajorityClassBaseline().fit(y_train)
    preds_maj = majority.predict(n)
    results["majority"] = {
        "accuracy": float((preds_maj == y_test).mean()),
        "predictions_mean": float(preds_maj.mean()),
    }

    random_bl = RandomBaseline(seed=42).fit(y_train)
    preds_rnd = random_bl.predict(n)
    results["random"] = {
        "accuracy": float((preds_rnd == y_test).mean()),
        "predictions_mean": float(preds_rnd.mean()),
    }

    persistence = PersistenceBaseline().fit(y_train)
    preds_per = persistence.predict(y_test, initial_label=int(y_train[-1]))
    results["persistence"] = {
        "accuracy": float((preds_per == y_test).mean()),
        "predictions_mean": float(preds_per.mean()),
    }

    return results


def run_baselines(
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    """Run all baselines across walk-forward folds and return per-fold metrics."""
    from src.validation import walk_forward_splits

    folds = walk_forward_splits(len(X))
    rows: list[dict] = []
    for fold in folds:
        y_train = np.asarray(y.iloc[fold.train_start : fold.train_end])
        y_test = np.asarray(y.iloc[fold.test_start : fold.test_end])
        fold_results = evaluate_baselines(y_train, y_test)
        for name, metrics in fold_results.items():
            rows.append({"fold": fold.fold_idx, "baseline": name, **metrics})
    return pd.DataFrame(rows)
