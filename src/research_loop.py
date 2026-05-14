"""Research loop utilities for feature and backtest sanity checks.

This module is intentionally separate from the production pipeline. It lets us
screen feature hypotheses on the in-sample window only, using a lightweight
walk-forward model and an explicit null test before a candidate is promoted.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from src.data_audit import clean, load_raw
from src.features import FEATURE_COLS, _build_feature_df
from src.validation import WalkForwardConfig, iter_fold_arrays, walk_forward_splits

HELD_BACK_START = pd.Timestamp("2025-07-01")
SEED = 42


@dataclass(frozen=True)
class FeatureSetResult:
    name: str
    n_samples: int
    n_oof: int
    n_folds: int
    accuracy: float
    balanced_accuracy: float
    auc: float
    max_abs_corr: float
    top_corr_pair: tuple[str, str]
    correct: np.ndarray


@dataclass(frozen=True)
class HypothesisResult:
    candidate: str
    baseline: str
    delta_accuracy: float
    p_value: float
    accepted: bool


def build_candidate_frame() -> pd.DataFrame:
    """Return the raw feature frame plus safe exploratory candidates."""
    raw = load_raw()
    cleaned = clean(raw)
    df = _build_feature_df(cleaned)

    nifty = cleaned["nifty"]
    bn = cleaned["banknifty"]
    vix = cleaned["vix"]
    close = nifty["Close"]

    df["close_vs_252d_high_exact"] = close / close.rolling(252).max() - 1
    df["close_vs_252d_high_expanding"] = (
        close / close.rolling(252, min_periods=20).max() - 1
    )
    df["close_vs_252d_low_expanding"] = (
        close / close.rolling(252, min_periods=20).min() - 1
    )
    df["ret_intraday"] = (nifty["Close"] - nifty["Open"]) / nifty["Open"]
    df["bn_ret_5d"] = bn["Adj Close"].pct_change(5)
    df["vix_ma_ratio"] = vix["Close"] / vix["Close"].rolling(20, min_periods=5).mean() - 1
    df["vix_5d_change"] = vix["Close"].pct_change(5)
    df["momentum_5_20"] = df["ret_5d"] - df["ret_20d"]
    return df


def _xy_for_columns(
    df: pd.DataFrame,
    feature_cols: list[str],
    held_back_start: pd.Timestamp = HELD_BACK_START,
) -> tuple[pd.DataFrame, pd.Series]:
    out = df[["date", *feature_cols, "target"]].dropna().reset_index(drop=True)
    X = out[feature_cols].copy()
    X.index = pd.DatetimeIndex(out["date"])
    y = pd.Series(out["target"].astype(int).values, index=X.index, name="target")
    pre_held_mask = X.index < held_back_start
    boundary_label_mask = X.index == X.index[pre_held_mask].max()
    train_mask = pre_held_mask & ~boundary_label_mask
    return X.loc[train_mask], y.loc[train_mask]


def evaluate_feature_set(
    name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    wf_cfg: WalkForwardConfig | None = None,
) -> FeatureSetResult:
    """Evaluate a feature set with fixed LightGBM parameters."""
    wf_cfg = wf_cfg or WalkForwardConfig()
    X, y = _xy_for_columns(df, feature_cols)
    folds = walk_forward_splits(len(X), wf_cfg)

    corr = X.corr().abs()
    np.fill_diagonal(corr.values, 0)
    max_abs_corr = float(corr.max().max())
    top_corr_pair = tuple(corr.stack().idxmax())

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": SEED,
        "n_jobs": -1,
        "num_leaves": 15,
        "learning_rate": 0.05,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }

    all_probs: list[np.ndarray] = []
    all_preds: list[np.ndarray] = []
    all_true: list[np.ndarray] = []

    for _, X_train, y_train, X_test, y_test in iter_fold_arrays(X, y, folds):
        split = int(len(y_train) * 0.85)
        train_data = lgb.Dataset(X_train[:split], label=y_train[:split])
        val_data = lgb.Dataset(X_train[split:], label=y_train[split:], reference=train_data)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        probs = model.predict(X_test)
        all_probs.append(probs)
        all_preds.append((probs >= 0.5).astype(int))
        all_true.append(y_test)

    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)
    true = np.concatenate(all_true)
    correct = (preds == true).astype(int)

    return FeatureSetResult(
        name=name,
        n_samples=len(X),
        n_oof=len(true),
        n_folds=len(folds),
        accuracy=float(correct.mean()),
        balanced_accuracy=float(balanced_accuracy_score(true, preds)),
        auc=float(roc_auc_score(true, probs)),
        max_abs_corr=max_abs_corr,
        top_corr_pair=top_corr_pair,
        correct=correct,
    )


def paired_bootstrap_feature_test(
    candidate: FeatureSetResult,
    baseline: FeatureSetResult,
    alpha: float = 0.10,
    n_bootstrap: int = 5000,
    seed: int = 123,
) -> HypothesisResult:
    """One-sided paired bootstrap test of candidate accuracy over baseline.

    H0: candidate accuracy improvement <= 0.
    The default alpha is 10% because this is a screening loop, not the final
    claim of edge. Anything promoted still needs the full walk-forward pipeline.
    """
    if len(candidate.correct) != len(baseline.correct):
        raise ValueError("candidate and baseline must have aligned OOF lengths")

    diff = candidate.correct - baseline.correct
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, len(diff), len(diff))
        boot[i] = diff[idx].mean()
    p_value = float((np.sum(boot <= 0) + 1) / (n_bootstrap + 1))

    return HypothesisResult(
        candidate=candidate.name,
        baseline=baseline.name,
        delta_accuracy=float(diff.mean()),
        p_value=p_value,
        accepted=bool(p_value <= alpha),
    )


def run_feature_research_loop(alpha: float = 0.10) -> pd.DataFrame:
    """Run the current feature screen and print a compact decision table."""
    df = build_candidate_frame()
    base_cols = FEATURE_COLS.copy()
    baseline = evaluate_feature_set("final_current", df, base_cols)

    variants = {
        "legacy_252d_high_exact": [
            "close_vs_252d_high_exact" if c == "close_vs_126d_high" else c
            for c in base_cols
        ],
        "expanding_252d_high": [
            "close_vs_252d_high_expanding" if c == "close_vs_126d_high" else c
            for c in base_cols
        ],
        "replace_with_252d_low": [
            "close_vs_252d_low_expanding" if c == "close_vs_126d_high" else c
            for c in base_cols
        ],
        "replace_with_ret_intraday": [
            "ret_intraday" if c == "close_vs_126d_high" else c for c in base_cols
        ],
        "replace_with_vix_ma_ratio": [
            "vix_ma_ratio" if c == "close_vs_126d_high" else c for c in base_cols
        ],
        "replace_with_vix_5d_change": [
            "vix_5d_change" if c == "close_vs_126d_high" else c for c in base_cols
        ],
    }

    rows = [
        {
            "name": baseline.name,
            "n_samples": baseline.n_samples,
            "n_oof": baseline.n_oof,
            "accuracy": baseline.accuracy,
            "balanced_accuracy": baseline.balanced_accuracy,
            "auc": baseline.auc,
            "max_abs_corr": baseline.max_abs_corr,
            "delta_accuracy": 0.0,
            "p_value": np.nan,
            "accepted": True,
        }
    ]

    for name, cols in variants.items():
        result = evaluate_feature_set(name, df, cols)
        accepted_corr = result.max_abs_corr < 0.85
        if len(result.correct) == len(baseline.correct):
            hyp = paired_bootstrap_feature_test(result, baseline, alpha=alpha)
            p_value = hyp.p_value
            delta = hyp.delta_accuracy
            accepted = accepted_corr and hyp.accepted
        else:
            p_value = np.nan
            delta = np.nan
            accepted = False
        rows.append(
            {
                "name": name,
                "n_samples": result.n_samples,
                "n_oof": result.n_oof,
                "accuracy": result.accuracy,
                "balanced_accuracy": result.balanced_accuracy,
                "auc": result.auc,
                "max_abs_corr": result.max_abs_corr,
                "delta_accuracy": delta,
                "p_value": p_value,
                "accepted": accepted,
            }
        )

    decisions = pd.DataFrame(rows)
    print(decisions.to_string(index=False))
    return decisions


if __name__ == "__main__":
    run_feature_research_loop()
