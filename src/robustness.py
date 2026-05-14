"""Robustness and diagnostic suite for NIFTY 50 direction prediction.

Implements seven post-pipeline analyses:
1. Purged/embargoed sensitivity suite with multiple split dates
2. Probability calibration check with reliability plots by volatility regime
3. Threshold-free trading analysis via probability deciles
4. Transaction cost sweep (0–15 bps)
5. Walk-forward feature stability report
6. Better null model (block-shuffled and regime-preserving label permutation)
7. Trade/no-trade filter based on confidence and volatility regime
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from src.backtest import long_flat_backtest, BacktestResult
from src.data_audit import clean, load_raw
from src.features import FEATURE_COLS, build_features
from src.validation import WalkForwardConfig, walk_forward_splits, iter_fold_arrays

SEED = 42
HELD_BACK_START = pd.Timestamp("2025-07-01")

DEFAULT_LGB_PARAMS = {
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


def _train_lgb(X_train, y_train, params=None):
    params = params or DEFAULT_LGB_PARAMS.copy()
    for k in ("num_leaves", "max_depth", "min_child_samples", "bagging_freq"):
        if k in params:
            params[k] = int(round(params[k]))
    n = len(y_train)
    split = int(n * 0.85)
    ds_tr = lgb.Dataset(X_train[:split], label=y_train[:split])
    ds_val = lgb.Dataset(X_train[split:], label=y_train[split:], reference=ds_tr)
    model = lgb.train(
        params, ds_tr, num_boost_round=500,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    return model


def _get_nifty_returns(dates):
    raw = load_raw()
    cleaned = clean(raw)
    prices = cleaned["nifty"].set_index("date")["Close"]
    returns = []
    for d in dates:
        if d in prices.index:
            idx = prices.index.get_loc(d)
            if idx + 1 < len(prices):
                returns.append((prices.iloc[idx + 1] - prices.iloc[idx]) / prices.iloc[idx])
            else:
                returns.append(0.0)
        else:
            returns.append(0.0)
    return np.array(returns)


# =========================================================================
# 1. PURGED & EMBARGOED SENSITIVITY SUITE
# =========================================================================

@dataclass
class SplitSensitivityResult:
    split_date: str
    train_rows: int
    test_rows: int
    accuracy: float
    balanced_accuracy: float
    auc: float
    sharpe: float
    total_return: float


def purged_sensitivity_suite(
    X: pd.DataFrame,
    y: pd.Series,
    split_dates: list[str] | None = None,
    embargo_days: int = 1,
    params: dict | None = None,
) -> pd.DataFrame:
    """Train on data before each split date, test on data after, with embargo.

    Uses a fixed recipe (no HPO) to isolate the effect of split boundary choice.
    """
    if split_dates is None:
        split_dates = [
            "2025-04-01", "2025-05-01", "2025-06-01",
            "2025-07-01", "2025-08-01", "2025-09-01",
        ]

    params = params or DEFAULT_LGB_PARAMS.copy()
    results = []

    for split_str in split_dates:
        split_ts = pd.Timestamp(split_str)

        train_mask = X.index < split_ts
        if train_mask.sum() == 0:
            continue

        boundary_date = X.index[train_mask].max()
        embargo_mask = (X.index > boundary_date) & (
            X.index <= boundary_date + pd.Timedelta(days=embargo_days * 3)
        )
        embargo_count = embargo_mask.sum()
        if embargo_count < embargo_days:
            embargo_count = embargo_days

        all_after = X.index >= split_ts
        dates_after = X.index[all_after]
        if len(dates_after) <= embargo_days:
            continue
        test_start_date = dates_after[embargo_days] if embargo_days < len(dates_after) else dates_after[0]
        test_mask = X.index >= test_start_date

        if test_mask.sum() < 10:
            continue

        X_tr = np.asarray(X[train_mask])
        y_tr = np.asarray(y[train_mask])
        X_te = np.asarray(X[test_mask])
        y_te = np.asarray(y[test_mask])

        model = _train_lgb(X_tr, y_tr, params.copy())
        probs = model.predict(X_te)
        preds = (probs >= 0.5).astype(int)

        acc = accuracy_score(y_te, preds)
        bal_acc = balanced_accuracy_score(y_te, preds)
        try:
            auc = roc_auc_score(y_te, probs)
        except ValueError:
            auc = 0.5

        test_dates = X.index[test_mask]
        fwd_returns = _get_nifty_returns(test_dates)
        bt = long_flat_backtest(probs, fwd_returns, test_dates, cost_bps=5.0)

        results.append(SplitSensitivityResult(
            split_date=split_str,
            train_rows=int(train_mask.sum()),
            test_rows=int(test_mask.sum()),
            accuracy=acc,
            balanced_accuracy=bal_acc,
            auc=auc,
            sharpe=bt.sharpe_ratio,
            total_return=bt.total_return,
        ))

    df = pd.DataFrame([vars(r) for r in results])
    return df


# =========================================================================
# 2. PROBABILITY CALIBRATION CHECK
# =========================================================================

@dataclass
class CalibrationResult:
    n_bins: int
    bin_edges: np.ndarray
    bin_mean_predicted: np.ndarray
    bin_mean_actual: np.ndarray
    bin_counts: np.ndarray
    ece: float
    regime_calibration: dict[str, dict]


def probability_calibration_check(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    vol_20d: np.ndarray | None = None,
    n_bins: int = 10,
) -> CalibrationResult:
    """Compute reliability diagram data and Expected Calibration Error.

    Optionally segments by volatility regime (low/medium/high).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_mean_predicted = np.zeros(n_bins)
    bin_mean_actual = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        bin_counts[i] = mask.sum()
        if mask.sum() > 0:
            bin_mean_predicted[i] = y_prob[mask].mean()
            bin_mean_actual[i] = y_true[mask].mean()

    total = len(y_true)
    ece = sum(
        (bin_counts[i] / total) * abs(bin_mean_actual[i] - bin_mean_predicted[i])
        for i in range(n_bins)
        if bin_counts[i] > 0
    )

    regime_cal = {}
    if vol_20d is not None:
        vol = np.asarray(vol_20d)
        p33, p66 = np.percentile(vol, [33, 66])
        regimes = {
            "low_vol": vol <= p33,
            "medium_vol": (vol > p33) & (vol <= p66),
            "high_vol": vol > p66,
        }
        for regime_name, regime_mask in regimes.items():
            if regime_mask.sum() < 5:
                continue
            r_true = y_true[regime_mask]
            r_prob = y_prob[regime_mask]
            r_ece = 0.0
            r_total = len(r_true)
            r_bins_pred = []
            r_bins_actual = []
            r_bins_count = []
            for i in range(n_bins):
                lo, hi = bin_edges[i], bin_edges[i + 1]
                if i == n_bins - 1:
                    m = (r_prob >= lo) & (r_prob <= hi)
                else:
                    m = (r_prob >= lo) & (r_prob < hi)
                cnt = m.sum()
                r_bins_count.append(cnt)
                if cnt > 0:
                    mp = r_prob[m].mean()
                    ma = r_true[m].mean()
                    r_bins_pred.append(mp)
                    r_bins_actual.append(ma)
                    r_ece += (cnt / r_total) * abs(ma - mp)
                else:
                    r_bins_pred.append(0.0)
                    r_bins_actual.append(0.0)

            regime_cal[regime_name] = {
                "n_samples": int(regime_mask.sum()),
                "ece": float(r_ece),
                "mean_prob": float(r_prob.mean()),
                "actual_positive_rate": float(r_true.mean()),
                "bin_mean_predicted": r_bins_pred,
                "bin_mean_actual": r_bins_actual,
                "bin_counts": r_bins_count,
            }

    return CalibrationResult(
        n_bins=n_bins,
        bin_edges=bin_edges,
        bin_mean_predicted=bin_mean_predicted,
        bin_mean_actual=bin_mean_actual,
        bin_counts=bin_counts,
        ece=float(ece),
        regime_calibration=regime_cal,
    )


# =========================================================================
# 3. THRESHOLD-FREE TRADING ANALYSIS (PROBABILITY DECILES)
# =========================================================================

@dataclass
class DecileResult:
    decile: int
    prob_low: float
    prob_high: float
    n_samples: int
    mean_prob: float
    actual_up_rate: float
    mean_return: float
    sharpe: float
    hit_rate: float
    cost_adjusted_return: float


def threshold_free_decile_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    returns: np.ndarray,
    cost_bps: float = 5.0,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Analyze performance by probability decile instead of a single threshold."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    returns = np.asarray(returns)

    try:
        decile_edges = np.percentile(y_prob, np.linspace(0, 100, n_deciles + 1))
    except Exception:
        decile_edges = np.linspace(y_prob.min(), y_prob.max(), n_deciles + 1)

    decile_edges = np.unique(decile_edges)
    actual_n_deciles = len(decile_edges) - 1

    results = []
    for i in range(actual_n_deciles):
        lo, hi = decile_edges[i], decile_edges[i + 1]
        if i == actual_n_deciles - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        if mask.sum() == 0:
            continue

        d_true = y_true[mask]
        d_prob = y_prob[mask]
        d_returns = returns[mask]

        cost = cost_bps / 10_000
        cost_adj_ret = d_returns.mean() - cost

        n = len(d_returns)
        if n > 1 and d_returns.std() > 0:
            sharpe = d_returns.mean() / d_returns.std() * np.sqrt(252)
        else:
            sharpe = 0.0

        hit = (d_returns > 0).mean()

        results.append(DecileResult(
            decile=i + 1,
            prob_low=float(lo),
            prob_high=float(hi),
            n_samples=int(mask.sum()),
            mean_prob=float(d_prob.mean()),
            actual_up_rate=float(d_true.mean()),
            mean_return=float(d_returns.mean()),
            sharpe=float(sharpe),
            hit_rate=float(hit),
            cost_adjusted_return=float(cost_adj_ret),
        ))

    return pd.DataFrame([vars(r) for r in results])


# =========================================================================
# 4. COST SWEEP
# =========================================================================

@dataclass
class CostSweepPoint:
    cost_bps: float
    sharpe: float
    total_return: float
    annualized_return: float
    hit_rate: float
    turnover: float
    psr: float
    n_trades: int


def cost_sweep(
    y_prob: np.ndarray,
    returns: np.ndarray,
    dates: np.ndarray,
    threshold: float = 0.50,
    cost_levels: list[float] | None = None,
) -> pd.DataFrame:
    """Sweep transaction costs from 0 to 15 bps and report backtest metrics."""
    if cost_levels is None:
        cost_levels = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0]

    results = []
    for cost in cost_levels:
        bt = long_flat_backtest(
            y_prob, returns, dates,
            cost_bps=cost, buy_threshold=threshold,
        )
        results.append(CostSweepPoint(
            cost_bps=cost,
            sharpe=bt.sharpe_ratio,
            total_return=bt.total_return,
            annualized_return=bt.annualized_return,
            hit_rate=bt.hit_rate,
            turnover=bt.turnover,
            psr=bt.probabilistic_sharpe,
            n_trades=bt.n_trades,
        ))

    df = pd.DataFrame([vars(r) for r in results])

    positive_sharpe = df[df["sharpe"] > 0]
    if len(positive_sharpe) > 0 and len(df[df["sharpe"] <= 0]) > 0:
        last_positive = positive_sharpe["cost_bps"].max()
        first_negative = df[df["sharpe"] <= 0]["cost_bps"].min()
        df.attrs["breakeven_bps_approx"] = (last_positive + first_negative) / 2
    else:
        df.attrs["breakeven_bps_approx"] = None

    return df


# =========================================================================
# 5. WALK-FORWARD FEATURE STABILITY
# =========================================================================

@dataclass
class FeatureStabilityResult:
    per_fold_importance: pd.DataFrame
    rank_correlation_matrix: pd.DataFrame
    mean_rank_correlation: float
    regime_importance: dict[str, pd.Series]
    unstable_features: list[str]


def walk_forward_feature_stability(
    X: pd.DataFrame,
    y: pd.Series,
    wf_cfg: WalkForwardConfig | None = None,
    params: dict | None = None,
    vol_feature: str = "vol_20d",
) -> FeatureStabilityResult:
    """Record per-fold feature importance and check stability across folds and regimes."""
    wf_cfg = wf_cfg or WalkForwardConfig()
    params = params or DEFAULT_LGB_PARAMS.copy()
    folds = walk_forward_splits(len(X), wf_cfg)
    feature_names = list(X.columns)

    fold_importances = []
    fold_ranks = []

    for fold, X_tr, y_tr, X_te, y_te in iter_fold_arrays(X, y, folds):
        model = _train_lgb(X_tr, y_tr, params.copy())
        imp = model.feature_importance(importance_type="gain")
        imp_dict = dict(zip(feature_names, imp))
        fold_importances.append({"fold": fold.fold_idx, **imp_dict})

        imp_series = pd.Series(imp_dict)
        ranks = imp_series.rank(ascending=False)
        fold_ranks.append(ranks)

    imp_df = pd.DataFrame(fold_importances).set_index("fold")
    rank_df = pd.DataFrame(fold_ranks)

    n_folds = len(fold_ranks)
    rank_corr = np.ones((n_folds, n_folds))
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            corr, _ = spearmanr(fold_ranks[i], fold_ranks[j])
            rank_corr[i, j] = corr
            rank_corr[j, i] = corr

    rank_corr_df = pd.DataFrame(
        rank_corr,
        index=[f"fold_{i}" for i in range(n_folds)],
        columns=[f"fold_{i}" for i in range(n_folds)],
    )
    mean_rank_corr = float(rank_corr[np.triu_indices(n_folds, k=1)].mean())

    rank_std = rank_df.std()
    unstable = rank_std[rank_std > len(feature_names) * 0.3].index.tolist()

    regime_importance = {}
    if vol_feature in feature_names:
        vol = X[vol_feature].values
        p33, p66 = np.percentile(vol[~np.isnan(vol)], [33, 66])
        regimes = {
            "low_vol": vol <= p33,
            "medium_vol": (vol > p33) & (vol <= p66),
            "high_vol": vol > p66,
        }
        for regime_name, regime_mask in regimes.items():
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]
            if len(y_regime) < 50:
                continue
            X_arr = np.asarray(X_regime)
            y_arr = np.asarray(y_regime)
            model = _train_lgb(X_arr, y_arr, params.copy())
            imp = dict(zip(feature_names, model.feature_importance(importance_type="gain")))
            regime_importance[regime_name] = pd.Series(imp).sort_values(ascending=False)

    return FeatureStabilityResult(
        per_fold_importance=imp_df,
        rank_correlation_matrix=rank_corr_df,
        mean_rank_correlation=mean_rank_corr,
        regime_importance=regime_importance,
        unstable_features=unstable,
    )


# =========================================================================
# 6. BETTER NULL MODEL (BLOCK & REGIME-PRESERVING SHUFFLING)
# =========================================================================

@dataclass
class NullModelResult:
    method: str
    real_accuracy: float
    null_mean: float
    null_std: float
    p_value: float
    n_shuffles: int
    null_distribution: np.ndarray


def _block_shuffle_labels(y: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Shuffle labels in contiguous blocks to preserve local autocorrelation."""
    n = len(y)
    n_blocks = int(np.ceil(n / block_size))
    blocks = [y[i * block_size: min((i + 1) * block_size, n)] for i in range(n_blocks)]
    rng.shuffle(blocks)
    return np.concatenate(blocks)[:n]


def _regime_preserving_shuffle(
    y: np.ndarray,
    regime_labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Shuffle labels independently within each regime."""
    y_shuffled = y.copy()
    for regime in np.unique(regime_labels):
        mask = regime_labels == regime
        indices = np.where(mask)[0]
        y_shuffled[indices] = rng.permutation(y[indices])
    return y_shuffled


def better_null_model(
    X: pd.DataFrame,
    y: pd.Series,
    wf_cfg: WalkForwardConfig | None = None,
    n_shuffles: int = 20,
    block_size: int = 21,
    vol_feature: str = "vol_20d",
    seed: int = SEED,
) -> dict[str, NullModelResult]:
    """Run three null model variants and compare.

    1. IID shuffle (existing baseline)
    2. Block shuffle (preserves local label autocorrelation)
    3. Regime-preserving shuffle (permutes within vol regimes)
    """
    wf_cfg = wf_cfg or WalkForwardConfig()
    folds = walk_forward_splits(len(X), wf_cfg)
    y_arr = np.asarray(y)
    rng = np.random.default_rng(seed)

    def _wf_accuracy(y_input: pd.Series) -> float:
        all_preds = []
        all_true = []
        for fold, X_tr, y_tr, X_te, y_te in iter_fold_arrays(X, y_input, folds):
            model = _train_lgb(X_tr, y_tr)
            probs = model.predict(X_te)
            all_preds.append((probs >= 0.5).astype(int))
            all_true.append(y_te)
        return float((np.concatenate(all_preds) == np.concatenate(all_true)).mean())

    real_acc = _wf_accuracy(y)
    print(f"  Real OOF accuracy: {real_acc:.4f}")

    vol = X[vol_feature].values if vol_feature in X.columns else None
    if vol is not None:
        p33, p66 = np.percentile(vol[~np.isnan(vol)], [33, 66])
        regime_labels = np.zeros(len(vol), dtype=int)
        regime_labels[vol > p33] = 1
        regime_labels[vol > p66] = 2
    else:
        regime_labels = np.zeros(len(y_arr), dtype=int)

    methods = {
        "iid_shuffle": lambda: rng.permutation(y_arr),
        "block_shuffle": lambda: _block_shuffle_labels(y_arr, block_size, rng),
        "regime_preserving": lambda: _regime_preserving_shuffle(y_arr, regime_labels, rng),
    }

    results = {}
    for method_name, shuffle_fn in methods.items():
        null_accs = []
        for i in range(n_shuffles):
            y_shuf = shuffle_fn()
            y_shuf_series = pd.Series(y_shuf, index=y.index)
            acc = _wf_accuracy(y_shuf_series)
            null_accs.append(acc)
            print(f"    {method_name} shuffle {i+1}/{n_shuffles}: acc={acc:.4f}")

        null_accs = np.array(null_accs)
        p_value = float(np.sum(null_accs >= real_acc) + 1) / (n_shuffles + 1)

        results[method_name] = NullModelResult(
            method=method_name,
            real_accuracy=real_acc,
            null_mean=float(null_accs.mean()),
            null_std=float(null_accs.std()),
            p_value=p_value,
            n_shuffles=n_shuffles,
            null_distribution=null_accs,
        )

    return results


# =========================================================================
# 7. TRADE/NO-TRADE FILTER
# =========================================================================

@dataclass
class TradeFilterResult:
    filter_name: str
    n_total: int
    n_traded: int
    trade_fraction: float
    unfiltered_sharpe: float
    filtered_sharpe: float
    unfiltered_return: float
    filtered_return: float
    unfiltered_hit_rate: float
    filtered_hit_rate: float
    filtered_max_dd: float


def trade_no_trade_filter(
    y_prob: np.ndarray,
    returns: np.ndarray,
    dates: np.ndarray,
    vol_20d: np.ndarray | None = None,
    threshold: float = 0.50,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Apply multiple trade/no-trade filters and compare vs unfiltered.

    Filters:
    1. Confidence filter: only trade when |prob - 0.5| > margin
    2. Volatility filter: skip trades in high-vol regime
    3. Combined: confidence + low/medium vol only
    4. Probability-ranked: only trade top/bottom quartile by probability
    """
    y_prob = np.asarray(y_prob)
    returns = np.asarray(returns)

    unfiltered_bt = long_flat_backtest(
        y_prob, returns, dates, cost_bps=cost_bps, buy_threshold=threshold,
    )

    results = []

    def _run_filtered(filter_name: str, trade_mask: np.ndarray):
        filtered_probs = np.where(trade_mask, y_prob, 0.0)
        filtered_bt = long_flat_backtest(
            filtered_probs, returns, dates,
            cost_bps=cost_bps, buy_threshold=threshold,
        )
        results.append(TradeFilterResult(
            filter_name=filter_name,
            n_total=len(y_prob),
            n_traded=int(trade_mask.sum()),
            trade_fraction=float(trade_mask.mean()),
            unfiltered_sharpe=unfiltered_bt.sharpe_ratio,
            filtered_sharpe=filtered_bt.sharpe_ratio,
            unfiltered_return=unfiltered_bt.total_return,
            filtered_return=filtered_bt.total_return,
            unfiltered_hit_rate=unfiltered_bt.hit_rate,
            filtered_hit_rate=filtered_bt.hit_rate,
            filtered_max_dd=filtered_bt.max_drawdown,
        ))

    results.append(TradeFilterResult(
        filter_name="unfiltered",
        n_total=len(y_prob),
        n_traded=int((y_prob >= threshold).sum()),
        trade_fraction=float((y_prob >= threshold).mean()),
        unfiltered_sharpe=unfiltered_bt.sharpe_ratio,
        filtered_sharpe=unfiltered_bt.sharpe_ratio,
        unfiltered_return=unfiltered_bt.total_return,
        filtered_return=unfiltered_bt.total_return,
        unfiltered_hit_rate=unfiltered_bt.hit_rate,
        filtered_hit_rate=unfiltered_bt.hit_rate,
        filtered_max_dd=unfiltered_bt.max_drawdown,
    ))

    for margin in [0.05, 0.10, 0.15]:
        confidence = np.abs(y_prob - 0.5)
        conf_mask = confidence > margin
        _run_filtered(f"confidence_{margin:.2f}", conf_mask)

    if vol_20d is not None:
        vol = np.asarray(vol_20d)
        p66 = np.percentile(vol[~np.isnan(vol)], 66)
        low_vol_mask = vol <= p66
        _run_filtered("low_medium_vol_only", low_vol_mask)

        for margin in [0.05, 0.10]:
            confidence = np.abs(y_prob - 0.5)
            combined = (confidence > margin) & low_vol_mask
            _run_filtered(f"conf_{margin:.2f}_+_low_vol", combined)

    p25, p75 = np.percentile(y_prob, [25, 75])
    extreme_mask = (y_prob <= p25) | (y_prob >= p75)
    _run_filtered("extreme_quartiles", extreme_mask)

    return pd.DataFrame([vars(r) for r in results])


# =========================================================================
# RUNNER: all 7 analyses
# =========================================================================

def run_full_robustness_suite(
    X: pd.DataFrame,
    y: pd.Series,
    oof_probs: np.ndarray,
    oof_true: np.ndarray,
    oof_returns: np.ndarray,
    oof_dates: np.ndarray,
    threshold: float = 0.50,
    wf_cfg: WalkForwardConfig | None = None,
    n_null_shuffles: int = 10,
) -> dict[str, Any]:
    """Run all 7 robustness analyses and return results dict."""
    print("\n" + "=" * 70)
    print("  ROBUSTNESS SUITE")
    print("=" * 70)

    # 1. Sensitivity suite
    print("\n--- 1. Purged/Embargoed Sensitivity Suite ---")
    sensitivity = purged_sensitivity_suite(X, y)
    print(sensitivity.to_string(index=False))

    # 2. Calibration
    print("\n--- 2. Probability Calibration Check ---")
    vol_20d = X["vol_20d"].values if "vol_20d" in X.columns else None
    oof_vol = None
    if vol_20d is not None and len(oof_true) <= len(vol_20d):
        oof_indices = np.arange(len(X) - len(oof_true), len(X))
        if len(oof_indices) == len(oof_true) and oof_indices[-1] < len(vol_20d):
            oof_vol = vol_20d[oof_indices]
    calibration = probability_calibration_check(oof_true, oof_probs, oof_vol)
    print(f"  ECE: {calibration.ece:.4f}")
    print(f"  Bins with data: {(calibration.bin_counts > 0).sum()}/{calibration.n_bins}")
    for i in range(calibration.n_bins):
        if calibration.bin_counts[i] > 0:
            print(f"    Bin {i+1}: predicted={calibration.bin_mean_predicted[i]:.3f}, "
                  f"actual={calibration.bin_mean_actual[i]:.3f}, n={calibration.bin_counts[i]}")
    for regime, data in calibration.regime_calibration.items():
        print(f"  {regime}: n={data['n_samples']}, ECE={data['ece']:.4f}, "
              f"mean_prob={data['mean_prob']:.3f}, actual_rate={data['actual_positive_rate']:.3f}")

    # 3. Decile analysis
    print("\n--- 3. Threshold-Free Decile Analysis ---")
    deciles = threshold_free_decile_analysis(oof_true, oof_probs, oof_returns)
    print(deciles.to_string(index=False))

    # 4. Cost sweep
    print("\n--- 4. Transaction Cost Sweep ---")
    costs = cost_sweep(oof_probs, oof_returns, oof_dates, threshold=threshold)
    print(costs[["cost_bps", "sharpe", "total_return", "psr"]].to_string(index=False))
    be = costs.attrs.get("breakeven_bps_approx")
    if be is not None:
        print(f"  Approximate breakeven cost: {be:.1f} bps")

    # 5. Feature stability
    print("\n--- 5. Walk-Forward Feature Stability ---")
    pre_held = X.index < HELD_BACK_START
    boundary = X.index == X.index[pre_held].max()
    train_mask = pre_held & ~boundary
    stability = walk_forward_feature_stability(
        X[train_mask], y[train_mask], wf_cfg,
    )
    print(f"  Mean rank correlation across folds: {stability.mean_rank_correlation:.4f}")
    print(f"  Unstable features: {stability.unstable_features or 'none'}")
    print("\n  Per-fold importance (top 5 by mean):")
    mean_imp = stability.per_fold_importance.mean().sort_values(ascending=False)
    for feat in mean_imp.head(5).index:
        vals = stability.per_fold_importance[feat]
        print(f"    {feat:25s}: mean={vals.mean():.1f}, std={vals.std():.1f}")
    for regime, imp in stability.regime_importance.items():
        print(f"\n  {regime} top features: {list(imp.head(3).index)}")

    # 6. Better null model
    print("\n--- 6. Better Null Model ---")
    null_results = better_null_model(
        X[train_mask], y[train_mask], wf_cfg,
        n_shuffles=n_null_shuffles,
    )
    for method, res in null_results.items():
        print(f"  {method}: real={res.real_accuracy:.4f}, "
              f"null_mean={res.null_mean:.4f}±{res.null_std:.4f}, "
              f"p={res.p_value:.4f}")

    # 7. Trade/no-trade filter
    print("\n--- 7. Trade/No-Trade Filter ---")
    oof_vol_for_filter = None
    if vol_20d is not None:
        try:
            oof_vol_for_filter = vol_20d[-len(oof_returns):]
        except Exception:
            pass
    trade_filter = trade_no_trade_filter(
        oof_probs, oof_returns, oof_dates,
        vol_20d=oof_vol_for_filter,
        threshold=threshold,
        cost_bps=5.0,
    )
    cols = ["filter_name", "n_traded", "trade_fraction",
            "filtered_sharpe", "filtered_return", "filtered_hit_rate"]
    print(trade_filter[cols].to_string(index=False))

    return {
        "sensitivity": sensitivity,
        "calibration": calibration,
        "deciles": deciles,
        "cost_sweep": costs,
        "feature_stability": stability,
        "null_models": null_results,
        "trade_filter": trade_filter,
    }
