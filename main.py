#!/usr/bin/env python
"""
End-to-end pipeline for NIFTY 50 next-day direction prediction.

Run:  python main.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)

# -- Project imports --
from src.data_audit import load_raw, audit, clean, print_report
from src.features import build_features, FEATURE_COLS
from src.validation import (
    WalkForwardConfig,
    walk_forward_splits,
    iter_fold_arrays,
    describe_splits,
)
from src.baselines import evaluate_baselines, run_baselines
from src.models import (
    EXPERIMENT_NAME,
    OPTUNA_N_TRIALS,
    SEED,
    run_walk_forward,
    select_final_params_from_walk_forward,
    select_probability_threshold,
)
from src.backtest import (
    BacktestResult,
    block_bootstrap_backtest_ci,
    buy_and_hold,
    compare_strategies,
    long_flat_backtest,
)
from src.analysis import (
    bootstrap_ci,
    permutation_test,
    random_label_test,
    full_statistical_report,
    accuracy,
    delong_test,
)
from src.robustness import run_full_robustness_suite

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
REPORT_DIR = PROJECT_ROOT / "report"
REPORT_DIR.mkdir(exist_ok=True)

# -- Held-back split --
HELD_BACK_START = pd.Timestamp("2025-07-01")


def _separator(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


# =========================================================================
# PHASE 1: Data Audit
# =========================================================================
def phase1_data_audit():
    _separator("PHASE 1: DATA AUDIT")
    raw = load_raw()
    report = audit(raw)
    print_report(report)
    return raw, report


# =========================================================================
# PHASE 2: Feature Engineering
# =========================================================================
def phase2_features():
    _separator("PHASE 2: FEATURE ENGINEERING")
    X, y = build_features()
    print(f"Feature matrix X: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Date range: {X.index.min().date()} to {X.index.max().date()}")
    print(f"Target distribution: up={int(y.sum())}, down={int(len(y)-y.sum())}")
    print(f"NaN count: {int(X.isna().sum().sum())}")

    # Correlation check
    corr = X.corr().abs()
    np.fill_diagonal(corr.values, 0)
    max_corr = corr.max().max()
    pair = corr.stack().idxmax()
    print(f"\nMax pairwise |correlation|: {max_corr:.4f} ({pair[0]} vs {pair[1]})")
    assert max_corr < 0.85, f"FAIL: max correlation {max_corr:.4f} >= 0.85"
    print("[OK] All feature-feature correlations < 0.85")

    return X, y


# =========================================================================
# PHASE 3: Walk-Forward Split
# =========================================================================
def phase3_validation(X, y):
    _separator("PHASE 3: WALK-FORWARD VALIDATION SETUP")

    # Split: training period vs held-back
    pre_held_mask = X.index < HELD_BACK_START
    boundary_label_mask = X.index == X.index[pre_held_mask].max()
    train_mask = pre_held_mask & ~boundary_label_mask
    held_mask = X.index >= HELD_BACK_START
    X_train_period = X[train_mask]
    y_train_period = y[train_mask]
    X_held = X[held_mask]
    y_held = y[held_mask]

    print(f"Training period: {X_train_period.index.min().date()} to {X_train_period.index.max().date()} ({len(X_train_period)} samples)")
    print(f"Held-back period: {X_held.index.min().date()} to {X_held.index.max().date()} ({len(X_held)} samples)")
    print(f"Boundary row excluded to keep labels out of held-back: {X.index[boundary_label_mask][0].date()}")

    wf_cfg = WalkForwardConfig()
    folds = walk_forward_splits(len(X_train_period), wf_cfg)
    print(f"\nWalk-forward folds: {len(folds)}")
    print(f"Config: initial_train={wf_cfg.initial_train_days}d, test={wf_cfg.test_days}d, step={wf_cfg.step_days}d, embargo={wf_cfg.embargo_days}d")

    desc = describe_splits(folds, X_train_period.index.to_series().reset_index(drop=True))
    print(f"\n{desc.to_string(index=False)}")

    return X_train_period, y_train_period, X_held, y_held, wf_cfg, folds


# =========================================================================
# PHASE 4: Baselines
# =========================================================================
def phase4_baselines(X_train, y_train):
    _separator("PHASE 4: BASELINE CLASSIFIERS")
    baseline_df = run_baselines(X_train, y_train)
    summary = baseline_df.groupby("baseline")["accuracy"].agg(["mean", "std"])
    print(summary)
    print()

    # Log baselines to MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="baselines"):
        for name, row in summary.iterrows():
            mlflow.log_metric(f"baseline_{name}_mean_acc", row["mean"])
            mlflow.log_metric(f"baseline_{name}_std_acc", row["std"])
        mlflow.log_param("baselines", list(summary.index))

    return baseline_df, summary


# =========================================================================
# PHASE 5: LightGBM Walk-Forward Training
# =========================================================================
def phase5_lightgbm(X_train, y_train, wf_cfg, n_trials=None):
    n_trials = n_trials or OPTUNA_N_TRIALS
    _separator(f"PHASE 5: LightGBM + OPTUNA WALK-FORWARD TRAINING ({n_trials} trials/fold)")
    results_df, oof_data = run_walk_forward(
        X_train, y_train,
        feature_names=list(X_train.columns),
        wf_cfg=wf_cfg,
        n_trials=n_trials,
    )
    print(f"\nPer-fold results:\n{results_df[['fold','train_size','accuracy','f1','roc_auc','log_loss','n_estimators']].to_string(index=False)}")

    oof_acc = accuracy_score(oof_data["true"], oof_data["preds"])
    oof_auc_val = roc_auc_score(oof_data["true"], oof_data["probs"])
    selected_threshold, threshold_score = select_probability_threshold(
        oof_data["true"], oof_data["probs"], metric="balanced_accuracy"
    )
    selected_preds = (oof_data["probs"] >= selected_threshold).astype(int)
    oof_data["selected_threshold"] = selected_threshold
    oof_data["selected_preds"] = selected_preds
    print(f"\nOOF Accuracy: {oof_acc:.4f}")
    print(f"OOF AUC:      {oof_auc_val:.4f}")
    print(f"OOF samples:  {len(oof_data['true'])}")
    print(
        f"OOF threshold selected in-sample: {selected_threshold:.4f} "
        f"(balanced accuracy={threshold_score:.4f})"
    )

    return results_df, oof_data


# =========================================================================
# PHASE 6: Compute Forward Returns for OOF
# =========================================================================
def phase6_oof_returns(X_train, oof_data):
    _separator("PHASE 6: OOF FORWARD RETURNS")

    raw = load_raw()
    cleaned = clean(raw)
    nifty = cleaned["nifty"]
    nifty_prices = nifty.set_index("date")["Close"]

    dates = X_train.index
    oof_indices = oof_data["fold_indices"]
    oof_dates = dates[oof_indices]

    oof_returns = []
    for d in oof_dates:
        if d in nifty_prices.index:
            idx_pos = nifty_prices.index.get_loc(d)
            if idx_pos + 1 < len(nifty_prices):
                ret = (nifty_prices.iloc[idx_pos + 1] - nifty_prices.iloc[idx_pos]) / nifty_prices.iloc[idx_pos]
            else:
                ret = 0.0
        else:
            ret = 0.0
        oof_returns.append(ret)

    oof_returns = np.array(oof_returns)
    print(f"OOF returns computed for {len(oof_returns)} samples")
    print(f"Mean daily return: {oof_returns.mean():.6f}")
    print(f"Date range: {oof_dates.min().date()} to {oof_dates.max().date()}")

    return oof_returns, oof_dates


# =========================================================================
# PHASE 7: Backtest
# =========================================================================
def phase7_backtest(oof_probs, oof_returns, oof_dates, threshold=0.50):
    _separator("PHASE 7: BACKTEST")

    # Model backtest
    bt = long_flat_backtest(oof_probs, oof_returns, oof_dates, cost_bps=5.0, buy_threshold=threshold)
    print(f"Strategy Performance (with 5bps cost):")
    print(f"  Sharpe Ratio:      {bt.sharpe_ratio:.4f}")
    print(f"  Max Drawdown:      {bt.max_drawdown:.4%}")
    print(f"  Hit Rate:          {bt.hit_rate:.4%}")
    print(f"  Turnover:          {bt.turnover:.4f}")
    print(f"  Total Return:      {bt.total_return:.4%}")
    print(f"  Annualized Return: {bt.annualized_return:.4%}")
    print(f"  N Trades:          {bt.n_trades}")
    print(f"  PSR vs 0 Sharpe:   {bt.probabilistic_sharpe:.4f}")

    # Buy-and-hold benchmark
    bh = buy_and_hold(oof_returns, oof_dates)
    print(f"\nBuy & Hold Benchmark:")
    print(f"  Sharpe Ratio:      {bh.sharpe_ratio:.4f}")
    print(f"  Max Drawdown:      {bh.max_drawdown:.4%}")
    print(f"  Total Return:      {bh.total_return:.4%}")

    return bt, bh


# =========================================================================
# PHASE 8: Statistical Analysis
# =========================================================================
def phase8_statistics(oof_true, oof_preds, oof_probs):
    _separator("PHASE 8: STATISTICAL SIGNIFICANCE TESTS")

    # Bootstrap CI on accuracy.
    ci = bootstrap_ci(oof_true, oof_preds, "accuracy", n_bootstrap=2000, alpha=0.05)
    print(f"Accuracy: {ci.point_estimate:.4f}  95% CI: [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]")

    # Bootstrap CI on F1.
    ci_f1 = bootstrap_ci(oof_true, oof_preds, "f1", n_bootstrap=2000, alpha=0.05)
    print(f"F1:       {ci_f1.point_estimate:.4f}  95% CI: [{ci_f1.ci_lower:.4f}, {ci_f1.ci_upper:.4f}]")

    # Bootstrap CI on AUC.
    ci_auc = bootstrap_ci(oof_true, oof_probs, "roc_auc", n_bootstrap=2000, alpha=0.05)
    print(f"AUC:      {ci_auc.point_estimate:.4f}  95% CI: [{ci_auc.ci_lower:.4f}, {ci_auc.ci_upper:.4f}]")

    # DeLong's test: model AUC vs random baseline AUC (MLstatkit)
    rng = np.random.default_rng(42)
    random_probs = rng.random(len(oof_true))  # uniform random scores as baseline
    delong = delong_test(oof_true, oof_probs, random_probs, alpha=0.05)
    print(f"\nDeLong test (model vs random baseline):")
    print(f"  Model AUC: {delong.auc_a:.4f}  CI: [{delong.ci_a[0]:.4f}, {delong.ci_a[1]:.4f}]")
    print(f"  Random AUC: {delong.auc_b:.4f}  CI: [{delong.ci_b[0]:.4f}, {delong.ci_b[1]:.4f}]")
    print(f"  z-statistic: {delong.z_statistic:.4f}")
    print(f"  p-value:     {delong.p_value:.4f}")
    print(f"  method:      {delong.method}")

    # Permutation test
    perm = permutation_test(oof_true, oof_preds, accuracy, n_permutations=1000)
    print(f"\nPermutation test (H0: no association):")
    print(f"  Observed accuracy: {perm.observed_statistic:.4f}")
    print(f"  p-value:           {perm.p_value:.4f}")

    # Random label test
    rand = random_label_test(oof_true, oof_preds, metric_fn=accuracy, n_trials=1000)
    print(f"\nRandom-label test:")
    print(f"  Model accuracy:    {rand.model_metric:.4f}")
    print(f"  Random mean +/- std: {rand.random_mean:.4f} +/- {rand.random_std:.4f}")
    print(f"  p-value:           {rand.p_value:.4f}")
    print(f"  Percentile rank:   {rand.percentile_rank:.1f}%")

    return {
        "accuracy_ci": ci,
        "f1_ci": ci_f1,
        "auc_ci": ci_auc,
        "delong": delong,
        "permutation": perm,
        "random_label": rand,
    }


# =========================================================================
# PHASE 9: Backtest Bootstrap CIs
# =========================================================================
def phase9_backtest_bootstrap(oof_probs, oof_returns, threshold=0.50, n_bootstrap=None, seed=42):
    _separator("PHASE 9: BLOCK-BOOTSTRAP BACKTEST CONFIDENCE INTERVALS")
    n_bootstrap = n_bootstrap or int(os.getenv("NIFTY_BOOTSTRAP_N", "500"))
    cis = block_bootstrap_backtest_ci(
        oof_probs,
        oof_returns,
        cost_bps=5.0,
        buy_threshold=threshold,
        n_bootstrap=n_bootstrap,
        block_size=5,
        seed=seed,
    )

    for name, (lo, hi) in cis.items():
        print(f"  {name:22s}: [{lo:.4f}, {hi:.4f}]")

    return {f"{name}_ci": bounds for name, bounds in cis.items()}


# =========================================================================
# PHASE 10: Held-Back Evaluation (Jul-Dec 2025)
# =========================================================================
def phase10_held_back(
    X_train,
    y_train,
    X_held,
    y_held,
    final_params=None,
    threshold=0.50,
):
    _separator("PHASE 10: HELD-BACK EVALUATION (Jul-Dec 2025) -- SINGLE PASS")

    X_tr_arr = np.asarray(X_train)
    y_tr_arr = np.asarray(y_train)
    X_held_arr = np.asarray(X_held)
    y_held_arr = np.asarray(y_held)

    final_params = final_params or {
        "num_leaves": 15,
        "learning_rate": 0.05,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": SEED,
        "n_jobs": -1,
        **final_params,
    }
    for int_param in ("num_leaves", "max_depth", "min_child_samples", "bagging_freq"):
        if int_param in params:
            params[int_param] = int(round(params[int_param]))

    n = len(y_tr_arr)
    split = int(n * 0.85)
    ds_tr = lgb.Dataset(X_tr_arr[:split], label=y_tr_arr[:split])
    ds_val = lgb.Dataset(X_tr_arr[split:], label=y_tr_arr[split:], reference=ds_tr)

    model = lgb.train(
        params, ds_tr, num_boost_round=2000,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)],
    )

    held_probs = model.predict(X_held_arr)
    held_preds = (held_probs >= threshold).astype(int)

    held_acc = accuracy_score(y_held_arr, held_preds)
    held_bal_acc = balanced_accuracy_score(y_held_arr, held_preds)
    held_f1 = f1_score(y_held_arr, held_preds, zero_division=0)
    held_ll = log_loss(y_held_arr, held_probs)
    try:
        held_auc = roc_auc_score(y_held_arr, held_probs)
    except ValueError:
        held_auc = 0.5

    print(f"Held-back period: {X_held.index.min().date()} to {X_held.index.max().date()}")
    print(f"Held-back samples: {len(y_held_arr)}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Accuracy:  {held_acc:.4f}")
    print(f"  Bal Acc:   {held_bal_acc:.4f}")
    print(f"  AUC:       {held_auc:.4f}")
    print(f"  F1:        {held_f1:.4f}")
    print(f"  Log-loss:  {held_ll:.4f}")
    cm = confusion_matrix(y_held_arr, held_preds, labels=[0, 1])
    print(f"  Confusion matrix [[TN, FP], [FN, TP]]: {cm.tolist()}")

    # Bootstrap CIs on held-back.
    ci = bootstrap_ci(y_held_arr, held_preds, "accuracy", n_bootstrap=2000)
    print(f"  Accuracy 95% CI: [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]")

    ci_bal = bootstrap_ci(
        y_held_arr,
        held_preds,
        lambda yt, yp: float(balanced_accuracy_score(yt, yp)),
        n_bootstrap=2000,
    )
    print(f"  Balanced Acc 95% CI: [{ci_bal.ci_lower:.4f}, {ci_bal.ci_upper:.4f}]")

    ci_auc = bootstrap_ci(y_held_arr, held_probs, "roc_auc", n_bootstrap=2000)
    print(f"  AUC 95% CI: [{ci_auc.ci_lower:.4f}, {ci_auc.ci_upper:.4f}]")

    ci_f1 = bootstrap_ci(y_held_arr, held_preds, "f1", n_bootstrap=2000)
    print(f"  F1 95% CI: [{ci_f1.ci_lower:.4f}, {ci_f1.ci_upper:.4f}]")

    # Held-back trading backtest.
    raw = load_raw()
    cleaned = clean(raw)
    nifty_prices = cleaned["nifty"].set_index("date")["Close"]
    held_returns = []
    for d in X_held.index:
        idx_pos = nifty_prices.index.get_loc(d)
        held_returns.append(
            (nifty_prices.iloc[idx_pos + 1] - nifty_prices.iloc[idx_pos])
            / nifty_prices.iloc[idx_pos]
        )
    held_returns = np.asarray(held_returns)
    held_bt = long_flat_backtest(
        held_probs,
        held_returns,
        X_held.index,
        cost_bps=5.0,
        buy_threshold=threshold,
    )
    held_bt_cis = block_bootstrap_backtest_ci(
        held_probs,
        held_returns,
        cost_bps=5.0,
        buy_threshold=threshold,
        n_bootstrap=1000,
        block_size=5,
    )
    print("\nHeld-back costed long/flat backtest:")
    print(f"  Sharpe Ratio:      {held_bt.sharpe_ratio:.4f}")
    print(f"  Max Drawdown:      {held_bt.max_drawdown:.4%}")
    print(f"  Hit Rate:          {held_bt.hit_rate:.4%}")
    print(f"  Turnover:          {held_bt.turnover:.4f}")
    print(f"  Total Return:      {held_bt.total_return:.4%}")
    print(f"  PSR vs 0 Sharpe:   {held_bt.probabilistic_sharpe:.4f}")
    for metric, (lo, hi) in held_bt_cis.items():
        print(f"  {metric} 95% CI: [{lo:.4f}, {hi:.4f}]")

    # Feature importance
    importance = dict(zip(FEATURE_COLS, model.feature_importance(importance_type="gain")))
    top = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\nFeature importance (gain):")
    for feat, imp in top:
        print(f"  {feat:25s}: {imp:.1f}")

    # Log to MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="held_back_eval"):
        mlflow.log_params(params)
        mlflow.log_param("period", "Jul-Dec 2025")
        mlflow.log_param("n_samples", len(y_held_arr))
        mlflow.log_param("decision_threshold", threshold)
        mlflow.log_metrics({
            "held_accuracy": held_acc,
            "held_balanced_accuracy": held_bal_acc,
            "held_auc": held_auc,
            "held_f1": held_f1,
            "held_logloss": held_ll,
            "held_accuracy_ci_lower": ci.ci_lower,
            "held_accuracy_ci_upper": ci.ci_upper,
            "held_balanced_accuracy_ci_lower": ci_bal.ci_lower,
            "held_balanced_accuracy_ci_upper": ci_bal.ci_upper,
            "held_auc_ci_lower": ci_auc.ci_lower,
            "held_auc_ci_upper": ci_auc.ci_upper,
            "held_backtest_sharpe": held_bt.sharpe_ratio,
            "held_backtest_max_drawdown": held_bt.max_drawdown,
            "held_backtest_hit_rate": held_bt.hit_rate,
            "held_backtest_turnover": held_bt.turnover,
            "held_backtest_total_return": held_bt.total_return,
        })

    return {
        "accuracy": held_acc,
        "balanced_accuracy": held_bal_acc,
        "auc": held_auc,
        "f1": held_f1,
        "logloss": held_ll,
        "ci": ci,
        "balanced_accuracy_ci": ci_bal,
        "auc_ci": ci_auc,
        "importance": importance,
        "confusion_matrix": cm,
        "backtest": held_bt,
        "backtest_cis": held_bt_cis,
        "model": model,
        "preds": held_preds,
        "probs": held_probs,
    }


# =========================================================================
# PHASE 11: Random-Label Walk-Forward Test
# =========================================================================
def phase11_random_label_walkforward(X_train, y_train, wf_cfg):
    _separator("PHASE 11: RANDOM-LABEL WALK-FORWARD TEST")
    from src.random_label_test import random_label_walkforward

    n_shuffles = int(os.getenv("NIFTY_RANDOM_LABEL_SHUFFLES", "5"))
    result = random_label_walkforward(X_train, y_train, n_shuffles=n_shuffles, wf_cfg=wf_cfg)
    print(f"\nReal OOF accuracy:   {result['real_accuracy']:.4f}")
    print(f"Random mean +/- std: {result['random_mean']:.4f} +/- {result['random_std']:.4f}")
    print(f"p-value:             {result['p_value']:.4f}")

    # Log to MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="random_label_test"):
        mlflow.log_metric("real_oof_accuracy", result["real_accuracy"])
        mlflow.log_metric("random_mean_accuracy", result["random_mean"])
        mlflow.log_metric("random_std_accuracy", result["random_std"])
        mlflow.log_metric("random_label_p_value", result["p_value"])
        mlflow.log_param("n_shuffles", result["n_shuffles"])

    return result


# =========================================================================
# PHASE 12: Final MLflow Logging
# =========================================================================
def phase12_final_logging(
    baseline_summary,
    oof_data,
    stats, bt_cis,
    held_back_results,
    bt,
    random_label_result,
):
    _separator("PHASE 12: FINAL MLFLOW SUMMARY LOGGING")

    oof_preds_for_report = oof_data.get("selected_preds", oof_data["preds"])
    oof_acc = float(accuracy_score(oof_data["true"], oof_preds_for_report))
    oof_auc = float(roc_auc_score(oof_data["true"], oof_data["probs"]))

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="final_summary"):
        # OOF metrics
        mlflow.log_metric("oof_accuracy", oof_acc)
        mlflow.log_metric("oof_auc", oof_auc)

        # Baselines
        for name, row in baseline_summary.iterrows():
            mlflow.log_metric(f"baseline_{name}_acc", row["mean"])

        # Statistical tests
        mlflow.log_metric("permutation_p_value", stats["permutation"].p_value)
        mlflow.log_metric("random_label_p_value", stats["random_label"].p_value)
        mlflow.log_metric("accuracy_ci_lower", stats["accuracy_ci"].ci_lower)
        mlflow.log_metric("accuracy_ci_upper", stats["accuracy_ci"].ci_upper)

        # DeLong test
        mlflow.log_metric("delong_z", stats["delong"].z_statistic)
        mlflow.log_metric("delong_p_value", stats["delong"].p_value)
        mlflow.log_metric("delong_model_auc", stats["delong"].auc_a)
        mlflow.log_metric("delong_random_auc", stats["delong"].auc_b)

        # Backtest
        mlflow.log_metric("backtest_sharpe", bt.sharpe_ratio)
        mlflow.log_metric("backtest_max_dd", bt.max_drawdown)
        mlflow.log_metric("backtest_hit_rate", bt.hit_rate)
        mlflow.log_metric("backtest_total_return", bt.total_return)
        mlflow.log_metric("backtest_probabilistic_sharpe", bt.probabilistic_sharpe)

        # Backtest CIs
        for key, (lo, hi) in bt_cis.items():
            mlflow.log_metric(f"{key}_lower", lo)
            mlflow.log_metric(f"{key}_upper", hi)

        # Held-back
        mlflow.log_metric("held_back_accuracy", held_back_results["accuracy"])
        mlflow.log_metric("held_back_balanced_accuracy", held_back_results["balanced_accuracy"])
        mlflow.log_metric("held_back_auc", held_back_results["auc"])
        mlflow.log_metric("held_back_backtest_sharpe", held_back_results["backtest"].sharpe_ratio)
        mlflow.log_metric("held_back_backtest_total_return", held_back_results["backtest"].total_return)

        # Random-label walk-forward
        mlflow.log_metric("random_label_wf_p_value", random_label_result["p_value"])
        mlflow.log_metric("random_label_wf_real_acc", random_label_result["real_accuracy"])

        mlflow.log_param("n_features", 12)
        mlflow.log_param("model", "LightGBM")
        mlflow.log_param("features", FEATURE_COLS)
        mlflow.log_param("decision_threshold", float(oof_data.get("selected_threshold", 0.50)))

    print("[OK] All results logged to MLflow")


# =========================================================================
# MAIN
# =========================================================================
def main():
    start_time = datetime.now()
    print(f"Pipeline started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Phase 1: Data Audit
    raw, audit_report = phase1_data_audit()

    # Phase 2: Features
    X, y = phase2_features()

    # Phase 3: Validation setup
    X_train, y_train, X_held, y_held, wf_cfg, folds = phase3_validation(X, y)

    # Phase 4: Baselines
    baseline_df, baseline_summary = phase4_baselines(X_train, y_train)

    # Phase 5: LightGBM walk-forward (with Optuna HPO) -- returns OOF data
    lgb_results, oof_data = phase5_lightgbm(X_train, y_train, wf_cfg)

    # Phase 6: Compute OOF forward returns
    oof_returns, oof_dates = phase6_oof_returns(X_train, oof_data)

    # Phase 7: Backtest
    decision_threshold = float(oof_data.get("selected_threshold", 0.50))
    final_params = select_final_params_from_walk_forward(lgb_results)
    print(f"\nFinal params selected from walk-forward only: {final_params}")

    bt, bh = phase7_backtest(oof_data["probs"], oof_returns, oof_dates, threshold=decision_threshold)

    # Phase 8: Statistical tests
    stats = phase8_statistics(
        oof_data["true"],
        oof_data.get("selected_preds", oof_data["preds"]),
        oof_data["probs"],
    )

    # Phase 9: Backtest bootstrap CIs
    bt_cis = phase9_backtest_bootstrap(oof_data["probs"], oof_returns, threshold=decision_threshold)

    # Phase 10: Held-back evaluation
    held_back = phase10_held_back(
        X_train,
        y_train,
        X_held,
        y_held,
        final_params=final_params,
        threshold=decision_threshold,
    )

    # Phase 11: Random-label walk-forward test
    random_label_result = phase11_random_label_walkforward(X_train, y_train, wf_cfg)

    # Phase 12: Final logging
    phase12_final_logging(
        baseline_summary, oof_data, stats, bt_cis, held_back, bt, random_label_result,
    )

    # Phase 13: Full Robustness Suite (7 analyses)
    _separator("PHASE 13: ROBUSTNESS SUITE (7 ANALYSES)")
    n_null = int(os.getenv("NIFTY_NULL_SHUFFLES", "10"))
    robustness = run_full_robustness_suite(
        X=X,
        y=y,
        oof_probs=oof_data["probs"],
        oof_true=oof_data["true"],
        oof_returns=oof_returns,
        oof_dates=oof_dates,
        threshold=decision_threshold,
        wf_cfg=wf_cfg,
        n_null_shuffles=n_null,
    )

    # Log robustness results to MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="robustness_suite"):
        # Sensitivity suite
        sens = robustness["sensitivity"]
        mlflow.log_metric("sensitivity_acc_mean", float(sens["accuracy"].mean()))
        mlflow.log_metric("sensitivity_acc_std", float(sens["accuracy"].std()))
        mlflow.log_metric("sensitivity_auc_mean", float(sens["auc"].mean()))
        mlflow.log_metric("sensitivity_n_splits", len(sens))

        # Calibration
        cal = robustness["calibration"]
        mlflow.log_metric("ece", cal.ece)

        # Cost sweep
        cs = robustness["cost_sweep"]
        mlflow.log_metric("sharpe_0bps", float(cs.loc[cs["cost_bps"] == 0, "sharpe"].iloc[0]) if 0 in cs["cost_bps"].values else 0.0)
        mlflow.log_metric("sharpe_5bps", float(cs.loc[cs["cost_bps"] == 5, "sharpe"].iloc[0]) if 5 in cs["cost_bps"].values else 0.0)
        mlflow.log_metric("sharpe_10bps", float(cs.loc[cs["cost_bps"] == 10, "sharpe"].iloc[0]) if 10 in cs["cost_bps"].values else 0.0)

        # Feature stability
        stab = robustness["feature_stability"]
        mlflow.log_metric("feature_rank_corr_mean", stab.mean_rank_correlation)
        mlflow.log_param("unstable_features", stab.unstable_features)

        # Null models
        for method, res in robustness["null_models"].items():
            mlflow.log_metric(f"null_{method}_p_value", res.p_value)
            mlflow.log_metric(f"null_{method}_mean", res.null_mean)

        # Trade filter — best filter by Sharpe
        tf = robustness["trade_filter"]
        best_filter = tf.loc[tf["filtered_sharpe"].idxmax()]
        mlflow.log_metric("best_filter_sharpe", float(best_filter["filtered_sharpe"]))
        mlflow.log_param("best_filter_name", best_filter["filter_name"])

    # -- Final Summary --
    _separator("PIPELINE COMPLETE")
    elapsed = datetime.now() - start_time
    oof_acc = float(accuracy_score(oof_data["true"], oof_data.get("selected_preds", oof_data["preds"])))
    oof_auc = float(roc_auc_score(oof_data["true"], oof_data["probs"]))

    print(f"Total time: {elapsed}")
    print(f"\n{'='*50}")
    print(f"  Feature matrix:          {X.shape}")
    print(f"  Walk-forward folds:      {len(folds)}")
    print(f"  OOF Accuracy:            {oof_acc:.4f}")
    print(f"  OOF AUC:                 {oof_auc:.4f}")
    print(f"  Permutation p-value:     {stats['permutation'].p_value:.4f}")
    print(f"  Random-label WF p-value: {random_label_result['p_value']:.4f}")
    print(f"  Backtest Sharpe:         {bt.sharpe_ratio:.4f}")
    print(f"  Backtest Max DD:         {bt.max_drawdown:.4%}")
    print(f"  Held-back Accuracy:      {held_back['accuracy']:.4f}")
    print(f"  Held-back AUC:           {held_back['auc']:.4f}")
    print(f"  --- Robustness ---")
    print(f"  ECE:                     {robustness['calibration'].ece:.4f}")
    print(f"  Feature rank corr:       {robustness['feature_stability'].mean_rank_correlation:.4f}")
    for method, res in robustness["null_models"].items():
        print(f"  Null ({method:20s}): p={res.p_value:.4f}")
    best_f = robustness["trade_filter"].loc[robustness["trade_filter"]["filtered_sharpe"].idxmax()]
    print(f"  Best filter:             {best_f['filter_name']} (Sharpe={best_f['filtered_sharpe']:.4f})")
    print(f"{'='*50}")

    return {
        "X": X, "y": y,
        "baselines": baseline_summary,
        "lgb_results": lgb_results,
        "oof_data": oof_data,
        "oof_returns": oof_returns,
        "backtest": bt,
        "stats": stats,
        "bt_cis": bt_cis,
        "held_back": held_back,
        "random_label": random_label_result,
        "robustness": robustness,
    }


if __name__ == "__main__":
    results = main()
