"""LightGBM training with Optuna HPO nested inside walk-forward folds, logged to MLflow."""

from __future__ import annotations

import warnings
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from src.validation import (
    FoldInfo,
    WalkForwardConfig,
    iter_fold_arrays,
    walk_forward_splits,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

SEED = 42
OPTUNA_N_TRIALS = 50
OPTUNA_INNER_FOLDS = 3

EXPERIMENT_NAME = "nifty50_direction"


def _inner_cv_score(
    params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> float:
    """Time-respecting inner CV on the training portion. Returns mean log-loss."""
    n = len(y_train)
    inner_fold_size = n // (OPTUNA_INNER_FOLDS + 1)
    scores = []

    for i in range(OPTUNA_INNER_FOLDS):
        val_end = n - i * inner_fold_size
        val_start = val_end - inner_fold_size
        tr_end = val_start
        if tr_end < inner_fold_size:
            continue

        ds_tr = lgb.Dataset(X_train[:tr_end], label=y_train[:tr_end])
        ds_val = lgb.Dataset(
            X_train[val_start:val_end],
            label=y_train[val_start:val_end],
            reference=ds_tr,
        )

        model = lgb.train(
            params,
            ds_tr,
            num_boost_round=500,
            valid_sets=[ds_val],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
        )
        preds = model.predict(X_train[val_start:val_end])
        scores.append(log_loss(y_train[val_start:val_end], preds))

    return float(np.mean(scores)) if scores else float("inf")


def _optuna_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> float:
    params: dict[str, Any] = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": SEED,
        "n_jobs": -1,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
    }
    return _inner_cv_score(params, X_train, y_train)


def _train_final_model(
    best_params: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> lgb.Booster:
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": SEED,
        "n_jobs": -1,
        **best_params,
    }
    n = len(y_train)
    split = int(n * 0.85)
    ds_tr = lgb.Dataset(X_train[:split], label=y_train[:split])
    ds_val = lgb.Dataset(X_train[split:], label=y_train[split:], reference=ds_tr)

    model = lgb.train(
        params,
        ds_tr,
        num_boost_round=1000,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
    )
    return model


def _evaluate(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
    }


def run_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: list[str] | None = None,
    wf_cfg: WalkForwardConfig | None = None,
    n_trials: int = OPTUNA_N_TRIALS,
) -> pd.DataFrame:
    wf_cfg = wf_cfg or WalkForwardConfig()
    folds = walk_forward_splits(len(X), wf_cfg)
    feature_names = feature_names or list(X.columns)

    mlflow.set_experiment(EXPERIMENT_NAME)
    results: list[dict[str, Any]] = []

    with mlflow.start_run(run_name="walk_forward") as parent_run:
        mlflow.log_params({
            "initial_train_days": wf_cfg.initial_train_days,
            "test_days": wf_cfg.test_days,
            "step_days": wf_cfg.step_days,
            "embargo_days": wf_cfg.embargo_days,
            "n_folds": len(folds),
            "n_optuna_trials": n_trials,
            "n_features": len(feature_names),
        })

        for fold, X_train, y_train, X_test, y_test in iter_fold_arrays(X, y, folds):
            print(f"--- Fold {fold.fold_idx} | train {fold.train_end - fold.train_start} | test {fold.test_end - fold.test_start} ---")

            with mlflow.start_run(
                run_name=f"fold_{fold.fold_idx}",
                nested=True,
            ):
                mlflow.log_params({
                    "fold_idx": fold.fold_idx,
                    "train_size": fold.train_end - fold.train_start,
                    "test_size": fold.test_end - fold.test_start,
                })

                study = optuna.create_study(
                    direction="minimize",
                    sampler=optuna.samplers.TPESampler(seed=SEED + fold.fold_idx),
                )
                study.optimize(
                    lambda trial: _optuna_objective(trial, X_train, y_train),
                    n_trials=n_trials,
                    show_progress_bar=False,
                )

                best_params = study.best_trial.params
                mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
                mlflow.log_metric("best_inner_logloss", study.best_value)

                model = _train_final_model(best_params, X_train, y_train)

                y_prob = model.predict(X_test)
                metrics = _evaluate(y_test, y_prob)
                mlflow.log_metrics(metrics)

                importance = dict(zip(feature_names, model.feature_importance(importance_type="gain")))
                top_features = sorted(importance, key=importance.get, reverse=True)[:10]
                mlflow.log_param("top_features", top_features)

                fold_result = {
                    "fold": fold.fold_idx,
                    "train_size": fold.train_end - fold.train_start,
                    **metrics,
                    "best_inner_logloss": study.best_value,
                    "n_estimators": model.num_trees(),
                }
                results.append(fold_result)
                print(f"   acc={metrics['accuracy']:.4f}  auc={metrics['roc_auc']:.4f}  logloss={metrics['log_loss']:.4f}")

        results_df = pd.DataFrame(results)
        summary = results_df[["accuracy", "f1", "roc_auc", "log_loss"]].mean().to_dict()
        mlflow.log_metrics({f"mean_{k}": v for k, v in summary.items()})
        print(f"\n=== Mean metrics: acc={summary['accuracy']:.4f}  auc={summary['roc_auc']:.4f}  logloss={summary['log_loss']:.4f} ===")

    # Collect OOF data across all folds
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    oof_indices = []
    oof_probs = []
    oof_true = []
    oof_preds = []
    fold_importances = []

    for fold, X_train, y_train, X_test, y_test in iter_fold_arrays(X, y, folds):
        best_params = results_df.iloc[fold.fold_idx].get("best_params", {})
        if not best_params:
            best_params = {
                "num_leaves": 15,
                "learning_rate": 0.05,
                "min_child_samples": 50,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
            }
        model = _train_final_model(best_params, X_train, y_train)
        probs = model.predict(X_test)
        oof_probs.append(probs)
        oof_true.append(y_test)
        oof_preds.append((probs >= 0.5).astype(int))
        oof_indices.append(np.arange(fold.test_start, fold.test_end))
        imp = dict(zip(feature_names, model.feature_importance(importance_type="gain")))
        fold_importances.append({"fold": fold.fold_idx, **imp})

    oof_data = {
        "true": np.concatenate(oof_true),
        "probs": np.concatenate(oof_probs),
        "preds": np.concatenate(oof_preds),
        "fold_indices": np.concatenate(oof_indices),
        "fold_importances": fold_importances,
    }

    return results_df, oof_data


def load_and_run(
    features_path: str = "D:\\nifty-direction-prediction\\data\\starter_features.csv",
    prices_path: str = "D:\\nifty-direction-prediction\\data\\nifty50.csv",
    n_trials: int = OPTUNA_N_TRIALS,
) -> pd.DataFrame:
    features = pd.read_csv(features_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
    prices = pd.read_csv(prices_path, parse_dates=["date"]).sort_values("date").reset_index(drop=True)

    prices["target"] = (prices["Close"].shift(-1) > prices["Close"]).astype(int)

    merged = features.merge(prices[["date", "target"]], on="date", how="inner").dropna(subset=["target"])
    merged = merged.dropna(axis=0, thresh=int(len(merged.columns) * 0.5))

    feature_cols = [c for c in merged.columns if c not in ("date", "target")]
    merged[feature_cols] = merged[feature_cols].ffill().fillna(0)

    X = merged[feature_cols]
    y = merged["target"].astype(int)

    return run_walk_forward(X, y, feature_names=feature_cols, n_trials=n_trials)


train_walk_forward = run_walk_forward


def select_final_params_from_walk_forward(results_df: pd.DataFrame) -> dict[str, Any]:
    """Select final model params from the best-performing walk-forward fold."""
    if "best_params" in results_df.columns:
        best_idx = results_df["roc_auc"].idxmax()
        return results_df.loc[best_idx, "best_params"]
    return {
        "num_leaves": 15,
        "learning_rate": 0.05,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }


def select_probability_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "balanced_accuracy",
    thresholds: list[float] | None = None,
) -> tuple[float, float]:
    """Select the probability threshold that maximizes the given metric on OOF data."""
    from sklearn.metrics import balanced_accuracy_score

    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.40, 0.65, 0.01)]

    best_threshold = 0.50
    best_score = 0.0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        if metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, preds)
        elif metric == "accuracy":
            score = accuracy_score(y_true, preds)
        elif metric == "f1":
            score = f1_score(y_true, preds, zero_division=0)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score


if __name__ == "__main__":
    results = load_and_run()
    print("\nPer-fold results:")
    print(results.to_string(index=False))
