"""Random-label baseline: train LightGBM on shuffled labels to establish null distribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.validation import walk_forward_splits, iter_fold_arrays, WalkForwardConfig


SEED = 42


def _train_predict_lgb(X_train, y_train, X_test):
    """Train a default LightGBM and return predicted probabilities."""
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "seed": SEED,
        "n_jobs": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_child_samples": 30,
    }
    n = len(y_train)
    split = int(n * 0.85)
    ds_tr = lgb.Dataset(X_train[:split], label=y_train[:split])
    ds_val = lgb.Dataset(X_train[split:], label=y_train[split:], reference=ds_tr)

    model = lgb.train(
        params, ds_tr,
        num_boost_round=300,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)],
    )
    return model.predict(X_test)


def random_label_walkforward(
    X: pd.DataFrame,
    y: pd.Series,
    n_shuffles: int = 20,
    wf_cfg: WalkForwardConfig | None = None,
    seed: int = SEED,
) -> dict:
    """Run walk-forward with real labels AND n_shuffles random-label variants.

    Returns
    -------
    dict with keys:
        real_accuracy : float  — OOF accuracy with real labels
        random_accuracies : list[float] — OOF accuracy for each shuffle
        p_value : float — fraction of random runs >= real accuracy
        real_oof_preds : np.ndarray
        real_oof_true : np.ndarray
    """
    wf_cfg = wf_cfg or WalkForwardConfig()
    folds = walk_forward_splits(len(X), wf_cfg)
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    rng = np.random.default_rng(seed)

    # --- Real labels walk-forward ---
    real_preds_all = []
    real_true_all = []
    for fold, X_tr, y_tr, X_te, y_te in iter_fold_arrays(X, y, folds):
        probs = _train_predict_lgb(X_tr, y_tr, X_te)
        real_preds_all.append((probs >= 0.5).astype(int))
        real_true_all.append(y_te)

    real_preds_all = np.concatenate(real_preds_all)
    real_true_all = np.concatenate(real_true_all)
    real_accuracy = float((real_preds_all == real_true_all).mean())

    # --- Random label shuffles ---
    random_accuracies = []
    for i in range(n_shuffles):
        y_shuffled = rng.permutation(y_arr)
        y_shuffled_series = pd.Series(y_shuffled, index=y.index)
        shuffle_preds = []
        shuffle_true = []
        for fold, X_tr, y_tr, X_te, y_te in iter_fold_arrays(X, y_shuffled_series, folds):
            probs = _train_predict_lgb(X_tr, y_tr, X_te)
            shuffle_preds.append((probs >= 0.5).astype(int))
            shuffle_true.append(y_te)
        shuffle_preds = np.concatenate(shuffle_preds)
        shuffle_true = np.concatenate(shuffle_true)
        acc = float((shuffle_preds == shuffle_true).mean())
        random_accuracies.append(acc)
        print(f"  Shuffle {i+1}/{n_shuffles}: accuracy={acc:.4f}")

    p_value = float(sum(1 for ra in random_accuracies if ra >= real_accuracy) + 1) / (n_shuffles + 1)

    return {
        "real_accuracy": real_accuracy,
        "random_accuracies": random_accuracies,
        "random_mean": float(np.mean(random_accuracies)),
        "random_std": float(np.std(random_accuracies)),
        "p_value": p_value,
        "n_shuffles": n_shuffles,
        "real_oof_preds": real_preds_all,
        "real_oof_true": real_true_all,
    }


if __name__ == "__main__":
    from src.features import build_features

    X, y = build_features()
    result = random_label_walkforward(X, y, n_shuffles=10)
    print(f"\nReal OOF accuracy:   {result['real_accuracy']:.4f}")
    print(f"Random mean±std:     {result['random_mean']:.4f} ± {result['random_std']:.4f}")
    print(f"p-value:             {result['p_value']:.4f}")
