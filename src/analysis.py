"""Statistical analysis: bootstrap CIs, permutation test, random-label baseline, DeLong test.

Uses MLstatkit for bootstrap CIs and DeLong's test (pre-built, optimized).
Falls back to manual implementations for custom metric functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# ── MLstatkit integration ────────────────────────────────────────────────────
from MLstatkit import (
    Bootstrapping as _mlstatkit_bootstrap,
    Delong_test as _mlstatkit_delong,
    Permutation_test as _mlstatkit_permutation,
)

# Supported metric strings for local bootstrap CIs.
_SUPPORTED_METRICS = {"accuracy", "f1", "precision", "recall", "roc_auc"}


# ── Result containers ────────────────────────────────────────────────────────

@dataclass
class BootstrapCI:
    """Result of a bootstrap confidence interval estimation."""

    point_estimate: float
    ci_lower: float
    ci_upper: float
    alpha: float
    n_bootstrap: int
    bootstrap_distribution: np.ndarray | None = None


@dataclass
class PermutationTestResult:
    """Result of a permutation test."""

    observed_statistic: float
    p_value: float
    n_permutations: int
    null_distribution: np.ndarray


@dataclass
class RandomLabelResult:
    """Result of a random-label baseline comparison."""

    model_metric: float
    random_mean: float
    random_std: float
    percentile_rank: float
    p_value: float
    n_trials: int
    random_distribution: np.ndarray


@dataclass
class DeLongResult:
    """Result of a DeLong test comparing two models' AUCs."""

    z_statistic: float
    p_value: float
    auc_a: float
    auc_b: float
    ci_a: tuple[float, float] | None = None
    ci_b: tuple[float, float] | None = None
    method: str = "delong"


# ── Helper ───────────────────────────────────────────────────────────────────

def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def _metric_from_string(metric: str) -> Callable[[np.ndarray, np.ndarray], float]:
    if metric == "accuracy":
        return lambda y_true, y_pred: float(accuracy_score(y_true, y_pred))
    if metric == "f1":
        return lambda y_true, y_pred: float(f1_score(y_true, y_pred, zero_division=0))
    if metric == "precision":
        return lambda y_true, y_pred: float(precision_score(y_true, y_pred, zero_division=0))
    if metric == "recall":
        return lambda y_true, y_pred: float(recall_score(y_true, y_pred, zero_division=0))
    if metric == "roc_auc":
        return lambda y_true, y_score: float(roc_auc_score(y_true, y_score))
    raise ValueError(f"Unsupported metric string: {metric}")


# ── Bootstrap CI (MLstatkit-accelerated) ─────────────────────────────────────

def bootstrap_ci(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] | str = accuracy,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> BootstrapCI:
    """Compute bootstrap confidence interval for a metric.

    Uses a local implementation even for standard metric strings. This is a
    little slower than MLstatkit, but it exposes the bootstrap distribution and
    avoids dependency-specific metric conventions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence_level = 1.0 - alpha

    if isinstance(metric_fn, str):
        if metric_fn not in _SUPPORTED_METRICS:
            raise ValueError(f"Unsupported metric string: {metric_fn}")
        metric_callable = _metric_from_string(metric_fn)
    else:
        metric_callable = metric_fn

    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_callable(y_true, y_pred)

    scores = []
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if isinstance(metric_fn, str) and metric_fn == "roc_auc":
            if len(np.unique(y_true[idx])) < 2:
                continue
        scores.append(metric_callable(y_true[idx], y_pred[idx]))
    scores = np.asarray(scores, dtype=float)

    lower = np.percentile(scores, 100 * alpha / 2)
    upper = np.percentile(scores, 100 * (1 - alpha / 2))

    return BootstrapCI(
        point_estimate=float(point),
        ci_lower=float(lower),
        ci_upper=float(upper),
        alpha=alpha,
        n_bootstrap=n_bootstrap,
        bootstrap_distribution=scores,
    )


# ── DeLong's Test (MLstatkit) ────────────────────────────────────────────────

def delong_test(
    y_true: ArrayLike,
    prob_model_a: ArrayLike,
    prob_model_b: ArrayLike,
    alpha: float = 0.05,
    n_boot: int = 5000,
    seed: int = 42,
) -> DeLongResult:
    """Compare AUCs of two models using DeLong's test (via MLstatkit).

    Tests H0: AUC_A == AUC_B.
    Returns z-statistic, p-value, and per-model AUCs with CIs.
    """
    y_true = np.asarray(y_true)
    prob_a = np.asarray(prob_model_a)
    prob_b = np.asarray(prob_model_b)

    result = _mlstatkit_delong(
        y_true, prob_a, prob_b,
        alpha=1.0 - alpha,
        return_ci=True,
        return_auc=True,
        n_boot=n_boot,
        random_state=seed,
        verbose=0,
    )
    # Returns: (z, p_value, ci_A, ci_B, auc_A, auc_B, info)
    z, p_value, ci_a, ci_b, auc_a, auc_b, info = result

    return DeLongResult(
        z_statistic=float(z),
        p_value=float(p_value),
        auc_a=float(auc_a),
        auc_b=float(auc_b),
        ci_a=(float(ci_a[0]), float(ci_a[1])),
        ci_b=(float(ci_b[0]), float(ci_b[1])),
        method=info.get("method", "delong"),
    )


# ── Permutation Test ─────────────────────────────────────────────────────────

def permutation_test(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = accuracy,
    n_permutations: int = 1000,
    seed: int = 42,
) -> PermutationTestResult:
    """Two-sided permutation test for whether the model beats chance.

    H0: the association between predictions and true labels is no better
    than random. Under H0, we permute y_true and recompute the metric.
    p-value = fraction of permuted scores >= observed score.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    observed = metric_fn(y_true, y_pred)

    null_dist = np.empty(n_permutations)
    for i in range(n_permutations):
        y_perm = rng.permutation(y_true)
        null_dist[i] = metric_fn(y_perm, y_pred)

    p_value = float((null_dist >= observed).sum() + 1) / (n_permutations + 1)

    return PermutationTestResult(
        observed_statistic=observed,
        p_value=p_value,
        n_permutations=n_permutations,
        null_distribution=null_dist,
    )


# ── MLstatkit Permutation Test (model comparison) ────────────────────────────

def permutation_test_two_models(
    y_true: ArrayLike,
    prob_model_a: ArrayLike,
    prob_model_b: ArrayLike,
    metric_str: str = "accuracy",
    n_permutations: int = 1000,
    threshold: float = 0.5,
    seed: int = 42,
) -> dict:
    """Compare two models using MLstatkit's permutation test.

    Returns metrics for both models and a p-value testing whether
    the difference is statistically significant.
    """
    y_true = np.asarray(y_true)
    prob_a = np.asarray(prob_model_a)
    prob_b = np.asarray(prob_model_b)

    metric_a, metric_b, p_value, benchmark, samples_mean, samples_std = _mlstatkit_permutation(
        y_true, prob_a, prob_b,
        metric_str=metric_str,
        n_bootstraps=n_permutations,
        threshold=threshold,
        random_state=seed,
    )

    return {
        "metric_a": float(metric_a),
        "metric_b": float(metric_b),
        "p_value": float(p_value),
        "benchmark": float(benchmark),
        "samples_mean": float(samples_mean),
        "samples_std": float(samples_std),
    }


# ── Random-Label Test ────────────────────────────────────────────────────────

def random_label_test(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    p_up: float | None = None,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = accuracy,
    n_trials: int = 1000,
    seed: int = 42,
) -> RandomLabelResult:
    """Compare model performance against random predictions.

    Generates random binary predictions n_trials times (with class
    probability p_up from the data if not specified) and computes the
    metric each time.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    if p_up is None:
        p_up = float(y_true.mean())

    model_metric = metric_fn(y_true, y_pred)

    random_scores = np.empty(n_trials)
    for i in range(n_trials):
        random_preds = (rng.random(n) < p_up).astype(int)
        random_scores[i] = metric_fn(y_true, random_preds)

    percentile_rank = float((random_scores < model_metric).mean()) * 100
    p_value = float((random_scores >= model_metric).sum() + 1) / (n_trials + 1)

    return RandomLabelResult(
        model_metric=model_metric,
        random_mean=float(random_scores.mean()),
        random_std=float(random_scores.std()),
        percentile_rank=percentile_rank,
        p_value=p_value,
        n_trials=n_trials,
        random_distribution=random_scores,
    )


# ── Full Statistical Report ─────────────────────────────────────────────────

def full_statistical_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_prob: ArrayLike | None = None,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = accuracy,
    n_bootstrap: int = 2000,
    n_permutations: int = 1000,
    n_random_trials: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Run all statistical tests and return a summary dict.

    Uses MLstatkit for bootstrap CIs (accuracy, f1, roc_auc) when possible.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Bootstrap CIs via MLstatkit for standard metrics
    ci_acc = bootstrap_ci(y_true, y_pred, "accuracy", n_bootstrap, alpha, seed)
    ci_f1 = bootstrap_ci(y_true, y_pred, "f1", n_bootstrap, alpha, seed)

    # AUC bootstrap (needs probabilities)
    ci_auc = None
    if y_prob is not None:
        y_prob = np.asarray(y_prob)
        ci_auc = bootstrap_ci(y_true, y_prob, "roc_auc", n_bootstrap, alpha, seed)

    # Custom metric CI
    ci_custom = bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap, alpha, seed)

    # Permutation test
    perm = permutation_test(y_true, y_pred, metric_fn, n_permutations, seed)

    # Random-label test
    rand = random_label_test(y_true, y_pred, None, metric_fn, n_random_trials, seed)

    return {
        "point_estimate": ci_custom.point_estimate,
        "ci_lower": ci_custom.ci_lower,
        "ci_upper": ci_custom.ci_upper,
        "ci_alpha": alpha,
        "accuracy_ci": ci_acc,
        "f1_ci": ci_f1,
        "auc_ci": ci_auc,
        "permutation_p_value": perm.p_value,
        "random_label_p_value": rand.p_value,
        "random_label_mean": rand.random_mean,
        "random_label_std": rand.random_std,
        "model_percentile_vs_random": rand.percentile_rank,
        "bootstrap": ci_custom,
        "permutation": perm,
        "random_label": rand,
    }


run_statistical_tests = full_statistical_report
