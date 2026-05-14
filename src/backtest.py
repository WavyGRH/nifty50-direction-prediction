"""Long/flat trading strategy backtest for direction-prediction models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import kurtosis, norm, skew

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    """Container for backtest performance metrics."""

    sharpe_ratio: float
    max_drawdown: float
    hit_rate: float
    turnover: float
    total_return: float
    annualized_return: float
    n_trades: int
    equity_curve: pd.Series
    strategy_returns: pd.Series
    probabilistic_sharpe: float


def long_flat_backtest(
    probabilities: ArrayLike,
    returns: ArrayLike,
    dates: ArrayLike | None = None,
    cost_bps: float = 0.0,
    buy_threshold: float = 0.50,
) -> BacktestResult:
    """Backtest a long/flat strategy with confidence thresholding.

    Position = 1 (long) when probability >= buy_threshold, 0 (flat/cash) otherwise.
    Strategy return on day t = position[t] * return[t] - transaction_cost[t].

    Parameters
    ----------
    probabilities : array-like of float
        Model predicted probabilities of class 1.
        probabilities[t] is the signal generated at end of day t.
    returns : array-like of float
        Forward returns aligned with predictions.  returns[t] is the return
        earned from close of day t to close of day t+1, i.e. the return the
        strategy captures when predictions[t] == 1.
    dates : array-like, optional
        Date index for the equity curve.
    cost_bps : float
        One-way transaction cost in basis points (applied on each position change).
    """
    probs = np.asarray(probabilities, dtype=float)
    rets = np.asarray(returns, dtype=float)
    assert len(probs) == len(rets), "probabilities and returns must have equal length"

    positions = (probs >= buy_threshold).astype(float)

    trades = np.abs(np.diff(positions, prepend=0))
    cost_per_trade = cost_bps / 10_000
    costs = trades * cost_per_trade

    strategy_returns = positions * rets - costs

    equity = (1 + strategy_returns).cumprod()
    if dates is not None:
        equity = pd.Series(equity, index=pd.to_datetime(dates), name="equity")
        strategy_returns_series = pd.Series(
            strategy_returns, index=pd.to_datetime(dates), name="strategy_return"
        )
    else:
        equity = pd.Series(equity, name="equity")
        strategy_returns_series = pd.Series(strategy_returns, name="strategy_return")

    sharpe = _annualized_sharpe(strategy_returns)
    max_dd = _max_drawdown(equity.values)
    hit_rate = _hit_rate(positions, rets)
    turnover = float(trades.sum()) / max(len(probs), 1)

    total_ret = float(equity.iloc[-1] - 1)
    n_years = len(rets) / TRADING_DAYS_PER_YEAR
    ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0
    psr = probabilistic_sharpe_ratio(strategy_returns, benchmark_sharpe=0.0)

    return BacktestResult(
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        hit_rate=hit_rate,
        turnover=turnover,
        total_return=total_ret,
        annualized_return=ann_ret,
        n_trades=int(trades.sum()),
        equity_curve=equity,
        strategy_returns=strategy_returns_series,
        probabilistic_sharpe=psr,
    )


def _annualized_sharpe(strategy_returns: np.ndarray) -> float:
    if len(strategy_returns) < 2:
        return 0.0
    mu = strategy_returns.mean()
    sigma = strategy_returns.std(ddof=1)
    if sigma == 0:
        return 0.0
    return float(mu / sigma * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    return float(drawdown.min())


def _hit_rate(positions: np.ndarray, returns: np.ndarray) -> float:
    """Fraction of days the strategy was positioned long and the market went up,
    plus days it was flat and the market went down, over total days."""
    correct = (positions == 1) & (returns > 0) | (positions == 0) & (returns <= 0)
    return float(correct.mean())


def probabilistic_sharpe_ratio(
    returns: ArrayLike,
    benchmark_sharpe: float = 0.0,
) -> float:
    """Probability that the strategy Sharpe exceeds ``benchmark_sharpe``.

    This follows the Probabilistic Sharpe Ratio adjustment for non-normal
    returns used by Bailey and Lopez de Prado. It is not a substitute for a
    held-out test, but it is a useful sanity check for a short backtest.
    """
    rets = np.asarray(returns, dtype=float)
    rets = rets[np.isfinite(rets)]
    if len(rets) < 3:
        return 0.0

    daily_sharpe = _annualized_sharpe(rets) / np.sqrt(TRADING_DAYS_PER_YEAR)
    benchmark_daily = benchmark_sharpe / np.sqrt(TRADING_DAYS_PER_YEAR)
    skewness = skew(rets, bias=False)
    kurt = kurtosis(rets, fisher=False, bias=False)
    denom = np.sqrt(
        max(
            (1 - skewness * daily_sharpe + ((kurt - 1) / 4) * daily_sharpe**2)
            / (len(rets) - 1),
            1e-12,
        )
    )
    return float(norm.cdf((daily_sharpe - benchmark_daily) / denom))


def block_bootstrap_backtest_ci(
    probabilities: ArrayLike,
    returns: ArrayLike,
    dates: ArrayLike | None = None,
    cost_bps: float = 0.0,
    buy_threshold: float = 0.50,
    n_bootstrap: int = 1000,
    block_size: int = 5,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Block-bootstrap confidence intervals for backtest metrics.

    Sampling contiguous blocks is more appropriate for daily strategy returns
    than iid resampling because market returns can be serially dependent.
    """
    probs = np.asarray(probabilities, dtype=float)
    rets = np.asarray(returns, dtype=float)
    if len(probs) != len(rets):
        raise ValueError("probabilities and returns must have equal length")
    if len(probs) == 0:
        raise ValueError("cannot bootstrap an empty backtest")

    rng = np.random.default_rng(seed)
    n = len(probs)
    block_size = max(1, min(block_size, n))
    metrics = {
        "sharpe": [],
        "max_drawdown": [],
        "hit_rate": [],
        "turnover": [],
        "total_return": [],
        "probabilistic_sharpe": [],
    }

    for _ in range(n_bootstrap):
        starts = rng.integers(0, n, size=int(np.ceil(n / block_size)))
        idx = np.concatenate(
            [np.arange(start, min(start + block_size, n)) for start in starts]
        )[:n]
        sample_dates = None
        if dates is not None:
            sample_dates = np.arange(n)
        bt = long_flat_backtest(
            probs[idx],
            rets[idx],
            dates=sample_dates,
            cost_bps=cost_bps,
            buy_threshold=buy_threshold,
        )
        metrics["sharpe"].append(bt.sharpe_ratio)
        metrics["max_drawdown"].append(bt.max_drawdown)
        metrics["hit_rate"].append(bt.hit_rate)
        metrics["turnover"].append(bt.turnover)
        metrics["total_return"].append(bt.total_return)
        metrics["probabilistic_sharpe"].append(bt.probabilistic_sharpe)

    return {
        name: (
            float(np.percentile(values, 2.5)),
            float(np.percentile(values, 97.5)),
        )
        for name, values in metrics.items()
    }


def compare_strategies(model_preds: np.ndarray, baseline_preds: dict[str, np.ndarray], returns: np.ndarray, dates: np.ndarray, cost_bps: float = 5.0) -> pd.DataFrame:
    """Run backtest for a model and multiple baselines, return comparison table."""
    all_results = {}

    all_results["model"] = long_flat_backtest(model_preds, returns, dates, cost_bps)
    for name, preds in baseline_preds.items():
        all_results[name] = long_flat_backtest(preds, returns, dates, cost_bps)

    rows = []
    for name, res in all_results.items():
        rows.append(
            {
                "strategy": name,
                "sharpe_ratio": res.sharpe_ratio,
                "max_drawdown": res.max_drawdown,
                "hit_rate": res.hit_rate,
                "turnover": res.turnover,
                "total_return": res.total_return,
                "annualized_return": res.annualized_return,
                "n_trades": res.n_trades,
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


backtest_strategy = long_flat_backtest


def buy_and_hold(
    returns: ArrayLike, dates: ArrayLike | None = None
) -> BacktestResult:
    """Buy-and-hold benchmark: always long."""
    preds = np.ones(len(np.asarray(returns)), dtype=int)
    return long_flat_backtest(preds, returns, dates, cost_bps=0.0)
