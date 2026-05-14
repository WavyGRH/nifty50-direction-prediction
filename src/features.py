"""Feature engineering for NIFTY 50 next-day direction prediction.

All final features are computed from the provided daily OHLCV/VIX files and
are safely available after the close of day T. The starter feature file is
treated as an untrusted vendor file and is audited separately rather than used
directly in the model.

Feature selection methodology: greedy forward selection from 41 candidates on
walk-forward OOF accuracy, followed by backward elimination. The core 10
features all individually contribute; 2 additional features (open_gap_sign,
high_low_range) are included to fill the 12-feature cap with minimal dilution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_audit import audit, clean, load_raw

FEATURE_COLS = [
    "gap_x_vol",
    "ret_1d_sign",
    "open_gap_sign",
    "ret_1d_abs",
    "ret_10d",
    "vol_5d",
    "volume_ratio_20d",
    "vix_above_ma20",
    "bn_ret_1d_sign",
    "vix_ret_1d_sign",
    "close_vs_low",
    "high_low_range",
]

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "features_clean.csv"


def _build_feature_df(cleaned: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build the full feature frame before final NA filtering."""
    nifty = cleaned["nifty"].copy()
    bn = cleaned["banknifty"].copy()
    vix = cleaned["vix"].copy()

    df = pd.DataFrame({"date": nifty["date"]})
    close = nifty["Close"]

    ret_1d = close.pct_change(1)
    open_gap = (nifty["Open"] - close.shift(1)) / close.shift(1)
    vol_20d = ret_1d.rolling(20).std()
    bn_ret_1d = bn["Adj Close"].pct_change(1)
    vix_close = vix["Close"]

    # Interaction: overnight gap scaled by current volatility regime.
    # Top-ranked solo feature (55.6% OOF accuracy); captures gap magnitude
    # relative to recent vol — large gaps in calm markets carry more signal.
    df["gap_x_vol"] = open_gap * vol_20d

    # Binary direction features. LightGBM cannot reliably split continuous
    # features at exactly zero. Lag-1 direction autocorrelation (+0.107,
    # runs-test z=-3.35) is the strongest signal; binary encoding lets trees
    # use it without threshold search error.
    df["ret_1d_sign"] = (ret_1d > 0).astype(float)
    df["open_gap_sign"] = (open_gap > 0).astype(float)

    # Return magnitude without direction — separates "how big" from "which way".
    df["ret_1d_abs"] = ret_1d.abs()

    # Medium-term momentum (2-week lookback).
    df["ret_10d"] = close.pct_change(10)

    # Short-term realized volatility (1-week). More responsive than vol_20d;
    # captures sudden vol spikes that shift next-day direction odds.
    df["vol_5d"] = ret_1d.rolling(5).std()

    # Volume anomaly detector.
    volume = nifty["Volume"].astype(float)
    volume_ma20 = volume.rolling(20, min_periods=1).mean()
    df["volume_ratio_20d"] = volume / volume_ma20

    # VIX regime: is implied vol elevated relative to its own trend?
    vix_ma20 = vix_close.rolling(20, min_periods=5).mean()
    df["vix_above_ma20"] = (vix_close > vix_ma20).astype(float).values

    # Cross-asset direction agreement: did Bank NIFTY close up today?
    df["bn_ret_1d_sign"] = (bn_ret_1d > 0).astype(float)

    # Implied-vol direction: did VIX fall today? (fear receding = bullish signal)
    vix_change = vix_close.pct_change(1)
    df["vix_ret_1d_sign"] = (vix_change < 0).astype(float).values

    # Intraday buying pressure: how far did the close finish above the low?
    df["close_vs_low"] = (close - nifty["Low"]) / close

    # Same-day realized range as a volatility/stress proxy.
    df["high_low_range"] = (nifty["High"] - nifty["Low"]) / close

    # Target: next-day close direction. Flat days mapped to down bucket.
    df["target"] = (close.shift(-1) > close).astype(float)
    df.loc[df.index[-1], "target"] = np.nan

    return df


def build_features() -> tuple[pd.DataFrame, pd.Series]:
    """Run the feature pipeline and return ``(X, y)`` with a DatetimeIndex."""
    raw = load_raw()
    audit(raw)
    cleaned = clean(raw)
    df = _build_feature_df(cleaned)

    out = df[["date", *FEATURE_COLS, "target"]].dropna().reset_index(drop=True)
    X = out[FEATURE_COLS].copy()
    X.index = pd.DatetimeIndex(out["date"])
    y = pd.Series(out["target"].values.astype(int), index=X.index, name="target")

    return X, y


def main() -> tuple[pd.DataFrame, pd.Series]:
    X, y = build_features()

    print(f"Clean feature matrix: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Date range: {X.index.min().date()} to {X.index.max().date()}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"NaN count: {int(X.isna().sum().sum())}")
    print(f"\nFeature stats:\n{X.describe().T[['mean', 'std', 'min', 'max']]}")

    out = X.copy()
    out["target"] = y
    out.to_csv(OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")

    return X, y


if __name__ == "__main__":
    main()
