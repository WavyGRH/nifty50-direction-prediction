"""
Data audit for NIFTY 50 direction-prediction project.
Loads raw CSVs, confirms known quality issues, and produces a clean aligned dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── Known data-quality expectations ──────────────────────────────────────────
EXPECTED_VIX_MISSING = 5
EXPECTED_BN_MISSING = 1
EXPECTED_ZERO_VOL_DAYS = 8
SATURDAY_DATE = pd.Timestamp("2025-02-01")


def load_raw() -> dict[str, pd.DataFrame]:
    """Load and minimally parse the three raw OHLCV CSVs."""
    frames = {}
    for name, filename in [("nifty", "nifty50.csv"), ("banknifty", "banknifty.csv"), ("vix", "indiavix.csv")]:
        df = pd.read_csv(DATA_DIR / filename, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        frames[name] = df
    return frames


def audit(frames: dict[str, pd.DataFrame]) -> dict:
    """Run all quality checks; return a report dict and raise on unexpected findings."""
    nifty, bn, vix = frames["nifty"], frames["banknifty"], frames["vix"]
    nifty_dates = set(nifty["date"])
    report = {}

    # 1. VIX missing dates
    vix_missing = sorted(nifty_dates - set(vix["date"]))
    report["vix_missing_dates"] = vix_missing
    assert len(vix_missing) == EXPECTED_VIX_MISSING, (
        f"Expected {EXPECTED_VIX_MISSING} VIX-missing dates, got {len(vix_missing)}: {vix_missing}"
    )

    # 2. BankNifty missing dates
    bn_missing = sorted(nifty_dates - set(bn["date"]))
    report["bn_missing_dates"] = bn_missing
    assert len(bn_missing) == EXPECTED_BN_MISSING, (
        f"Expected {EXPECTED_BN_MISSING} BN-missing dates, got {len(bn_missing)}: {bn_missing}"
    )

    # 3. Zero-volume NIFTY days
    zero_vol = nifty.loc[nifty["Volume"] == 0, "date"].tolist()
    report["zero_volume_dates"] = zero_vol
    assert len(zero_vol) == EXPECTED_ZERO_VOL_DAYS, (
        f"Expected {EXPECTED_ZERO_VOL_DAYS} zero-volume days, got {len(zero_vol)}"
    )

    # 4. Saturday session
    saturdays = nifty[nifty["date"].dt.dayofweek == 5]
    report["saturday_dates"] = saturdays["date"].tolist()
    assert SATURDAY_DATE in saturdays["date"].values, (
        f"Expected Saturday session on {SATURDAY_DATE}"
    )

    return report


def clean(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Produce cleaned, NIFTY-date-aligned DataFrames.
    - Keep the Saturday session (it was a real NSE trading day).
    - Forward-fill VIX/BankNifty on their missing dates.
    - Mark zero-volume NIFTY days (volume set to NaN so features can handle it).
    """
    nifty = frames["nifty"].copy()
    bn = frames["banknifty"].copy()
    vix = frames["vix"].copy()

    # Align BankNifty and VIX to NIFTY's date index via left-join + forward-fill
    bn = (
        nifty[["date"]]
        .merge(bn, on="date", how="left")
        .sort_values("date")
    )
    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        bn[col] = bn[col].ffill()
    bn["Volume"] = bn["Volume"].fillna(0).astype(np.int64)

    vix = (
        nifty[["date"]]
        .merge(vix, on="date", how="left")
        .sort_values("date")
    )
    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        vix[col] = vix[col].ffill()
    vix["Volume"] = vix["Volume"].fillna(0).astype(np.int64)

    # Mark zero-volume NIFTY days — volume becomes NaN so downstream code can decide
    nifty.loc[nifty["Volume"] == 0, "Volume"] = np.nan

    return {"nifty": nifty, "banknifty": bn, "vix": vix}


def print_report(report: dict) -> None:
    print("=" * 60)
    print("DATA AUDIT REPORT")
    print("=" * 60)
    print(f"VIX missing {len(report['vix_missing_dates'])} dates: "
          f"{[str(d.date()) for d in report['vix_missing_dates']]}")
    print(f"BankNifty missing {len(report['bn_missing_dates'])} dates: "
          f"{[str(d.date()) for d in report['bn_missing_dates']]}")
    print(f"Zero-volume NIFTY days: {len(report['zero_volume_dates'])}")
    print(f"Saturday sessions: {[str(d.date()) for d in report['saturday_dates']]}")
    print("All checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    raw = load_raw()
    report = audit(raw)
    print_report(report)
    cleaned = clean(raw)
    for name, df in cleaned.items():
        print(f"\n{name}: {df.shape}, nulls={df.isnull().sum().sum()}")
