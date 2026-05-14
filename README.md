# NIFTY 50 Next-Day Direction Prediction

Predicting next-day close direction (up/down) of the NIFTY 50 index using a LightGBM classifier with walk-forward validation, built for the Quant Singularity Financial ML Intern screening task.

## Quick Start

```bash
# 1. Clone and install
git clone <repo-url>
cd nifty-direction-prediction
pip install -r requirements.txt

# 2. Run the full pipeline (data audit -> features -> walk-forward training -> backtest -> statistical tests -> held-back eval)
python main.py

# 3. View MLflow results
mlflow ui
# Then open   
```

## Project Structure

```
nifty-direction-prediction/
├── data/
│   ├── nifty50.csv              # NIFTY 50 OHLCV (2022-01-03 to 2025-12-31)
│   ├── banknifty.csv            # Bank NIFTY OHLCV (same range)
│   ├── indiavix.csv             # India VIX OHLC (same range)
│   ├── starter_features.csv     # Pre-built features (audited, partially used)
│   └── features_clean.csv       # Generated: final 12-feature matrix with target
├── src/
│   ├── data_audit.py            # Data quality checks and cleaning
│   ├── features.py              # 12 features from raw OHLCV
│   ├── validation.py            # Expanding-window walk-forward CV
│   ├── models.py                # LightGBM + Optuna HPO with MLflow logging
│   ├── baselines.py             # Majority, random, persistence baselines
│   ├── backtest.py              # Long/flat trading strategy backtest
│   ├── random_label_test.py     # Walk-forward random-label null test
│   ├── analysis.py              # Bootstrap CIs, permutation test, DeLong's test
│   ├── research_loop.py         # Feature screening with paired bootstrap
│   └── robustness.py            # 7-analysis robustness/diagnostic suite
├── notebooks/
│   ├── 01_data_audit.ipynb      # Data quality findings
│   └── 02_eda.ipynb             # EDA: correlations, distributions, target balance
├── report/
│   ├── correlation_heatmap.png  # Feature correlation matrix
│   ├── feature_distributions.png
│   ├── target_distribution.png
│   ├── feature_target_correlations.png
│   └── feature_timeseries.png
├── main.py                      # End-to-end 12-phase pipeline
├── requirements.txt             # Pinned dependencies
└── README.md
```

## Methodology

### Data Audit

All three raw datasets were audited for quality issues:

| Issue | Count | Action |
|-------|-------|--------|
| VIX missing dates | 5 | Forward-fill onto NIFTY date grid |
| Bank NIFTY missing dates | 1 | Forward-fill |
| NIFTY zero-volume days | 8 | Marked NaN, handled in features |
| Saturday special session (2025-02-01) | 1 | Kept (real NSE trading day) |
| Bank NIFTY Adj Close != Close | 248 rows | Used Adj Close for returns |
| `ma5_smooth_signal` in starter features | -- | **Dropped: look-ahead leakage** (nulls at END, not beginning; corr with next-day return = -0.64) |
| `ret_zscore` in starter features | -- | **Dropped: perfect collinearity** with `ret_1d` (r = 1.0000) |

### Features (12)

Selected via greedy forward selection from 41 candidates on walk-forward OOF accuracy, followed by backward elimination. All features are computable at end-of-day T.

| # | Feature | Category | Description |
|---|---------|----------|-------------|
| 1 | `gap_x_vol` | Interaction | Overnight gap x 20d volatility — gap magnitude relative to vol regime |
| 2 | `ret_1d_sign` | Direction | Binary: 1 if yesterday's return > 0, else 0 (encodes lag-1 autocorrelation) |
| 3 | `open_gap_sign` | Direction | Binary: 1 if overnight gap > 0, else 0 |
| 4 | `ret_1d_abs` | Magnitude | Absolute 1-day return (separates "how big" from "which way") |
| 5 | `ret_10d` | Momentum | 10-day simple return |
| 6 | `vol_5d` | Volatility | 5-day rolling std of returns (short-term vol) |
| 7 | `volume_ratio_20d` | Liquidity | Volume / 20-day mean volume |
| 8 | `vix_above_ma20` | Vol regime | Binary: 1 if VIX > its 20-day moving average |
| 9 | `bn_ret_1d_sign` | Cross-asset | Binary: 1 if Bank NIFTY closed up today |
| 10 | `vix_ret_1d_sign` | Vol direction | Binary: 1 if VIX fell today (fear receding) |
| 11 | `close_vs_low` | Microstructure | (Close - Low) / Close — intraday buying pressure |
| 12 | `high_low_range` | Volatility | (High - Low) / Close — same-day realized range |

Max pairwise |correlation| = 0.641 (ret_1d_abs vs high_low_range), well under the 0.85 constraint.

Key design decision: 5 of 12 features are binary. Research loop analysis found that LightGBM cannot reliably find the optimal split at exactly zero on continuous return features. Binary encoding of direction signals (ret_1d_sign, open_gap_sign) directly captures the lag-1 autocorrelation (+0.107) that is the strongest predictive signal in this dataset.

### Walk-Forward Validation

Expanding-window walk-forward (no k-fold, no tuning on held-back period):

- **Initial training window**: 252 days (1 year)
- **Test window**: 21 days (1 month, non-overlapping)
- **Step size**: 21 days (monthly refit)
- **Embargo**: 1 day (prevents label leakage)
- **Total folds**: 27
- **Hyperparameter search**: Optuna Bayesian optimization (50 trials/fold) with time-respecting inner CV

### Model

**LightGBM** binary classifier with per-fold Optuna HPO. Key search dimensions: learning rate, num_leaves, max_depth, min_child_samples, subsample, colsample_bytree, L1/L2 regularization, min_split_gain.

Inner CV uses time-respecting reverse-chronological splits (3 inner folds) with early stopping on validation log-loss.

### Baselines

Three baselines evaluated across all walk-forward folds:

| Baseline | Mean Accuracy |
|----------|---------------|
| Majority class | ~53.3% |
| Random (class-prior) | ~49.4% |
| Persistence (yesterday's direction) | ~55.2% |

### Backtest

Long/flat strategy: go long when model P(up) >= threshold, flat otherwise. Includes configurable transaction costs (default 5bps one-way). Metrics: Sharpe ratio, max drawdown, hit rate, turnover, with bootstrap 95% CIs.

### Statistical Tests

- **Bootstrap CIs** on all metrics (accuracy, F1, AUC) via MLstatkit
- **DeLong's test** comparing model AUC vs random baseline
- **Permutation test** (1000 permutations) for H0: no association
- **Random-label walk-forward test** (full walk-forward with shuffled labels as null distribution)
- **Probabilistic Sharpe Ratio** for backtest significance

### Held-Back Evaluation

Jul-Dec 2025 (125 samples) held completely out of all training and hyperparameter selection. Model trained once on full training period with conservative defaults, evaluated in a single pass. No tuning on this window.

## Pipeline Phases

`main.py` runs 13 sequential phases:

1. **Data audit** — verify all quality issues, clean and align
2. **Feature engineering** — compute 12 features, validate correlations
3. **Validation setup** — define walk-forward folds, split train/held-back
4. **Baselines** — majority, random, persistence across folds
5. **LightGBM walk-forward** — Optuna HPO + training per fold, OOF predictions
6. **OOF forward returns** — compute daily returns for backtest
7. **Backtest** — long/flat strategy with costs, buy-and-hold benchmark
8. **Statistical tests** — bootstrap CIs, permutation, random-label, DeLong
9. **Backtest bootstrap CIs** — confidence intervals on Sharpe, max DD, hit rate
10. **Held-back evaluation** — single-pass eval on Jul-Dec 2025
11. **Random-label walk-forward** — null distribution via label shuffling
12. **Final MLflow logging** — summary run with all metrics
13. **Robustness suite** — 7 post-pipeline analyses (see below)

### Robustness Suite (Phase 13)

Seven diagnostic analyses in `src/robustness.py`:

| # | Analysis | What it tests |
|---|----------|---------------|
| 1 | Purged/embargoed sensitivity | Fixed-recipe refits across 6 split dates — checks if results are boundary-sensitive |
| 2 | Probability calibration | Reliability diagrams + ECE, segmented by volatility regime |
| 3 | Threshold-free decile analysis | Performance by probability decile — no single cutoff dependency |
| 4 | Cost sweep (0–15 bps) | Sharpe/return sensitivity to transaction costs, finds breakeven |
| 5 | Feature stability | Per-fold importance rank correlation + importance by volatility regime |
| 6 | Better null model | Block-shuffled and regime-preserving label permutation (not just iid) |
| 7 | Trade/no-trade filter | Confidence + volatility filters to skip low-conviction trades |

## Experiment Tracking

All runs logged to MLflow from the first experiment. View with:

```bash
mlflow ui
```

Logged artifacts include: per-fold metrics, hyperparameter trials, feature importance, best parameters, OOF/held-back metrics, baseline comparisons, statistical test results.

## Requirements

- Python 3.10+
- See `requirements.txt` for pinned dependencies

Key libraries: LightGBM, Optuna, MLflow, MLstatkit (bootstrap CIs, DeLong's test), mlfinpy (purged CV), scikit-learn, pandas, numpy, matplotlib, seaborn.

## Reproducibility

- All random seeds fixed (`SEED = 42`)
- Walk-forward splits are deterministic
- Optuna sampler seeded per fold (`SEED + fold_idx`)
- No hardcoded paths — all relative to project root

## References

- Lopez de Prado (2018), *Advances in Financial Machine Learning*
- Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio"
- Efron & Tibshirani (1993), *An Introduction to the Bootstrap*
- Raschka (2022), "Confidence Intervals for Machine Learning Classifiers"
