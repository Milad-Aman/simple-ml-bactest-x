# SMA Backtesting (with ML & Robustness)

A Python-based backtesting framework that uses SMA crossovers (baseline) and optional ML-driven signals,
with proper walk-forward validation, transaction cost modeling, and statistical significance testing.

## Strategy

- **ML overlay**: Features (lagged returns, vol/ATR, RSI, SMA distance, Donchian position). Model: Logistic Regression
  Walk-forward validation and t→t+1 execution. Fees and slippage included.

## Summary of what this repository achieves

- Data: Yahoo Finance downloader or CSV.
- Indicators: SMA/EMA, RSI, rolling volatility, ATR, Donchian channels, price-to-SMA distance.
- Strategies: ML classifier overlay.
- Walk-forward evaluation with aggregated OOS metrics.
- Execution realism: next-bar trading, volatility targeting, fees/slippage, leverage cap.
- Metrics: Sharpe, Sortino, Calmar, max drawdown, profit factor, hit-rate, turnover, exposure, 5% VaR/ES.
- Bias & significance checks: Monte-Carlo permutation p-values for Sharpe (block-aware), Wald–Wolfowitz runs test.
- Outputs: equity & fold plots, CSVs for returns/positions, and summary JSON in `reports/`.
- Reproducible runs via YAML configs and `--seed`.

## Quickstart

```bash
python -m pip install -r requirements.txt

# ML overlay (LogisticRegression)
python -m ml_backtest.main --config configs/ml_rf.yaml
```
