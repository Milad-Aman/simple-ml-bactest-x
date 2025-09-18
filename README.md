# ML Backtest

Single-asset walk-forward pipeline that trains a calibrated logistic-regression classifier on technical features and routes its signals through a volatility-targeted execution engine with realistic transaction costs.

## Strategy

- **Data**: Yahoo Finance downloader or CSV ingestor normalises timestamps and appends log returns.
- **Features**: Rolling volatility, RSI, slow-SMA distance, and Donchian-channel location (lagged returns and fast-SMA distance are scaffolded but presently disabled).
- **Model**: \(\ell_1\)-penalised logistic regression with optional isotonic/Platt calibration, searching probability thresholds and minimum holding periods under exposure/turnover/return constraints.
- **Execution**: T+1 fills, EWMA volatility targeting, leverage caps, and basis-point cost model convert predictions to log-PnL and equity.
- **Validation**: Walk-forward rebuild of the full pipeline per fold with Monte-Carlo block-permutation Sharpe p-values and Wald–Wolfowitz runs tests on out-of-sample returns.

## Summary of what this repository achieves

- Walk-forward evaluation that avoids look-ahead while aggregating fold metrics, positions, and returns.
- Metric suite covering Sharpe, Sortino, Calmar, drawdowns, VaR/ES, turnover, and exposure.
- YAML-driven configuration for reproducible experiments via `configs/ml_lg.yaml`.
- Modular codebase (`ml_backtest/`) split across data loading, indicators, feature engineering, execution, models/policies, and statistical tests.
- Notebook (`main.ipynb`) to inspect outputs interactively after a run.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Then run main.ipynb in this virtual environment.

