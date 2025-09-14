"""
Walkforward backtesting utilities.

This module implements a simple walkforward evaluation framework:
- `rolling_windows` yields sequential train/test index splits with a fixed
  training window size (`min_train`) and a fixed forward test window size
  (`test_size`). The window advances by `test_size` each fold.
- `walkforward_run` orchestrates strategy signal generation per fold, executes
  a simple trading engine, collects per-fold metrics, and concatenates the
  out-of-sample (OOS) returns, positions, and equity.

Strategies supported via `config["strategy"]["name"]`:
- "sma": Simple moving average crossover (see `ml_backtest/strategy/sma.py`).
- "donchian": Donchian channel breakout (see `ml_backtest/strategy/donchian.py`).
- "ml_classifier": Supervised classifier over engineered features
  (see `ml_backtest/ml/pipelines.py` and `ml_backtest/features/featureset.py`).

Execution parameters are read from `config["execution"]` and passed to
`backtest_t1` (see `ml_backtest/execution/engine.py`). The framework avoids lookahead by
fitting models and generating signals using only data within each fold.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from .d_engine import backtest_t1
from .e_features import build_feature_matrix
from .f_ml_pipelines import log_reg, fit_predict_proba
from .c_metrics import summarise

def rolling_windows(index, min_train: int, test_size: int):

    start = 0
    n = len(index)
    while True:
        train_end = start + min_train
        test_end = train_end + test_size
        if test_end > n:
            break
        train_idx = index[start:train_end]
        test_idx = index[train_end:test_end]
        yield train_idx, test_idx
        start += test_size


def walkforward_run(config, df: pd.DataFrame, rng=None): 

    if rng is None:
      rng = np.random.default_rng(config.get("seed", None))

    price = df["Adj Close"]
    ret = np.log(price).diff().dropna()
    df = df.loc[ret.index]

    min_train = int(config["walkforward"]["min_train"])
    test_size = int(config["walkforward"]["test_size"])

    folds = []
    oos_returns = []
    oos_positions = []
    oos_equity_df = []

    for ti, vi in rolling_windows(df.index, min_train=min_train, test_size=test_size):
        test = df.loc[vi]

        p = config["strategy"]["params"]
        X = build_feature_matrix(df, **p["features"])
        y = (np.log(df["Adj Close"]).diff().shift(-1) > 0).astype(int)
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]  # Get first column if DataFrame
        y = pd.Series(y, name="y", index=y.index if hasattr(y, 'index') else X.index)
        Xy = pd.concat([X, y], axis=1).dropna()
        train_idx = Xy.index.intersection(ti)
        test_idx  = Xy.index.intersection(vi)
        X_train, y_train = Xy.loc[train_idx].drop(columns=["y"]), Xy.loc[train_idx, "y"]
        X_test  = Xy.loc[test_idx].drop(columns=["y"])
        model = log_reg()
        proba = fit_predict_proba(model, X_train, y_train, X_test)
        thr = float(p.get("prob_threshold", 0.55))
        sig_test = pd.Series(0.0, index=test_idx)
        sig_test[proba > thr] = 1.0
        sig_test[proba < (1-thr)] = -1.0
        

        exe = config["execution"]
        bt_test = backtest_t1(test["Adj Close"], np.log(test["Adj Close"]).diff(), sig_test,
                              fees_bps=exe["fees_bps"],
                              slippage_bps=exe["slippage_bps"],
                              target_vol_annual=exe["target_vol_annual"],
                              vol_lookback=exe["vol_lookback"],
                              max_leverage=exe["max_leverage"])

        metrics = summarise(bt_test["pnl_net"], bt_test["pos"])
        metrics["start"] = str(test.index[0].date())
        metrics["end"] = str(test.index[-1].date())
        folds.append(metrics)
        oos_returns.append(bt_test["pnl_net"])
        oos_positions.append(bt_test["pos"])
        oos_equity_df.append(bt_test[["equity"]])

    folds_df = pd.DataFrame(folds)
    oos_ret = pd.concat(oos_returns).sort_index()
    oos_pos = pd.concat(oos_positions).sort_index()
    oos_eq = pd.concat(oos_equity_df).sort_index()
    return folds_df, oos_ret, oos_pos, oos_eq
