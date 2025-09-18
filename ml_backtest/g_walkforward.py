"""Walkforward backtesting helpers."""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .c_metrics import summarise
from .d_engine import backtest_t1
from .e_features import build_feature_matrix
from .e_orthogonalise import fit_scaler_on_train, transform_with_scaler, orthogonalise_on_base
from .f_model_policy import build_logit, fit_calibrated, predict_proba_both, tune_policy_on_train, policy_signals


def rolling_windows(index, min_train: int, test_size: int):
    """Yield rolling train/test index pairs."""
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


@dataclass
class WalkforwardFold:
    """Store artefacts for a walkforward fold."""

    train_index: pd.Index
    test_index: pd.Index
    threshold: float
    min_hold: int
    train_metrics: dict
    oos_metrics: dict
    proba_train: pd.Series
    proba_test: pd.Series
    backtest: pd.DataFrame


def _init_rng(config, rng):
    """Return rng seeded from config when missing."""
    if rng is not None:
        return rng
    return np.random.default_rng(config.get("seed", None))


def _walkforward_settings(config):
    """Extract walkforward, strategy, and execution settings."""
    min_train = int(config["walkforward"]["min_train"])
    test_size = int(config["walkforward"]["test_size"])
    strat_params = config.get("strategy", {}).get("params", {})
    policy_cfg = strat_params.get("policy", {})
    exe = config["execution"]
    return min_train, test_size, strat_params, policy_cfg, exe


def _prepare_indices(Xy, train_window, test_window, min_train):
    """Compute train and test indices or return None."""
    train_idx = Xy.index.intersection(train_window)
    test_idx = Xy.index.intersection(test_window)
    if len(train_idx) < min_train or len(test_idx) == 0:
        return None
    return train_idx, test_idx


def _split_features(Xy, train_idx, test_idx):
    """Split features and labels for train and test slices."""
    X_train = Xy.loc[train_idx].drop(columns=["y"])
    y_train = Xy.loc[train_idx, "y"]
    if y_train.nunique() < 2:
        return None
    X_test = Xy.loc[test_idx].drop(columns=["y"])
    return X_train, y_train, X_test


def _scale_and_orthogonalise(X_train, X_test):
    """Scale features then orthogonalise targets."""
    scaler, X_train_s = fit_scaler_on_train(X_train)
    X_test_s = transform_with_scaler(X_test, scaler)
    base_cols = []
    base_cols += [c for c in X_train_s.columns if c.startswith("dist_sma_")]
    base_cols += [c for c in X_train_s.columns if c.startswith("vol_")]
    base_cols = list(dict.fromkeys(base_cols))[:2]
    targets = [c for c in X_train_s.columns if c not in base_cols]
    X_train_o, X_test_o, _ = orthogonalise_on_base(X_train_s, X_test_s, base_cols, target_cols=targets)
    return X_train_o, X_test_o


def _fit_probabilities(X_train_o, y_train, X_test_o, strat_params):
    """Train calibrated classifier and return probabilities."""
    clf_base = build_logit(C=0.5, penalty="l1", max_iter=500)
    calibration_method = strat_params.get("calibration", "isotonic")
    cal = fit_calibrated(clf_base, X_train_o, y_train, method=calibration_method)
    return predict_proba_both(cal, X_train_o, X_test_o)


def _policy_grids(policy_cfg):
    """Build policy search grids."""
    thr_grid_cfg = policy_cfg.get("thr_grid", (0.50, 0.70, 0.01))
    if isinstance(thr_grid_cfg, np.ndarray):
        thr_grid_cfg = thr_grid_cfg.tolist()
    if isinstance(thr_grid_cfg, (list, tuple)):
        thr_grid = np.arange(*thr_grid_cfg)
    else:
        thr_grid = np.arange(0.50, 0.70, 0.01)
    if thr_grid.size == 0:
        thr_grid = np.arange(0.50, 0.70, 0.01)

    min_hold_grid = policy_cfg.get("min_hold_grid", (1, 3, 5))
    if isinstance(min_hold_grid, np.ndarray):
        min_hold_grid = min_hold_grid.tolist()
    if isinstance(min_hold_grid, (int, float)):
        min_hold_grid = [int(min_hold_grid)]
    if len(min_hold_grid) == 0:
        min_hold_grid = [1]
    return thr_grid, min_hold_grid


def _build_fold_backtest(test, p_test, thr_star, mh_star, exe):
    """Run policy signals and execution backtest for the test window."""
    sig_test = policy_signals(p_test, thr_star, min_hold=mh_star)
    sig_test = sig_test.reindex(test.index, fill_value=0.0)
    bt_test = backtest_t1(
        test["Adj Close"],
        np.log(test["Adj Close"]).diff(),
        sig_test,
        fees_bps=exe["fees_bps"],
        slippage_bps=exe["slippage_bps"],
        target_vol_annual=exe["target_vol_annual"],
        vol_lookback=exe["vol_lookback"],
        max_leverage=exe["max_leverage"],
    )
    bt_test_clean = bt_test.dropna(subset=["pnl_net"])
    if bt_test_clean.empty:
        return None
    return bt_test_clean


def _metrics_from_backtest(bt_frame, test):
    """Build summary metrics from the cleaned backtest."""
    metrics = summarise(bt_frame["pnl_net"], bt_frame["pos"])
    metrics["start"] = str(test.index[0].date())
    metrics["end"] = str(test.index[-1].date())
    return metrics


def prepare_walkforward_inputs(config, df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Return price series and feature/label data for the walkforward run."""
    strat_params = config.get("strategy", {}).get("params", {})
    feature_params = strat_params.get("features", {})

    price = df["Adj Close"]
    if isinstance(price, pd.DataFrame):
        price = price.iloc[:, 0]

    log_price = np.log(price)
    ret = log_price.diff().dropna()
    df_aligned = df.loc[ret.index]

    X = build_feature_matrix(df_aligned, **feature_params)

    next_ret = log_price.diff().shift(-1)
    y = pd.Series((next_ret > 0).astype(int), index=next_ret.index, name="y")
    Xy = pd.concat([X, y], axis=1).dropna()

    price_clean = price.loc[Xy.index]
    return price_clean, Xy


def iter_walkforward_folds(prepped, config, rng=None):
    """Yield walkforward folds built from prepared inputs."""
    rng = _init_rng(config, rng)
    price = prepped["price"]
    Xy = prepped["Xy"]
    df = prepped["df"]
    min_train, test_size, strat_params, policy_cfg, exe = _walkforward_settings(config)

    for train_window, test_window in rolling_windows(price.index, min_train=min_train, test_size=test_size):
        test = df.loc[test_window]

        idx_pair = _prepare_indices(Xy, train_window, test_window, min_train)
        if idx_pair is None:
            continue
        train_idx, test_idx = idx_pair

        split = _split_features(Xy, train_idx, test_idx)
        if split is None:
            continue
        X_train, y_train, X_test = split

        X_train_o, X_test_o = _scale_and_orthogonalise(X_train, X_test)
        p_train, p_test = _fit_probabilities(X_train_o, y_train, X_test_o, strat_params)

        thr_grid, min_hold_grid = _policy_grids(policy_cfg)
        constraints = policy_cfg.get("constraints", {})
        thr_star, mh_star, mtrain = tune_policy_on_train(
            p_train,
            df.loc[train_idx, "Adj Close"],
            exe,
            constraints,
            thr_grid=thr_grid,
            min_hold_grid=min_hold_grid,
        )

        bt_test_clean = _build_fold_backtest(test, p_test, thr_star, mh_star, exe)
        if bt_test_clean is None:
            continue

        metrics = _metrics_from_backtest(bt_test_clean, test)

        yield WalkforwardFold(
            train_index=train_idx,
            test_index=test_idx,
            threshold=thr_star,
            min_hold=mh_star,
            train_metrics=mtrain,
            oos_metrics=metrics,
            proba_train=p_train,
            proba_test=p_test,
            backtest=bt_test_clean,
        )


def walkforward_run(config, df: pd.DataFrame, rng=None, collect_debug: bool = False):
    """Run the walkforward pass and collect outputs."""
    rng = _init_rng(config, rng)

    price, Xy = prepare_walkforward_inputs(config, df)
    df = df.loc[price.index]

    prepped = {"price": price, "Xy": Xy, "df": df}

    folds = []
    oos_returns = []
    oos_positions = []
    oos_equity_df = []
    collected_folds = []

    for fold in iter_walkforward_folds(prepped, config, rng=rng):
        folds.append(dict(fold.oos_metrics))
        oos_returns.append(fold.backtest["pnl_net"])
        oos_positions.append(fold.backtest["pos"])
        oos_equity_df.append(fold.backtest[["equity"]])
        if collect_debug:
            collected_folds.append(fold)

    if not folds:
        raise ValueError("Walkforward produced no valid folds; check config or data sufficiency.")

    folds_df = pd.DataFrame(folds)
    oos_ret = pd.concat(oos_returns).sort_index()
    oos_pos = pd.concat(oos_positions).sort_index()
    oos_eq = pd.concat(oos_equity_df).sort_index()

    if collect_debug:
        return folds_df, oos_ret, oos_pos, oos_eq, collected_folds
    return folds_df, oos_ret, oos_pos, oos_eq
