from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Iterable, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from .d_engine import backtest_t1
from .c_metrics import summarise

# ----- Classifier -----

def build_logit(C: float = 0.5, penalty: str = "l1", max_iter: int = 500) -> LogisticRegression:
    return LogisticRegression(penalty=penalty, solver="liblinear", C=C, max_iter=max_iter)

def fit_calibrated(
    base_clf,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "isotonic",
    cv: int = 3,
):
    """Fit classifier + probability calibration on TRAIN only."""
    cal = CalibratedClassifierCV(base_clf, method=method, cv=cv)
    cal.fit(X_train, y_train)
    return cal

def predict_proba_both(calibrated_clf, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    p_train = pd.Series(calibrated_clf.predict_proba(X_train)[:, 1], index=X_train.index)
    p_test  = pd.Series(calibrated_clf.predict_proba(X_test)[:, 1],  index=X_test.index)
    return p_train, p_test

# ----- Policy (decision layer) -----

def policy_signals(proba: pd.Series, thr: float, min_hold: int = 1) -> pd.Series:
    """Two-sided policy with optional minimum hold."""
    raw = np.where(proba > thr,  1.0, np.where(proba < (1.0 - thr), -1.0, 0.0))
    if min_hold <= 1:
        return pd.Series(raw, index=proba.index)

    pos, cur, held = [], 0.0, 0
    for s in raw:
        if cur != 0 and s != cur and held < min_hold:
            s = cur
        if s != cur:
            cur, held = s, 1 if s != 0 else 0
        else:
            held = held + 1 if cur != 0 else 0
        pos.append(cur)
    return pd.Series(pos, index=proba.index)

def _violations(metrics: Dict[str, float], constraints: Dict) -> float:
    """Aggregate constraint violations to a single non-negative penalty."""
    v = 0.0
    lo, hi = constraints["exposure_range"]
    v += max(0.0, lo - metrics["exposure"]) + max(0.0, metrics["exposure"] - hi)
    v += max(0.0, metrics["turnover"] - constraints["turnover_max"]) / max(1e-9, constraints["turnover_max"])
    v += max(0.0, constraints["ann_return_log_min"] - metrics["ann_return_log"])
    # Optionally enforce floors on train Sharpe/Calmar, uncomment if desired:
    # v += max(0.0, constraints.get("sharpe_min", 0) - metrics["sharpe"])
    # v += max(0.0, constraints.get("calmar_min", 0) - metrics["calmar"])
    return v

def tune_policy_on_train(
    p_train: pd.Series,
    price_train: pd.Series,
    exe: Dict,
    constraints: Dict,
    thr_grid: Iterable[float] = np.linspace(0.50, 0.70, 41),
    min_hold_grid: Iterable[int] = (1, 3, 5),
) -> Tuple[float, int, Dict[str, float]]:
    """Search (thr, min_hold); feasibility first, then max Sharpe on TRAIN."""
    best, best_score = (0.55, 1, {}), -1e9
    ret_train = np.log(price_train).diff()  # for engine
    for thr in thr_grid:
        for mh in min_hold_grid:
            sig = policy_signals(p_train, thr, min_hold=mh)
            bt = backtest_t1(
                price_train, ret_train, sig,
                fees_bps=exe["fees_bps"], slippage_bps=exe["slippage_bps"],
                target_vol_annual=exe["target_vol_annual"], vol_lookback=exe["vol_lookback"],
                max_leverage=exe["max_leverage"]
            )
            m = summarise(bt["pnl_net"], bt["pos"])  # dict-like
            pen = _violations(m, constraints)
            score = (m["sharpe"] if pen == 0.0 else m["sharpe"] - 10.0 * pen)
            if score > best_score:
                best, best_score = (thr, mh, m), score
    thr_star, mh_star, mtrain = best
    return float(thr_star), int(mh_star), mtrain
