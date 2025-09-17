"""
Common performance metrics for backtests.

- Unless noted, functions expect period log returns. 
- Annualisation assumes 252 trading days by default.
"""

import numpy as np
import pandas as pd

def sharpe(returns: pd.Series, ann_factor: int = 252) -> float:
    """Annualised Sharpe ratio; 0 if stdev is zero."""
    mu = returns.mean()
    sd = returns.std(ddof=0)
    return np.sqrt(ann_factor) * (mu / sd) if sd > 0 else 0.0

def sortino(returns: pd.Series, ann_factor: int = 252) -> float:
    """Annualised Sortino using downside stdev; 0 if no downside."""
    downside = returns[returns < 0]
    dd = downside.std(ddof=0)
    mu = returns.mean()
    return np.sqrt(ann_factor) * (mu / dd) if dd > 0 else 0.0

def max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown (most negative drawdown) of an equity curve."""
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax
    return dd.min()

def calmar(returns: pd.Series, equity: pd.Series) -> float:
    """Calmar ratio = annualised return / |max drawdown|; 0 if no drawdown."""
    ann = returns.mean() * 252
    mdd = abs(max_drawdown(equity))
    return ann / mdd if mdd > 0 else 0.0

def profit_factor(returns: pd.Series) -> float:
    """Sum of gains divided by sum of losses; inf if no losses."""
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return (gains / losses) if losses > 0 else np.inf

def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with positive return."""
    n = len(returns)
    return (returns > 0).sum() / n if n > 0 else 0.0

def exposure(positions: pd.Series) -> float:
    """Fraction of periods with nonzero position."""
    n = len(positions)
    return (positions != 0).sum() / n if n > 0 else 0.0

def turnover(positions: pd.Series) -> float:
    """Sum of absolute position changes over time."""
    return positions.diff().abs().sum()

def var_es(returns: pd.Series, alpha: float = 0.05):
    """Return (VaR, ES) at level `alpha`; returns (0, 0) if empty input."""
    if len(returns) == 0:
        return 0.0, 0.0
    q = returns.quantile(alpha)
    es = returns[returns <= q].mean() if (returns <= q).any() else q
    return q, es

def summarise(log_returns: pd.Series, positions: pd.Series) -> dict:
    """Compute a concise metrics dict from log returns and positions.

    Builds an equity curve via `exp(cumsum)`, then computes annualised return,
    Sharpe/Sortino, drawdown metrics, win/loss stats, exposure/turnover, and
    5% VaR/ES.
    """
    equity = np.exp(log_returns.cumsum())
    var5, es5 = var_es(log_returns, alpha=0.05)
    out = {
        "ann_return_log": log_returns.mean() * 252,
        "sharpe": sharpe(log_returns),
        "sortino": sortino(log_returns),
        "max_dd": max_drawdown(equity),
        "calmar": calmar(log_returns, equity),
        "profit_factor": profit_factor(log_returns),
        "hit_rate": hit_rate(log_returns),
        "exposure": exposure(positions),
        "turnover": turnover(positions),
        "var_5": var5,
        "es_5": es5,
    }
    return out
