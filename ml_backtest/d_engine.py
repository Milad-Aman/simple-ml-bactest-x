"""
Execution helpers for simple trading backtests.

Includes:
- `_as_series`: utility to coerce inputs to 1-D pandas Series.
- `backtest_t1`: 1-bar delayed execution with vol targeting and linear costs.

Conventions:
- Inputs are aligned to the same index; returns are log returns per period.
- Transaction costs are applied to absolute position changes in bps.
"""
import numpy as np
import pandas as pd
from .b_indicators import ewm_vol

def _as_series(x, idx=None, dtype=float) -> pd.Series:
    """Coerce input to a 1-D pandas Series.

    Preserves the provided `idx` if given; for DataFrames, requires exactly
    one column. Optionally casts to `dtype`.

    Args:
        x: Series, DataFrame (1 col), array-like, or scalar.
        idx: Optional index to reindex onto.
        dtype: Optional dtype for the resulting Series.

    Returns:
        pd.Series: 1-D series aligned to `idx` if provided.
    """
    if isinstance(x, pd.Series):
        s = x.copy()
    elif isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"Expected 1 column, got {x.shape[1]}")
        s = x.iloc[:, 0]
    else:
        # numpy array / list / scalar -> Series
        x = np.asarray(x)
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        if idx is None and isinstance(idx, (list, np.ndarray)):
            idx = pd.Index(idx)
        s = pd.Series(x, index=idx)
    if dtype is not None:
        s = s.astype(dtype)
    if idx is not None:
        s = s.reindex(idx)
    return s

def backtest_t1(
    prices,
    returns,
    raw_signal,
    fees_bps: float = 0.0,
    slippage_bps: float = 0.0,
    target_vol_annual: float = 0.15,
    vol_lookback: int = 30,
    max_leverage: float = 2.0,
) -> pd.DataFrame:
    """Vol-targeted backtest with 1-bar delay and linear costs.

    Workflow:
    - Coerce inputs to aligned Series on the same index.
    - Annualise EWMA vol from `returns` and compute scaling to hit
      `target_vol_annual`, capped by `max_leverage`.
    - Position `pos = clip(sig * scale, ±max_leverage)`.
    - PnL is computed with 1-bar delay: `pnl = pos.shift(1) * returns`.
    - Costs are applied to absolute position changes in bps
      (`fees_bps + slippage_bps`) converted to fraction.
    - Net pnl and equity are reported in log space and exp(cumsum), respectively.

    Args:
        prices: Price series aligned to `returns`.
        returns: Log returns per period.
        raw_signal: Target directional signal in [-1, 1].
        fees_bps: Fee in basis points per unit traded.
        slippage_bps: Slippage in basis points per unit traded.
        target_vol_annual: Target annualised volatility.
        vol_lookback: EWMA span for daily vol estimate.
        max_leverage: Cap for absolute position and scaling.

    Returns:
        pd.DataFrame with columns:
        - `price`, `ret`, `pos`, `pnl`, `costs`, `pnl_net`, `equity`.
    """
    # Coerce to 1-D Series
    returns = _as_series(returns)
    idx = returns.index
    prices = _as_series(prices, idx=idx)
    sig = _as_series(raw_signal, idx=idx).fillna(0.0)

    # Vol targeting (annualised EWMA)
    r = _as_series(returns)
    vol_d = ewm_vol(r, span=vol_lookback) * np.sqrt(252.0)
    scale = (target_vol_annual / vol_d).clip(0, max_leverage).fillna(0.0)
    pos = (sig * scale).clip(-max_leverage, max_leverage)

    # PnL in log-return space
    pnl = pos.shift(1).fillna(0.0) * returns

    # Transaction costs on position changes
    bps_to_frac = 1e-4
    dpos = pos.diff().abs()
    if len(dpos) > 0:
        dpos.iloc[0] = abs(pos.iloc[0])  # opening trade
    costs = dpos * ((fees_bps + slippage_bps) * bps_to_frac)

    pnl_net = pnl - costs

    out = pd.DataFrame(index=idx)
    out["price"] = prices
    out["ret"] = returns
    out["pos"] = pos
    out["pnl"] = pnl
    out["costs"] = costs
    out["pnl_net"] = pnl_net
    out["equity"] = np.exp(out["pnl_net"].cumsum())
    return out
