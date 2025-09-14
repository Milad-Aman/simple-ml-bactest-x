"""
Lightweight technical indicators used across features and strategies.
"""

import numpy as np
import pandas as pd

def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average with full-window requirement."""
    return series.rolling(window, min_periods=window).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with `adjust=False` meaning the recursive EMA is used."""
    return series.ewm(span=span, adjust=False).mean()

def ewm_vol(series: pd.Series, span: int = 30) -> pd.Series:
    """Exponentially Weighted Moving standard deviation of returns with span `span` (daily units)."""
    return series.ewm(span=span, adjust=False).std()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder-style RSI using EWMs of up/down moves."""
    delta = close.diff()
    up = (delta.clip(lower=0)).ewm(alpha=1 / period, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = up / (down.replace(0, np.nan))
    out = 100 - 100 / (1 + rs)
    return out

def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range over `window` periods."""
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()

def rolling_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation over `window` periods."""
    return returns.rolling(window, min_periods=window).std()

def donchian(high: pd.Series, low: pd.Series, lookback: int = 20):
    """Donchian channel: (upper, lower, mid) over `lookback`."""
    upper = high.rolling(lookback, min_periods=lookback).max()
    lower = low.rolling(lookback, min_periods=lookback).min()
    mid = (upper + lower) / 2.0
    return upper, lower, mid

def price_to_sma_distance(close: pd.Series, window: int) -> pd.Series:
    """Relative distance of price to its SMA: close / SMA - 1."""
    return (close / sma(close, window) - 1.0)
