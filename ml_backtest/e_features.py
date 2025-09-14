"""
Feature matrix construction for ML strategies.

Builds lagged returns, rolling volatility, RSI, distances to SMAs, and a
Donchian position feature from OHLCV input data.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from .b_indicators import rsi, price_to_sma_distance, donchian

def build_feature_matrix(
    df: pd.DataFrame,
    r_lags=(1, 2, 3, 5),
    vol_lookback=20,
    rsi_period=14,
    sma_fast=20,
    sma_slow=100,
    donchian_lb=55,
    price_col="Adj Close",
) -> pd.DataFrame:
    """Create a tabular feature matrix from price data.

    Features:
    - `r_lag_k`: log-return lagged by k periods for k in `r_lags`.
    - `vol_{vol_lookback}`: rolling stdev of log returns.
    - `rsi_{rsi_period}`: RSI of price.
    - `dist_sma_*`: price-to-SMA distance for fast and slow windows.
    - `donch_pos`: normalised position within Donchian channel.

    Args:
        df (pd.DataFrame): Input OHLCV with columns including `price_col`, High, Low.
        price_col (str): Column used for returns/indicators (default `Adj Close`).

    Returns:
        pd.DataFrame: Feature matrix indexed like `df`, with NaNs dropped.
    """
    X = pd.DataFrame(index=df.index)
    r = np.log(df[price_col]).diff()
    for L in r_lags:
        X[f"r_lag_{L}"] = r.shift(L)
    X[f"vol_{vol_lookback}"] = r.rolling(vol_lookback, min_periods=vol_lookback).std()
    X[f"rsi_{rsi_period}"] = rsi(df[price_col], rsi_period)
    X[f"dist_sma_{sma_fast}"] = price_to_sma_distance(df[price_col], sma_fast)
    X[f"dist_sma_{sma_slow}"] = price_to_sma_distance(df[price_col], sma_slow)
    u,l,m = donchian(df["High"], df["Low"], lookback=donchian_lb)
    X[f"donch_pos"] = (df[price_col] - m) / (u - l).replace(0, np.nan)
    return X.dropna()
