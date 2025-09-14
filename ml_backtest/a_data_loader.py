"""
Utilities for loading and preparing market data.

This module:
- Loads price data from Yahoo Finance (`load_yahoo`) or a CSV file (`load_csv`).
- Normalises indices to timezone-aware UTC `DatetimeIndex` for consistency.
- Adds log returns to a DataFrame (`add_log_returns`).

Conventions:
- Expected columns for price data are OHCLV: `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.
- Indices are converted to UTC and sorted ascending by timestamp.

Notes:
- `load_yahoo` depends on the `yfinance` package.
- `load_csv` validates presence of the expected OHLCV columns and raises on missing ones.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

def _to_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a UTC `DatetimeIndex` and sort by time.

    If the DataFrame does not already have a `DatetimeIndex`, this function will
    look for a `Date` column, parse it as timezone-aware UTC, and set it as the
    index. Otherwise, the existing index is localised or converted to UTC. The
    returned frame is sorted ascending by index.

    Args:
        df: Input DataFrame with either a `DatetimeIndex` or a `Date` column.

    Returns:
        DataFrame indexed by UTC `DatetimeIndex`, sorted by time.

    Raises:
        ValueError if neither a `DatetimeIndex` nor a `Date` column is present.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], utc=True)
            df = df.set_index("Date")
        else:
            raise ValueError("Data must have a DatetimeIndex or a 'Date' column")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df.sort_index()

def load_yahoo(ticker: str, start: str, end: str, interval: str = "1d") -> pd.DataFrame:
    """Load OHLCV data from Yahoo Finance via `yfinance`.

    Args:
        ticker: Yahoo ticker symbol (e.g., "SPY").
        start: Start date (YYYY-MM-DD).
        end: End date (YYYY-MM-DD).
        interval: Bar interval accepted by Yahoo (e.g., "1d", "1h").

    Returns:
        DataFrame indexed by UTC datetimes with OHLCV columns.
    """
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False)
    df = df.rename(columns=str.title).reset_index().rename(columns={"Date": "Date"})
    df = _to_utc_index(df)
    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]

def load_csv(path: str) -> pd.DataFrame:
    """Load historical price data from a CSV file and validate columns.

    The CSV must include either a `Date` column (parsable to datetime) or be
    indexed by a `DatetimeIndex` already. It must contain the standard OHLCV
    columns.

    Args:
        path: Filesystem path to the CSV file.

    Returns:
        DataFrame indexed by UTC datetimes with required OHLCV columns.

    Raises:
        ValueError if any required column is missing.
    """
    df = pd.read_csv(path)
    df = _to_utc_index(df)
    cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df[cols]

def add_log_returns(df: pd.DataFrame, price_col: str = "Adj Close") -> pd.DataFrame:
    """Append a `ret` column of log returns computed from `price_col`.

    The log return at time t is defined as `log(price_t) - log(price_{t-1})`.
    The first value will be `NaN` due to differencing.

    Args:
        df: Input price DataFrame.
        price_col: Column name to compute returns from (default: `Adj Close`).

    Returns:
        A copy of `df` with an added `ret` column of log returns.
    """
    df = df.copy()
    df["ret"] = np.log(df[price_col]).diff()
    return df
