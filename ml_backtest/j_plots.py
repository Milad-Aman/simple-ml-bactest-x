"""
Plotting helpers for backtest artifacts.
"""

import matplotlib.pyplot as plt
import pandas as pd

def plot_equity_and_dd(oos_df: pd.DataFrame, title: str, path: str):
    """Plot and save the OOS equity curve to `path`."""
    eq = oos_df["equity"]
    fig = plt.figure(figsize=(10,6))
    eq.plot()
    plt.title(title)
    plt.ylabel("Equity (rebased)")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_fold_metrics(df_metrics: pd.DataFrame, title: str, path: str):
    """Bar chart of per-fold Sharpe values, saved to `path`."""
    fig = plt.figure(figsize=(9,4))
    df_metrics["sharpe"].plot(kind="bar")
    plt.title(title)
    plt.ylabel("Sharpe (OOS)")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
