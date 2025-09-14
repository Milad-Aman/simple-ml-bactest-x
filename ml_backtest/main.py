"""
CLI entry point for running a walkforward backtest.

This module:
- Parses a YAML config path from the command line.
- Loads market data from Yahoo Finance or a CSV based on the config.
- Computes log returns and runs a walkforward backtest using the configured strategy.
- Evaluates results with summary metrics, a Monte Carlo permutation p-value, and a runs test.
- Writes artifacts (CSVs, plots, summary.json) to a timestamped folder under `reports/` and prints the
  summary JSON to stdout.
"""

from __future__ import annotations
import argparse, os, yaml, json
import numpy as np
from datetime import datetime
from pathlib import Path
from .a_data_loader import load_yahoo, load_csv, add_log_returns
from .g_walkforward import walkforward_run
from .c_metrics import summarise
from .h_mc_permutation import mc_permutation_pvalue
from .i_runs_test import runs_test
from .j_plots import plot_equity_and_dd, plot_fold_metrics


def parse_args():
    """
    Parse command-line arguments for the backtest runner.

    Returns:
        argparse.Namespace: Contains `config` (str, YAML path) and optional `seed` (int).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--seed", type=int, default=None)
    return ap.parse_args()

def load_data(cfg):
    """
    Load and prepare price data according to the config.

    Expects:
        cfg["data"]["source"]: "yahoo" or "csv".
        If "yahoo": requires keys: "ticker", "start", "end"; optional: "interval" (default "1d").
        If "csv": requires key: "path" (str).

    Returns:
        pandas.DataFrame: Price data with additional log-return columns via `add_log_returns`.

    Raises:
        ValueError: If the data source is unknown.
    """
    src = cfg["data"]["source"]
    if src == "yahoo":
        df = load_yahoo(cfg["data"]["ticker"], cfg["data"]["start"], cfg["data"]["end"], cfg["data"].get("interval","1d"))
    elif src == "csv":
        df = load_csv(cfg["data"]["path"])
    else:
        raise ValueError("Unknown data source")
    return add_log_returns(df)

def main():
    """
    Execute the full walkforward backtest workflow.

    - Parses CLI args and config
    - Seeds RNG (optional override via `--seed`)
    - Loads data and runs walkforward backtest
    - Computes metrics and statistical tests
    - Writes artifacts under `reports/<tag>_YYYYMMDD_HHMMSS/` - the date of backtest run
    - Prints a JSON summary to stdout
    """
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42) if args.seed is None else args.seed
    rng = np.random.default_rng(seed)

    df = load_data(cfg)

    folds_df, oos_ret, oos_pos, oos_eq = walkforward_run(cfg, df, rng)
    summary = summarise(oos_ret, oos_pos)
    pval = mc_permutation_pvalue(oos_ret, n_iter=1000, block=5, seed=seed)
    z_run, p_run = runs_test(oos_ret)

    tag = cfg["output"].get("tag","run")
    outdir = Path("reports") / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    outdir.mkdir(parents=True, exist_ok=True)

    folds_df.to_csv(outdir / "fold_metrics.csv", index=False)
    oos_ret.to_csv(outdir / "oos_returns.csv")
    oos_pos.to_csv(outdir / "oos_positions.csv")
    oos_eq.to_csv(outdir / "oos_equity.csv")

    plot_equity_and_dd(oos_eq, f"Equity (OOS) - {tag}", str(outdir / "equity.png"))
    plot_fold_metrics(folds_df, f"Fold Sharpe (OOS) - {tag}", str(outdir / "fold_sharpe.png"))

    summary_out = {
        "config": cfg,
        "summary": summary,
        "mc_permutation_pvalue": pval,
        "runs_test_z": z_run,
        "runs_test_p": p_run,
        "artifact_dir": str(outdir)
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    print(json.dumps(summary_out, indent=2))

if __name__ == "__main__":
    main()
