import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from IPython.display import display, Markdown, HTML

def _fmt(label, val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "—"
    if label in {"Ann. return (arith)", "Hit rate", "Exposure", "Max drawdown", "VaR 5%", "ES 5%"}:
        digits = 1 if label in {"Hit rate", "Exposure", "Max drawdown"} else 2
        return f"{val:.{digits}%}"
    if label in {"Sharpe", "Sortino", "Calmar"}:
        return f"{val:.2f}"
    if label == "Profit factor":
        return f"{val:.2f}"
    if label == "Turnover":
        return f"{val:.1f}"
    return str(val)

def _targets_from_cfg(cfg: dict) -> dict:
    """Build target strings; prefer YAML constraints when available."""
    constraints = (
        cfg.get("strategy", {})
          .get("params", {})
          .get("policy", {})
          .get("constraints", {})
        if isinstance(cfg, dict) else {}
    )

    targets = {}

    # Ann return: prefer log min from YAML, convert to arithmetic
    if "ann_return_log_min" in constraints:
        ann_min = np.expm1(constraints["ann_return_log_min"])
        targets["Ann. return (arith)"] = f"≥ {ann_min:.0%}"
    else:
        targets["Ann. return (arith)"] = "≥ 8%"

    # Exposure range
    if "exposure_range" in constraints and isinstance(constraints["exposure_range"], (list, tuple)) and len(constraints["exposure_range"]) == 2:
        lo, hi = constraints["exposure_range"]
        targets["Exposure"] = f"{lo:.0%}–{hi:.0%}"
    else:
        targets["Exposure"] = "20%–60%"

    # Turnover max
    if "turnover_max" in constraints:
        targets["Turnover"] = f"≤ {constraints['turnover_max']:.0f}"
    else:
        targets["Turnover"] = "≤ 100"

    # Optional YAML mins
    if "sharpe_min" in constraints:
        targets["Sharpe"] = f"≥ {constraints['sharpe_min']:.2f}"
    else:
        targets["Sharpe"] = "≥ 0.90"

    if "calmar_min" in constraints:
        targets["Calmar"] = f"≥ {constraints['calmar_min']:.2f}"
    else:
        targets["Calmar"] = "≥ 1.00"

    # Defaults for others
    targets.setdefault("Sortino", "≥ 1.20")
    targets.setdefault("Profit factor", "≥ 1.30")
    targets.setdefault("Hit rate", "≥ 53%")
    targets.setdefault("Max drawdown", "≥ −25%")  # (not more negative than −25%)
    targets.setdefault("VaR 5%", "≥ −3%")
    targets.setdefault("ES 5%", "≥ −5%")

    # Statistical tests
    targets_tests = {
        "MC permutation p-value": "≤ 0.01",
        "Runs test z": "∣z∣ ≤ 2.0",
        "Runs test p": "≥ 0.05",
    }

    return targets, targets_tests

def show_backtest_report(summary_out, *, title=None):
    cfg = summary_out.get("config", {})
    s   = dict(summary_out.get("summary", {}))

    # derive arithmetic annual return from log return
    ann_arith = np.expm1(s.get("ann_return_log", np.nan))

    # targets (from YAML where possible)
    targets, targets_tests = _targets_from_cfg(cfg)

    rows = [
        ("Ann. return (arith)", ann_arith),
        ("Sharpe",              s.get("sharpe")),
        ("Sortino",             s.get("sortino")),
        ("Max drawdown",        s.get("max_dd")),
        ("Calmar",              s.get("calmar")),
        ("Profit factor",       s.get("profit_factor")),
        ("Hit rate",            s.get("hit_rate")),
        ("Exposure",            s.get("exposure")),
        ("Turnover",            s.get("turnover")),
        ("VaR 5%",              s.get("var_5")),
        ("ES 5%",               s.get("es_5")),
    ]

    metrics_df = pd.DataFrame({
        "Metric": [k for k,_ in rows],
        "Value":  [_fmt(k, v) for k,v in rows],
        "Target": [targets.get(k, "") for k,_ in rows],
    })

    tests_df = pd.DataFrame({
        "Test":  ["MC permutation p-value", "Runs test z", "Runs test p"],
        "Value": [
            summary_out.get("mc_permutation_pvalue"),
            summary_out.get("runs_test_z"),
            summary_out.get("runs_test_p"),
        ],
        "Target": [
            targets_tests["MC permutation p-value"],
            targets_tests["Runs test z"],
            targets_tests["Runs test p"],
        ],
    })

    # format tests 'Value'
    def _fmt_test(name, v):
        if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
        if "p-value" in name or name.endswith(" p"): return f"{v:.4f}"
        if name.endswith(" z"): return f"{v:.2f}"
        return str(v)
    tests_df["Value"] = [_fmt_test(n, v) for n, v in zip(tests_df["Test"], tests_df["Value"])]

    tag = (cfg.get("output") or {}).get("tag", "run") if isinstance(cfg, dict) else "run"
    title = title or f"Backtest Summary — {tag}"

    # --- Render ---
    display(Markdown(f"# {title}"))
    display(Markdown("## Metrics"))
    display(metrics_df.style.hide(axis="index"))

    display(Markdown("## Statistical Tests"))
    display(tests_df.style.hide(axis="index"))

    # Collapsible config (unchanged)
    display(Markdown("## Config"))
    cfg_yaml = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    display(HTML(
        f"""<details style="margin:8px 0;"><summary style="cursor:pointer;">Show YAML</summary>
<pre style="overflow:auto; background:#0f172a; color:#e2e8f0; padding:12px; border-radius:8px;">{cfg_yaml}</pre>
</details>"""
    ))


# PLOTS

def plot_equity_and_dd(oos_df: pd.DataFrame, title: str):
    """Plot and save the OOS equity curve to `path`."""
    eq = oos_df["equity"]
    eq.plot()
    plt.title(title)
    plt.ylabel("Equity (rebased)")
    plt.tight_layout()

def plot_fold_metrics(df_metrics: pd.DataFrame, title: str):
    """Bar chart of per-fold Sharpe values, saved to `path`."""
    df_metrics["sharpe"].plot(kind="bar")
    plt.title(title)
    plt.ylabel("Sharpe (OOS)")
    plt.tight_layout()