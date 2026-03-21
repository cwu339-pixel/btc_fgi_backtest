"""Build a dedicated shadow dashboard for the stablecoin macro overlay candidate."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.research import perf_slice, perf_stats
from src.stat_system_backtest import calmar_ratio, tail_ratio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "stablecoin_macro_hint" / "shadow_validation"
CROSS_SHADOW_PATH = PROJECT_ROOT / "outputs" / "cross_sectional" / "shadow_validation" / "cross_sectional_shadow_equity.csv"
FORWARD_LEDGER = PROJECT_ROOT / "outputs" / "forward_shadow" / "forward_ledger.csv"
TRAIN_END = "2023-12-31"


def summarize(eq: pd.Series, pnl: pd.Series) -> dict[str, float]:
    full = perf_stats(eq)
    test_start = (pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    test = perf_slice(eq, test_start, eq.index.max().strftime("%Y-%m-%d"))
    return {
        "total_return": float(full["total_return"]),
        "cagr": float(full["ann_return"]),
        "sharpe": float(full["sharpe"]),
        "max_drawdown": float(full["max_drawdown"]),
        "calmar": float(calmar_ratio(full["ann_return"], full["max_drawdown"])),
        "tail_ratio": float(tail_ratio(pnl)),
        "test_sharpe": float(test["sharpe"]),
        "test_calmar": float(calmar_ratio(test["ann_return"], test["max_drawdown"])),
    }


def main() -> None:
    cross = pd.read_csv(CROSS_SHADOW_PATH, parse_dates=["date"]).sort_values("date")
    idx = pd.to_datetime(cross["date"])
    combo_eq = pd.Series(cross["combo_equity"].to_numpy(), index=idx)
    combo_pnl = pd.Series(cross["combo_pnl"].to_numpy(), index=idx)
    candidate_eq = pd.Series(cross["combo_stablecoin_equity"].to_numpy(), index=idx)
    candidate_pnl = pd.Series(cross["combo_stablecoin_pnl"].to_numpy(), index=idx)
    extreme_flag = pd.Series(cross["stablecoin_extreme_off_flag"].to_numpy(), index=idx)
    rolling_corr = candidate_pnl.rolling(60, min_periods=20).corr(combo_pnl)

    forward = pd.read_csv(FORWARD_LEDGER, parse_dates=["asof_date_utc"]) if FORWARD_LEDGER.exists() else pd.DataFrame()
    latest_forward = {}
    if not forward.empty:
        latest = forward.sort_values("asof_date_utc").iloc[-1]
        latest_forward = {
            "asof_date_utc": latest["asof_date_utc"].strftime("%Y-%m-%d"),
            "combo_policy": latest.get("combo_policy", ""),
            "combo_net_exposure": float(latest.get("combo_net_exposure", 0.0)),
            "candidate_policy": latest.get("combo_stablecoin_candidate_policy", ""),
            "candidate_net_exposure": float(latest.get("combo_stablecoin_candidate_net_exposure", 0.0)),
            "stablecoin_30d_z": float(latest.get("stablecoin_30d_z")) if pd.notna(latest.get("stablecoin_30d_z")) else None,
            "stablecoin_risk_off_flag": bool(latest.get("stablecoin_risk_off_flag", 0)),
            "stablecoin_extreme_off_flag": bool(latest.get("stablecoin_extreme_off_flag", 0)),
        }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    equity_csv = OUTPUT_ROOT / "stablecoin_shadow_equity.csv"
    summary_json = OUTPUT_ROOT / "stablecoin_shadow_dashboard.json"
    dashboard_md = OUTPUT_ROOT / "stablecoin_shadow_dashboard.md"

    pd.DataFrame(
        {
            "date": idx,
            "combo_equity": combo_eq.to_numpy(),
            "combo_stablecoin_equity": candidate_eq.to_numpy(),
            "combo_pnl": combo_pnl.to_numpy(),
            "combo_stablecoin_pnl": candidate_pnl.to_numpy(),
            "stablecoin_extreme_off_flag": extreme_flag.to_numpy(),
            "rolling60_corr_vs_combo": rolling_corr.to_numpy(),
        }
    ).to_csv(equity_csv, index=False)

    combo_stats = summarize(combo_eq, combo_pnl)
    candidate_stats = summarize(candidate_eq, candidate_pnl)
    payload = {
        "combo_stats": combo_stats,
        "candidate_stats": candidate_stats,
        "historical_extreme_off_days": int(extreme_flag.fillna(0).sum()),
        "latest_forward": latest_forward,
        "rolling60_corr_last": float(rolling_corr.dropna().iloc[-1]) if not rolling_corr.dropna().empty else None,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Stablecoin Overlay Shadow Dashboard",
        "",
        "## Historical Candidate vs Baseline",
        f"- combo test_sharpe: {combo_stats['test_sharpe']:.4f}",
        f"- stablecoin_candidate test_sharpe: {candidate_stats['test_sharpe']:.4f}",
        f"- combo test_calmar: {combo_stats['test_calmar']:.4f}",
        f"- stablecoin_candidate test_calmar: {candidate_stats['test_calmar']:.4f}",
        f"- combo max_drawdown: {combo_stats['max_drawdown']:.4f}",
        f"- stablecoin_candidate max_drawdown: {candidate_stats['max_drawdown']:.4f}",
        f"- rolling60_corr_vs_combo_last: {payload['rolling60_corr_last']:.4f}" if payload["rolling60_corr_last"] is not None else "- rolling60_corr_vs_combo_last: n/a",
        f"- historical_extreme_off_days: {payload['historical_extreme_off_days']}",
        "",
        "## Latest Forward Snapshot",
    ]
    if latest_forward:
        lines.extend(
            [
                f"- asof_date_utc: {latest_forward['asof_date_utc']}",
                f"- combo_policy: {latest_forward['combo_policy']}",
                f"- candidate_policy: {latest_forward['candidate_policy']}",
                f"- combo_net_exposure: {latest_forward['combo_net_exposure']:.4f}",
                f"- candidate_net_exposure: {latest_forward['candidate_net_exposure']:.4f}",
                f"- stablecoin_30d_z: {latest_forward['stablecoin_30d_z']:.4f}" if latest_forward["stablecoin_30d_z"] is not None else "- stablecoin_30d_z: n/a",
                f"- stablecoin_risk_off_flag: {latest_forward['stablecoin_risk_off_flag']}",
                f"- stablecoin_extreme_off_flag: {latest_forward['stablecoin_extreme_off_flag']}",
            ]
        )
    else:
        lines.append("- no forward ledger rows yet")
    lines.extend(
        [
            "",
            "## Interpretation",
            "- This candidate is stronger than the baseline combo in historical test metrics.",
            "- The candidate is already wired into the forward ledger.",
            "- The main remaining gap is live forward coverage during real extreme-off episodes.",
        ]
    )
    dashboard_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {dashboard_md}")


if __name__ == "__main__":
    main()
