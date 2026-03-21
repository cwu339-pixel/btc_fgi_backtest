"""Build a shadow dashboard for the shock reversal provisional third pillar."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.production_shock_reversal import build_shock_payloads
from src.research import perf_slice, perf_stats
from src.stat_system_backtest import calmar_ratio, tail_ratio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "shock_reversal" / "shadow_validation"
COMBO_SHADOW_PATH = PROJECT_ROOT / "outputs" / "cross_sectional" / "shadow_validation" / "cross_sectional_shadow_equity.csv"
POLICY_PATH = PROJECT_ROOT / "config" / "shock_shadow_combo_policy_v1.json"
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
    signal_log, latest_payload = build_shock_payloads()
    combo_df = pd.read_csv(COMBO_SHADOW_PATH, parse_dates=["date"]).sort_values("date")
    combo_idx = pd.to_datetime(combo_df["date"])
    combo_pnl = pd.Series(combo_df["combo_pnl"].to_numpy(), index=combo_idx)
    combo_eq = pd.Series(combo_df["combo_equity"].to_numpy(), index=combo_idx)

    shock_idx = pd.to_datetime(signal_log["date"])
    shock_eq = pd.Series(signal_log["portfolio_equity"].to_numpy(), index=shock_idx)
    shock_pnl = pd.Series(signal_log["portfolio_pnl"].to_numpy(), index=shock_idx)

    aligned = pd.concat(
        [
            shock_eq.rename("shock_eq"),
            shock_pnl.rename("shock_pnl"),
            combo_eq.rename("combo_eq"),
            combo_pnl.rename("combo_pnl"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    shock_w = float(policy["shock_weight"])
    combo_w = float(policy["combo_weight"])
    combo_shadow_pnl = shock_w * aligned["shock_pnl"] + combo_w * aligned["combo_pnl"]
    combo_shadow_eq = (1.0 + combo_shadow_pnl).cumprod()

    trigger_series = signal_log["reclaim_flag"].astype(int)
    signal_log["month"] = pd.to_datetime(signal_log["date"]).dt.to_period("M").astype(str)
    monthly_triggers = signal_log.groupby("month")["reclaim_flag"].sum()
    recent_90 = int(trigger_series.tail(90).sum())
    recent_365 = int(trigger_series.tail(365).sum())
    latest_trigger_rows = signal_log.loc[signal_log["reclaim_flag"] == 1]
    latest_trigger_date = (
        pd.to_datetime(latest_trigger_rows.iloc[-1]["date"]).strftime("%Y-%m-%d") if not latest_trigger_rows.empty else "n/a"
    )

    rolling_corr = aligned["shock_pnl"].rolling(60, min_periods=20).corr(aligned["combo_pnl"])
    avg_turnover = float(signal_log["portfolio_turnover"].mean())

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    equity_csv = OUTPUT_ROOT / "shock_shadow_equity.csv"
    summary_json = OUTPUT_ROOT / "shock_shadow_dashboard.json"
    dashboard_md = OUTPUT_ROOT / "shock_shadow_dashboard.md"

    pd.DataFrame(
        {
            "date": aligned.index,
            "shock_equity": aligned["shock_eq"].to_numpy(),
            "combo_equity": aligned["combo_eq"].to_numpy(),
            "shock_combo_equity": combo_shadow_eq.to_numpy(),
            "shock_pnl": aligned["shock_pnl"].to_numpy(),
            "combo_pnl": aligned["combo_pnl"].to_numpy(),
            "shock_combo_pnl": combo_shadow_pnl.to_numpy(),
            "rolling60_corr_vs_combo": rolling_corr.to_numpy(),
        }
    ).to_csv(equity_csv, index=False)

    payload = {
        "latest_shock_signal": latest_payload,
        "shock_stats": summarize(aligned["shock_eq"], aligned["shock_pnl"]),
        "combo_stats": summarize(aligned["combo_eq"], aligned["combo_pnl"]),
        "shock_combo_stats": summarize(combo_shadow_eq, combo_shadow_pnl),
        "latest_trigger_date": latest_trigger_date,
        "recent_90d_triggers": recent_90,
        "recent_365d_triggers": recent_365,
        "avg_turnover": avg_turnover,
        "combo_policy": policy,
        "rolling60_corr_last": float(rolling_corr.dropna().iloc[-1]) if rolling_corr.dropna().any() else None,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Shock Reversal Shadow Dashboard",
        "",
        "## Latest Signal",
        f"- date: {latest_payload['date']}",
        f"- target_exposure: {latest_payload['target_exposure']:.4f}",
        f"- hold_days_remaining: {latest_payload['hold_days_remaining']}",
        f"- reason: {latest_payload['reason']}",
        f"- selected_assets: {', '.join(latest_payload['selected_assets']) or 'none'}",
        "",
        "## Trigger Activity",
        f"- latest_trigger_date: {latest_trigger_date}",
        f"- triggers_last_90d: {recent_90}",
        f"- triggers_last_365d: {recent_365}",
        f"- avg_monthly_triggers: {monthly_triggers.mean():.2f}",
        "",
        "## Test Stats",
        f"- shock test_sharpe: {payload['shock_stats']['test_sharpe']:.4f}",
        f"- shock test_calmar: {payload['shock_stats']['test_calmar']:.4f}",
        f"- combo test_sharpe: {payload['combo_stats']['test_sharpe']:.4f}",
        f"- shock_combo test_sharpe: {payload['shock_combo_stats']['test_sharpe']:.4f}",
        f"- shock_combo test_calmar: {payload['shock_combo_stats']['test_calmar']:.4f}",
        "",
        "## Combo Shadow View",
        f"- combo_policy: {policy['policy_name']}",
        f"- shock_weight: {shock_w:.2f}",
        f"- combo_weight: {combo_w:.2f}",
        f"- rolling60_corr_vs_combo_last: {payload['rolling60_corr_last'] if payload['rolling60_corr_last'] is not None else 'n/a'}",
        f"- avg_turnover: {avg_turnover:.4f}",
    ]
    dashboard_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {dashboard_md}")


if __name__ == "__main__":
    main()
