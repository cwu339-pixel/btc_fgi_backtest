"""Build a compact risk-budget and kill-switch sheet for the current forward stack."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ops"
POLICY_PATH = PROJECT_ROOT / "config" / "risk_budget_policy_v1.json"
LAST_RUN_PATH = PROJECT_ROOT / "outputs" / "forward_shadow" / "last_run.json"
FORWARD_LEDGER = PROJECT_ROOT / "outputs" / "forward_shadow" / "forward_ledger.csv"
SHOCK_LEDGER = PROJECT_ROOT / "outputs" / "forward_shadow" / "shock_forward_ledger.csv"
CROSS_SHADOW = PROJECT_ROOT / "outputs" / "cross_sectional" / "shadow_validation" / "cross_sectional_shadow_equity.csv"
SHOCK_SHADOW = PROJECT_ROOT / "outputs" / "shock_reversal" / "shadow_validation" / "shock_shadow_equity.csv"


def annualized_vol(ret: pd.Series) -> float:
    return float(ret.std(ddof=0) * (365 ** 0.5)) if len(ret.dropna()) >= 20 else float("nan")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    last_run = json.loads(LAST_RUN_PATH.read_text(encoding="utf-8"))
    ledger = pd.read_csv(FORWARD_LEDGER) if FORWARD_LEDGER.exists() else pd.DataFrame()
    shock_ledger = pd.read_csv(SHOCK_LEDGER) if SHOCK_LEDGER.exists() else pd.DataFrame()
    cross_shadow = pd.read_csv(CROSS_SHADOW, parse_dates=["date"]).sort_values("date")
    shock_shadow = pd.read_csv(SHOCK_SHADOW, parse_dates=["date"]).sort_values("date")

    lookback_cut = cross_shadow["date"].max() - pd.Timedelta(days=89)
    cross_90 = cross_shadow.loc[cross_shadow["date"] >= lookback_cut].copy()
    shock_90 = shock_shadow.loc[shock_shadow["date"] >= lookback_cut].copy()

    if not cross_90.empty:
        main_combo_weekly_dd = float(
            ((1.0 + cross_90["combo_pnl"]).cumprod() / (1.0 + cross_90["combo_pnl"]).cumprod().cummax() - 1.0).min()
        )
    else:
        main_combo_weekly_dd = float("nan")
    if not shock_90.empty:
        shock_combo_weekly_dd = float(
            ((1.0 + shock_90["shock_combo_pnl"]).cumprod() / (1.0 + shock_90["shock_combo_pnl"]).cumprod().cummax() - 1.0).min()
        )
    else:
        shock_combo_weekly_dd = float("nan")

    recent_halts = int(pd.to_numeric(ledger.get("halt", 0), errors="coerce").fillna(0).tail(30).sum()) if not ledger.empty else 0
    latest_combo = last_run.get("combo", {})
    latest_shock = last_run.get("shock", {})

    rows = [
        {
            "pillar": "btc_v2",
            "role": policy["pillars"]["btc_v2"]["role"],
            "risk_budget_share": policy["pillars"]["btc_v2"]["risk_budget_share"],
            "max_weight": policy["pillars"]["btc_v2"]["max_weight"],
            "current_weight": float(latest_combo.get("btc_weight", 0.0)) * float(last_run.get("btc", {}).get("v2_target_exposure", 0.0)),
            "ann_vol_90d": annualized_vol(cross_90["btc_v2_pnl"]) if not cross_90.empty else float("nan"),
        },
        {
            "pillar": "cross_sectional",
            "role": policy["pillars"]["cross_sectional"]["role"],
            "risk_budget_share": policy["pillars"]["cross_sectional"]["risk_budget_share"],
            "max_weight": policy["pillars"]["cross_sectional"]["max_weight"],
            "current_weight": float(latest_combo.get("cross_weight", 0.0)),
            "ann_vol_90d": annualized_vol(cross_90["cross_pnl"]) if not cross_90.empty else float("nan"),
        },
        {
            "pillar": "shock_reversal",
            "role": policy["pillars"]["shock_reversal"]["role"],
            "risk_budget_share": policy["pillars"]["shock_reversal"]["risk_budget_share"],
            "max_weight": policy["pillars"]["shock_reversal"]["max_weight"],
            "current_weight": float(latest_shock.get("target_exposure", 0.0)) * 0.25,
            "ann_vol_90d": annualized_vol(shock_90["shock_pnl"]) if not shock_90.empty else float("nan"),
        },
    ]
    sheet = pd.DataFrame(rows)
    sheet["budget_utilization"] = sheet["current_weight"] / sheet["max_weight"].replace(0.0, pd.NA)

    kill_switch = {
        "data_halt_now": bool(last_run.get("halt", False)),
        "recent_halts_30d": recent_halts,
        "kill_on_data_halt": bool(policy["portfolio"]["kill_switch_on_data_halt"]),
        "kill_on_three_halts_30d": bool(policy["portfolio"]["kill_switch_on_three_halts_30d"]),
        "combo_main_90d_drawdown": main_combo_weekly_dd,
        "shock_combo_90d_drawdown": shock_combo_weekly_dd,
        "max_weekly_drawdown_limit": float(policy["portfolio"]["max_weekly_drawdown"]),
    }

    out_csv = OUTPUT_ROOT / "risk_budget_sheet.csv"
    out_json = OUTPUT_ROOT / "risk_budget_sheet.json"
    out_md = OUTPUT_ROOT / "risk_budget_sheet.md"
    sheet.to_csv(out_csv, index=False)
    sheet_records = sheet.to_dict(orient="records")
    out_json.write_text(
        json.dumps({"policy": policy, "kill_switch": kill_switch, "sheet": sheet_records}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Risk Budget & Kill Switch Sheet",
        "",
        f"- policy_name: {policy['policy_name']}",
        f"- latest_asof_date: {last_run.get('asof_date_utc', 'n/a')}",
        f"- data_halt_now: {kill_switch['data_halt_now']}",
        f"- recent_halts_30d: {kill_switch['recent_halts_30d']}",
        f"- combo_main_90d_drawdown: {kill_switch['combo_main_90d_drawdown']:.4f}" if pd.notna(kill_switch["combo_main_90d_drawdown"]) else "- combo_main_90d_drawdown: n/a",
        f"- shock_combo_90d_drawdown: {kill_switch['shock_combo_90d_drawdown']:.4f}" if pd.notna(kill_switch["shock_combo_90d_drawdown"]) else "- shock_combo_90d_drawdown: n/a",
        "",
        "## Pillars",
    ]
    for row in sheet_records:
        ann_vol = row["ann_vol_90d"]
        util = row["budget_utilization"]
        lines.extend(
            [
                f"### {row['pillar']}",
                f"- role: {row['role']}",
                f"- risk_budget_share: {row['risk_budget_share']:.2f}",
                f"- max_weight: {row['max_weight']:.2f}",
                f"- current_weight: {row['current_weight']:.4f}",
                f"- budget_utilization: {util:.4f}" if pd.notna(util) else "- budget_utilization: n/a",
                f"- ann_vol_90d: {ann_vol:.4f}" if pd.notna(ann_vol) else "- ann_vol_90d: n/a",
                "",
            ]
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
