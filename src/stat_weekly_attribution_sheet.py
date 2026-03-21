"""Generate a compact weekly attribution sheet for the current strategy stack."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ops"
FORWARD_LEDGER = PROJECT_ROOT / "outputs" / "forward_shadow" / "forward_ledger.csv"
SHOCK_LEDGER = PROJECT_ROOT / "outputs" / "forward_shadow" / "shock_forward_ledger.csv"
CROSS_SHADOW = PROJECT_ROOT / "outputs" / "cross_sectional" / "shadow_validation" / "cross_sectional_shadow_equity.csv"
SHOCK_SHADOW = PROJECT_ROOT / "outputs" / "shock_reversal" / "shadow_validation" / "shock_shadow_equity.csv"
INCIDENT_LOG = PROJECT_ROOT / "outputs" / "forward_shadow" / "incident_log.csv"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cross = pd.read_csv(CROSS_SHADOW, parse_dates=["date"]).sort_values("date")
    shock = pd.read_csv(SHOCK_SHADOW, parse_dates=["date"]).sort_values("date")
    forward = pd.read_csv(FORWARD_LEDGER, parse_dates=["asof_date_utc"]) if FORWARD_LEDGER.exists() else pd.DataFrame()
    shock_ledger = pd.read_csv(SHOCK_LEDGER, parse_dates=["asof_date_utc"]) if SHOCK_LEDGER.exists() else pd.DataFrame()
    incidents = pd.read_csv(INCIDENT_LOG, parse_dates=["timestamp_utc"]) if INCIDENT_LOG.exists() else pd.DataFrame()

    end = cross["date"].max()
    start = end - pd.Timedelta(days=6)
    cross_week = cross.loc[cross["date"] >= start].copy()
    shock_week = shock.loc[shock["date"] >= start].copy()
    forward_week = forward.loc[forward["asof_date_utc"] >= start].copy() if not forward.empty else pd.DataFrame()
    shock_forward_week = shock_ledger.loc[shock_ledger["asof_date_utc"] >= start].copy() if not shock_ledger.empty else pd.DataFrame()
    incident_week = incidents.loc[incidents["timestamp_utc"] >= start.tz_localize("UTC")] if not incidents.empty else pd.DataFrame()

    rows = [
        {"pillar": "btc_v2", "weekly_pnl": float(cross_week["btc_v2_pnl"].sum()), "avg_daily_pnl": float(cross_week["btc_v2_pnl"].mean())},
        {"pillar": "cross_sectional", "weekly_pnl": float(cross_week["cross_pnl"].sum()), "avg_daily_pnl": float(cross_week["cross_pnl"].mean())},
        {"pillar": "combo_25_75", "weekly_pnl": float(cross_week["combo_pnl"].sum()), "avg_daily_pnl": float(cross_week["combo_pnl"].mean())},
        {
            "pillar": "combo_stablecoin_overlay",
            "weekly_pnl": float(cross_week["combo_stablecoin_pnl"].sum()),
            "avg_daily_pnl": float(cross_week["combo_stablecoin_pnl"].mean()),
        },
        {
            "pillar": "combo_stablecoin_delta",
            "weekly_pnl": float((cross_week["combo_stablecoin_pnl"] - cross_week["combo_pnl"]).sum()),
            "avg_daily_pnl": float((cross_week["combo_stablecoin_pnl"] - cross_week["combo_pnl"]).mean()),
        },
        {"pillar": "shock_reversal", "weekly_pnl": float(shock_week["shock_pnl"].sum()), "avg_daily_pnl": float(shock_week["shock_pnl"].mean())},
        {"pillar": "shock25_combo75", "weekly_pnl": float(shock_week["shock_combo_pnl"].sum()), "avg_daily_pnl": float(shock_week["shock_combo_pnl"].mean())},
    ]
    sheet = pd.DataFrame(rows).sort_values("weekly_pnl", ascending=False)

    v2_reason_counts = (
        forward_week["v2_reason"].fillna("n/a").value_counts().head(10).to_dict() if not forward_week.empty and "v2_reason" in forward_week.columns else {}
    )
    shock_reason_counts = (
        shock_forward_week["reason"].fillna("n/a").value_counts().head(10).to_dict() if not shock_forward_week.empty and "reason" in shock_forward_week.columns else {}
    )

    out_csv = OUTPUT_ROOT / "weekly_attribution_sheet.csv"
    out_json = OUTPUT_ROOT / "weekly_attribution_sheet.json"
    out_md = OUTPUT_ROOT / "weekly_attribution_sheet.md"
    sheet.to_csv(out_csv, index=False)
    out_json.write_text(
        json.dumps(
            {
                "window_start": start.strftime("%Y-%m-%d"),
                "window_end": end.strftime("%Y-%m-%d"),
                "rows": rows,
                "v2_reason_counts": v2_reason_counts,
                "shock_reason_counts": shock_reason_counts,
                "incident_count": int(len(incident_week)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Weekly Attribution Sheet",
        "",
        f"- window_start: {start.strftime('%Y-%m-%d')}",
        f"- window_end: {end.strftime('%Y-%m-%d')}",
        f"- incidents: {len(incident_week)}",
        "",
        "## PnL By Pillar",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['pillar']}",
                f"- weekly_pnl: {row['weekly_pnl']:.6f}",
                f"- avg_daily_pnl: {row['avg_daily_pnl']:.6f}",
                "",
            ]
        )
    lines.append("## BTC v2 Reasons")
    if v2_reason_counts:
        lines.extend([f"- {k}: {v}" for k, v in v2_reason_counts.items()])
    else:
        lines.append("- none")
    lines.extend(["", "## Shock Reasons"])
    if shock_reason_counts:
        lines.extend([f"- {k}: {v}" for k, v in shock_reason_counts.items()])
    else:
        lines.append("- none")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
