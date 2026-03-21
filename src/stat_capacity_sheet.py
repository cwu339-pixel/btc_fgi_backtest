"""Estimate rough capacity limits for the current strategy stack using spot quote volume proxies."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "ops"
LAST_RUN_PATH = PROJECT_ROOT / "outputs" / "forward_shadow" / "last_run.json"
MARKET_DIR = PROJECT_ROOT / "data" / "market"
PARTICIPATION_BPS = 5.0


def load_market(asset: str) -> pd.DataFrame:
    path = MARKET_DIR / f"binance_spot_daily_{asset}USDT.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values("date")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    last_run = json.loads(LAST_RUN_PATH.read_text(encoding="utf-8"))

    latest_combo_weights = last_run.get("combo", {}).get("target_weights", {})
    shock_assets = last_run.get("shock", {}).get("selected_assets", [])
    assets = sorted(set(["BTC", "ETH", "SOL", "XRP", "BNB"]) | set(latest_combo_weights.keys()) | set(shock_assets))

    rows = []
    for asset in assets:
        try:
            df = load_market(asset)
        except Exception:
            continue
        tail = df.tail(60).copy()
        median_quote = float(pd.to_numeric(tail["quote_volume"], errors="coerce").median())
        advised_notional = median_quote * (PARTICIPATION_BPS / 10000.0)
        rows.append(
            {
                "asset": asset,
                "median_quote_volume_60d": median_quote,
                "participation_bps": PARTICIPATION_BPS,
                "suggested_max_notional": advised_notional,
                "combo_target_weight": float(latest_combo_weights.get(asset, 0.0)),
                "combo_capacity_at_target_weight": advised_notional / max(float(latest_combo_weights.get(asset, 0.0)), 1e-9)
                if float(latest_combo_weights.get(asset, 0.0)) > 0
                else float("nan"),
            }
        )

    sheet = pd.DataFrame(rows).sort_values("suggested_max_notional", ascending=False)
    combo_cap = float(sheet["combo_capacity_at_target_weight"].dropna().min()) if sheet["combo_capacity_at_target_weight"].dropna().any() else float("nan")

    out_csv = OUTPUT_ROOT / "capacity_estimate_sheet.csv"
    out_json = OUTPUT_ROOT / "capacity_estimate_sheet.json"
    out_md = OUTPUT_ROOT / "capacity_estimate_sheet.md"
    sheet.to_csv(out_csv, index=False)
    out_json.write_text(
        json.dumps(
            {
                "participation_bps": PARTICIPATION_BPS,
                "latest_asof_date": last_run.get("asof_date_utc"),
                "approx_combo_max_notional": combo_cap,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Capacity Estimate Sheet",
        "",
        f"- latest_asof_date: {last_run.get('asof_date_utc', 'n/a')}",
        f"- participation_bps: {PARTICIPATION_BPS:.1f}",
        f"- approx_combo_max_notional: {combo_cap:,.2f}" if pd.notna(combo_cap) else "- approx_combo_max_notional: n/a",
        "",
        "## Assets",
    ]
    for row in rows:
        lines.extend(
            [
                f"### {row['asset']}",
                f"- median_quote_volume_60d: {row['median_quote_volume_60d']:,.2f}",
                f"- suggested_max_notional: {row['suggested_max_notional']:,.2f}",
                f"- combo_target_weight: {row['combo_target_weight']:.4f}",
                f"- combo_capacity_at_target_weight: {row['combo_capacity_at_target_weight']:,.2f}" if pd.notna(row["combo_capacity_at_target_weight"]) else "- combo_capacity_at_target_weight: n/a",
                "",
            ]
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
