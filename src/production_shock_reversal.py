"""Generate production-like daily signal logs for the shock reversal shadow strategy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.stat_shock_reversal import build_panel, build_shock_signals, zscore_last


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "shock_reversal" / "production_signals"
BEST_CONFIG = {
    "threshold": 1.25,
    "trigger": "midpoint_reclaim",
    "top_k": 0,
    "hold_days": 7,
    "fee": 0.0010,
    "slip": 0.0010,
}


def build_shock_payloads() -> tuple[pd.DataFrame, dict]:
    opens, highs, lows, closes, volumes = build_panel()
    ret_df = closes.pct_change(fill_method=None).fillna(0.0)
    positions, stress = build_shock_signals(
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
        threshold=BEST_CONFIG["threshold"],
        trigger=BEST_CONFIG["trigger"],
        top_k=BEST_CONFIG["top_k"],
        hold_days=BEST_CONFIG["hold_days"],
    )

    intraday_range = (highs - lows) / closes.replace(0.0, np.nan)
    shock_flag_df = ((stress > BEST_CONFIG["threshold"]) & (ret_df < 0)).astype(int)
    midpoint = (opens.shift(1) + closes.shift(1)) / 2.0
    reclaim_flag_df = (shock_flag_df.shift(1, fill_value=0).astype(bool) & (closes > midpoint)).astype(int)

    # Remaining holding days by asset.
    remaining = pd.DataFrame(0, index=positions.index, columns=positions.columns, dtype=int)
    for lag in range(BEST_CONFIG["hold_days"]):
        rem = BEST_CONFIG["hold_days"] - lag
        remaining = np.maximum(
            remaining,
            reclaim_flag_df.shift(lag, fill_value=0).astype(int) * rem,
        )
    remaining = pd.DataFrame(remaining, index=positions.index, columns=positions.columns)

    turnover = positions.diff().abs().sum(axis=1).fillna(positions.abs().sum(axis=1))
    pnl = (positions.shift(1).fillna(0.0) * ret_df.fillna(0.0)).sum(axis=1) - (
        BEST_CONFIG["fee"] + BEST_CONFIG["slip"]
    ) * turnover
    eq = (1.0 + pnl).cumprod()

    shock_score = stress.max(axis=1, skipna=True)
    shock_flag_any = (shock_flag_df.sum(axis=1) > 0).astype(int)
    reclaim_flag_any = (reclaim_flag_df.sum(axis=1) > 0).astype(int)
    hold_days_remaining = remaining.max(axis=1)

    rows: list[dict[str, object]] = []
    for dt in positions.index:
        active = positions.loc[dt][positions.loc[dt] > 0].sort_values(ascending=False)
        if reclaim_flag_any.loc[dt]:
            reason = "midpoint_reclaim_entry"
        elif hold_days_remaining.loc[dt] > 0 and active.any():
            reason = "reversal_hold"
        elif shock_flag_any.loc[dt]:
            reason = "shock_day_wait_reclaim"
        else:
            reason = "inactive"
        row: dict[str, object] = {
            "date": dt,
            "threshold": BEST_CONFIG["threshold"],
            "trigger": BEST_CONFIG["trigger"],
            "shock_score": float(shock_score.loc[dt]) if pd.notna(shock_score.loc[dt]) else np.nan,
            "shock_flag": int(shock_flag_any.loc[dt]),
            "reclaim_flag": int(reclaim_flag_any.loc[dt]),
            "target_exposure": float(active.sum()),
            "hold_days_remaining": int(hold_days_remaining.loc[dt]),
            "reason": reason,
            "selected_assets": ",".join(active.index.tolist()),
            "portfolio_equity": float(eq.loc[dt]),
            "portfolio_pnl": float(pnl.loc[dt]),
            "portfolio_turnover": float(turnover.loc[dt]),
        }
        for asset in positions.columns:
            row[f"{asset}_weight"] = float(positions.loc[dt, asset])
            row[f"{asset}_remaining"] = int(remaining.loc[dt, asset])
        rows.append(row)

    signal_log = pd.DataFrame(rows)
    latest = signal_log.iloc[-1].copy()
    selected_assets = [asset for asset in str(latest["selected_assets"]).split(",") if asset]
    selected_weights = {asset: float(latest[f"{asset}_weight"]) for asset in selected_assets}
    latest_payload = {
        "date": pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"),
        "threshold": float(latest["threshold"]),
        "trigger": str(latest["trigger"]),
        "shock_score": None if pd.isna(latest["shock_score"]) else float(latest["shock_score"]),
        "shock_flag": int(latest["shock_flag"]),
        "reclaim_flag": int(latest["reclaim_flag"]),
        "target_exposure": float(latest["target_exposure"]),
        "hold_days_remaining": int(latest["hold_days_remaining"]),
        "reason": str(latest["reason"]),
        "selected_assets": selected_assets,
        "selected_weights": selected_weights,
        "portfolio_equity": float(latest["portfolio_equity"]),
        "portfolio_pnl": float(latest["portfolio_pnl"]),
        "portfolio_turnover": float(latest["portfolio_turnover"]),
        "config": BEST_CONFIG,
    }
    return signal_log, latest_payload


def main() -> None:
    signal_log, latest_payload = build_shock_payloads()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_csv = OUTPUT_ROOT / "shock_signal_log.csv"
    latest_json = OUTPUT_ROOT / "shock_reversal_latest.json"
    latest_md = OUTPUT_ROOT / "shock_reversal_latest.md"
    signal_log.to_csv(log_csv, index=False)
    latest_json.write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")

    lines = [
        "# Shock Reversal Production Signal",
        "",
        f"- date: {latest_payload['date']}",
        f"- threshold: {latest_payload['threshold']:.2f}",
        f"- trigger: {latest_payload['trigger']}",
        f"- shock_score: {latest_payload['shock_score'] if latest_payload['shock_score'] is not None else 'n/a'}",
        f"- shock_flag: {latest_payload['shock_flag']}",
        f"- reclaim_flag: {latest_payload['reclaim_flag']}",
        f"- target_exposure: {latest_payload['target_exposure']:.4f}",
        f"- hold_days_remaining: {latest_payload['hold_days_remaining']}",
        f"- reason: {latest_payload['reason']}",
        f"- selected_assets: {', '.join(latest_payload['selected_assets']) or 'none'}",
        f"- portfolio_pnl: {latest_payload['portfolio_pnl']:.6f}",
        f"- portfolio_turnover: {latest_payload['portfolio_turnover']:.4f}",
    ]
    for asset, weight in latest_payload["selected_weights"].items():
        lines.append(f"- {asset}_weight: {weight:.4f}")
    latest_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {log_csv}")
    print(f"Wrote {latest_json}")
    print(f"Wrote {latest_md}")


if __name__ == "__main__":
    main()
