"""Generate daily production signal logs for the cross-sectional second pillar."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.stat_cross_sectional import build_panel, build_positions, run_portfolio_backtest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cross_sectional" / "production_signals"
BEST_CONFIG = {
    "universe_assets": ("BTC", "ETH", "SOL", "XRP", "BNB"),
    "lookback": 21,
    "top_k": 2,
    "rebalance_days": 1,
    "ranking_method": "risk_adjusted",
    "weighting_method": "equal",
    "fee": 0.0010,
    "slip": 0.0010,
}
RESEARCH_CANDIDATE_V1 = {
    "universe_assets": ("BTC", "ETH", "SOL", "XRP", "BNB"),
    "lookback": 21,
    "top_k": 2,
    "rebalance_days": 2,
    "ranking_method": "risk_adjusted",
    "weighting_method": "equal",
    "fee": 0.0010,
    "slip": 0.0010,
    "hold_buffer": 0,
}


def build_cross_sectional_payloads(config: dict | None = None, label: str = "cross_1d") -> tuple[pd.DataFrame, dict]:
    cfg = dict(BEST_CONFIG if config is None else config)
    close_df = build_panel(tuple(cfg["universe_assets"]))[list(cfg["universe_assets"])]
    positions = build_positions(
        close_df=close_df,
        lookback=cfg["lookback"],
        top_k=cfg["top_k"],
        rebalance_days=cfg["rebalance_days"],
        ranking_method=cfg["ranking_method"],
        weighting_method=cfg["weighting_method"],
    )
    eq, turnover, pnl = run_portfolio_backtest(
        positions,
        close_df,
        fee=cfg["fee"],
        slip=cfg["slip"],
    )

    latest_date = positions.index[-1]
    latest_weights = positions.iloc[-1]
    active = latest_weights[latest_weights > 0].sort_values(ascending=False)
    selected_assets = ",".join(active.index.tolist())
    selected_weights = {asset: float(weight) for asset, weight in active.items()}

    signal_log = pd.DataFrame(
        {
            "date": positions.index,
            "selected_assets": positions.apply(
                lambda row: ",".join(row[row > 0].index.tolist()),
                axis=1,
            ),
            "portfolio_equity": eq.to_numpy(),
            "portfolio_pnl": pnl.to_numpy(),
            "portfolio_turnover": turnover.to_numpy(),
        }
    )
    for asset in positions.columns:
        signal_log[f"{asset}_weight"] = positions[asset].to_numpy()

    latest_payload = {
        "signal_version": label,
        "date": latest_date.strftime("%Y-%m-%d"),
        "universe_assets": list(cfg["universe_assets"]),
        "selected_assets": active.index.tolist(),
        "selected_weights": selected_weights,
        "portfolio_equity": float(eq.iloc[-1]),
        "portfolio_pnl": float(pnl.iloc[-1]),
        "portfolio_turnover": float(turnover.iloc[-1]),
        "config": cfg,
        "close_snapshot": {
            asset: float(close_df.loc[latest_date, asset]) for asset in close_df.columns
        },
    }
    return signal_log, latest_payload


def main() -> None:
    signal_log, latest_payload = build_cross_sectional_payloads()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_csv = OUTPUT_ROOT / "cross_sectional_signal_log.csv"
    latest_json = OUTPUT_ROOT / "cross_sectional_latest.json"
    latest_md = OUTPUT_ROOT / "cross_sectional_latest.md"

    signal_log.to_csv(log_csv, index=False)
    latest_json.write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")

    lines = [
        "# Cross-Sectional Production Signal",
        "",
        f"- date: {latest_payload['date']}",
        f"- universe_assets: {', '.join(latest_payload['universe_assets'])}",
        f"- selected_assets: {', '.join(latest_payload['selected_assets']) or 'none'}",
        f"- portfolio_equity: {latest_payload['portfolio_equity']:.4f}",
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
