"""Build a compact shadow-validation dashboard for BTC 1d v1 vs v2."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.stat_system_backtest import calmar_ratio, run_position_backtest, tail_ratio
from src.research import perf_slice, perf_stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIGNAL_ROOT = PROJECT_ROOT / "outputs" / "production_signals"
SHADOW_ROOT = PROJECT_ROOT / "outputs" / "system_backtest" / "shadow_validation"
FEE = 0.0004
TRAIN_END = "2023-12-31"


def summarize_shadow(eq: pd.Series, pnl: pd.Series, turnover: pd.Series, position: pd.Series) -> dict[str, float]:
    full = perf_stats(eq)
    test_start = (pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    test = perf_slice(eq, test_start, eq.index.max().strftime("%Y-%m-%d"))
    test_mask = eq.index > pd.Timestamp(TRAIN_END)
    return {
        "total_return": float(full["total_return"]),
        "cagr": float(full["ann_return"]),
        "sharpe": float(full["sharpe"]),
        "max_drawdown": float(full["max_drawdown"]),
        "calmar": float(calmar_ratio(full["ann_return"], full["max_drawdown"])),
        "tail_ratio": float(tail_ratio(pnl)),
        "avg_turnover": float(turnover.mean()),
        "avg_abs_exposure": float(position.abs().mean()),
        "test_sharpe": float(test["sharpe"]),
        "test_calmar": float(calmar_ratio(test["ann_return"], test["max_drawdown"])),
        "test_tail_ratio": float(tail_ratio(pnl.loc[test_mask])),
    }


def main() -> None:
    signal_log = pd.read_csv(SIGNAL_ROOT / "btc_1d_signal_log.csv", parse_dates=["date"]).sort_values("date")
    latest_payload = json.loads((SIGNAL_ROOT / "btc_1d_signal_latest.json").read_text(encoding="utf-8"))
    divergence_report = (SHADOW_ROOT / "btc_1d_divergence_report.md").read_text(encoding="utf-8")

    idx = pd.to_datetime(signal_log["date"])
    close = pd.Series(signal_log["close"].to_numpy(), index=idx)
    ret = close.pct_change().fillna(0.0)
    v1_pos = pd.Series(signal_log["v1_target_exposure"].to_numpy(), index=idx)
    v2_pos = pd.Series(signal_log["v2_target_exposure"].to_numpy(), index=idx)

    v1_eq, v1_turnover, v1_pnl = run_position_backtest(v1_pos, ret, FEE)
    v2_eq, v2_turnover, v2_pnl = run_position_backtest(v2_pos, ret, FEE)

    v1_stats = summarize_shadow(v1_eq, v1_pnl, v1_turnover, v1_pos)
    v2_stats = summarize_shadow(v2_eq, v2_pnl, v2_turnover, v2_pos)

    SHADOW_ROOT.mkdir(parents=True, exist_ok=True)
    equity_csv = SHADOW_ROOT / "btc_1d_shadow_equity.csv"
    summary_json = SHADOW_ROOT / "btc_1d_shadow_dashboard.json"
    dashboard_md = SHADOW_ROOT / "btc_1d_shadow_dashboard.md"

    pd.DataFrame(
        {
            "date": idx,
            "v1_equity": v1_eq.to_numpy(),
            "v2_equity": v2_eq.to_numpy(),
            "v1_pnl": v1_pnl.to_numpy(),
            "v2_pnl": v2_pnl.to_numpy(),
            "v1_target_exposure": v1_pos.to_numpy(),
            "v2_target_exposure": v2_pos.to_numpy(),
        }
    ).to_csv(equity_csv, index=False)

    payload = {
        "latest": latest_payload,
        "v1_stats": v1_stats,
        "v2_stats": v2_stats,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# BTC 1d Shadow Dashboard",
        "",
        "## Latest Signal",
        f"- date: {latest_payload['date']}",
        f"- regime: {latest_payload['regime']}",
        f"- v1 exposure: {latest_payload['v1_target_exposure']:.4f}",
        f"- v2 exposure: {latest_payload['v2_target_exposure']:.4f}",
        f"- exposure delta: {latest_payload['target_exposure_delta']:.4f}",
        f"- reason: {latest_payload['v2_reason']}",
        "",
        "## Shadow Equity Stats",
        f"- v1 sharpe: {v1_stats['sharpe']:.4f}",
        f"- v2 sharpe: {v2_stats['sharpe']:.4f}",
        f"- v1 calmar: {v1_stats['calmar']:.4f}",
        f"- v2 calmar: {v2_stats['calmar']:.4f}",
        f"- v1 max_drawdown: {v1_stats['max_drawdown']:.4f}",
        f"- v2 max_drawdown: {v2_stats['max_drawdown']:.4f}",
        f"- v1 avg_turnover: {v1_stats['avg_turnover']:.4f}",
        f"- v2 avg_turnover: {v2_stats['avg_turnover']:.4f}",
        "",
        "## Test-Window Stats",
        f"- v1 test_sharpe: {v1_stats['test_sharpe']:.4f}",
        f"- v2 test_sharpe: {v2_stats['test_sharpe']:.4f}",
        f"- v1 test_calmar: {v1_stats['test_calmar']:.4f}",
        f"- v2 test_calmar: {v2_stats['test_calmar']:.4f}",
        "",
        "## Divergence Snapshot",
    ]
    for line in divergence_report.splitlines()[2:10]:
        if line:
            lines.append(line)

    dashboard_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {dashboard_md}")


if __name__ == "__main__":
    main()
