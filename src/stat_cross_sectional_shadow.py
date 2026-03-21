"""Build a compact shadow dashboard for the cross-sectional second pillar and combo portfolio."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.production_cross_sectional import RESEARCH_CANDIDATE_V1, build_cross_sectional_payloads
from src.research import perf_slice, perf_stats
from src.stat_stablecoin_macro_hint import load_stablecoin_macro_filter
from src.stat_system_backtest import calmar_ratio, tail_ratio
from src.stat_vix_hint import load_vix_filter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cross_sectional" / "shadow_validation"
BTC_SHADOW_PATH = PROJECT_ROOT / "outputs" / "system_backtest" / "shadow_validation" / "btc_1d_shadow_equity.csv"
DSR_DIAG_PATH = PROJECT_ROOT / "outputs" / "statistical_validation" / "dsr_pbo" / "strategy_diagnostics.csv"
PBO_PATH = PROJECT_ROOT / "outputs" / "statistical_validation" / "dsr_pbo" / "pbo_summary.csv"
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
    signal_log, latest_payload = build_cross_sectional_payloads(label="cross_1d")
    signal_log_2d, latest_payload_2d = build_cross_sectional_payloads(
        config=RESEARCH_CANDIDATE_V1,
        label="cross_2d_candidate",
    )
    idx = pd.to_datetime(signal_log["date"])
    idx_2d = pd.to_datetime(signal_log_2d["date"])
    cross_eq = pd.Series(signal_log["portfolio_equity"].to_numpy(), index=idx)
    cross_pnl = pd.Series(signal_log["portfolio_pnl"].to_numpy(), index=idx)
    cross_2d_eq = pd.Series(signal_log_2d["portfolio_equity"].to_numpy(), index=idx_2d)
    cross_2d_pnl = pd.Series(signal_log_2d["portfolio_pnl"].to_numpy(), index=idx_2d)

    btc_df = pd.read_csv(BTC_SHADOW_PATH, parse_dates=["date"]).sort_values("date")
    btc_idx = pd.to_datetime(btc_df["date"])
    btc_eq = pd.Series(btc_df["v2_equity"].to_numpy(), index=btc_idx)
    btc_pnl = pd.Series(btc_df["v2_pnl"].to_numpy(), index=btc_idx)

    aligned = pd.concat(
        [
            btc_eq.rename("btc_eq"),
            btc_pnl.rename("btc_pnl"),
            cross_eq.rename("cross_eq"),
            cross_pnl.rename("cross_pnl"),
            cross_2d_eq.rename("cross_2d_eq"),
            cross_2d_pnl.rename("cross_2d_pnl"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    combo_ret = 0.25 * aligned["btc_pnl"] + 0.75 * aligned["cross_pnl"]
    combo_eq = (1.0 + combo_ret).cumprod()
    combo_2d_ret = 0.25 * aligned["btc_pnl"] + 0.75 * aligned["cross_2d_pnl"]
    combo_2d_eq = (1.0 + combo_2d_ret).cumprod()

    stablecoin = load_stablecoin_macro_filter().copy()
    stablecoin["date"] = pd.to_datetime(stablecoin["date"])
    stablecoin = stablecoin.set_index("date").reindex(aligned.index).ffill()
    stablecoin_extreme_off_lag1 = stablecoin["stablecoin_extreme_off_flag"].shift(1).fillna(0.0)
    combo_stablecoin_ret = combo_ret * (1.0 - 1.0 * stablecoin_extreme_off_lag1)
    combo_stablecoin_eq = (1.0 + combo_stablecoin_ret).cumprod()

    vix = load_vix_filter().copy()
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix.set_index("date").reindex(aligned.index).ffill()
    vix_elevated_lag1 = vix["elevated_vix_flag"].shift(1).fillna(0.0)
    combo_vix_ret = combo_ret * (1.0 - 0.5 * vix_elevated_lag1)
    combo_vix_eq = (1.0 + combo_vix_ret).cumprod()

    btc_stats = summarize(aligned["btc_eq"], aligned["btc_pnl"])
    cross_stats = summarize(aligned["cross_eq"], aligned["cross_pnl"])
    cross_2d_stats = summarize(aligned["cross_2d_eq"], aligned["cross_2d_pnl"])
    combo_stats = summarize(combo_eq, combo_ret)
    combo_2d_stats = summarize(combo_2d_eq, combo_2d_ret)
    combo_stablecoin_stats = summarize(combo_stablecoin_eq, combo_stablecoin_ret)
    combo_vix_stats = summarize(combo_vix_eq, combo_vix_ret)
    dsr_diag = pd.read_csv(DSR_DIAG_PATH) if DSR_DIAG_PATH.exists() else pd.DataFrame()
    pbo_summary = pd.read_csv(PBO_PATH).iloc[0].to_dict() if PBO_PATH.exists() else {}

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    equity_csv = OUTPUT_ROOT / "cross_sectional_shadow_equity.csv"
    summary_json = OUTPUT_ROOT / "cross_sectional_shadow_dashboard.json"
    dashboard_md = OUTPUT_ROOT / "cross_sectional_shadow_dashboard.md"

    pd.DataFrame(
        {
            "date": aligned.index,
            "btc_v2_equity": aligned["btc_eq"].to_numpy(),
            "cross_equity": aligned["cross_eq"].to_numpy(),
            "cross_2d_equity": aligned["cross_2d_eq"].to_numpy(),
            "combo_equity": combo_eq.to_numpy(),
            "combo_2d_equity": combo_2d_eq.to_numpy(),
            "combo_stablecoin_equity": combo_stablecoin_eq.to_numpy(),
            "combo_vix_equity": combo_vix_eq.to_numpy(),
            "btc_v2_pnl": aligned["btc_pnl"].to_numpy(),
            "cross_pnl": aligned["cross_pnl"].to_numpy(),
            "cross_2d_pnl": aligned["cross_2d_pnl"].to_numpy(),
            "combo_pnl": combo_ret.to_numpy(),
            "combo_2d_pnl": combo_2d_ret.to_numpy(),
            "combo_stablecoin_pnl": combo_stablecoin_ret.to_numpy(),
            "combo_vix_pnl": combo_vix_ret.to_numpy(),
            "stablecoin_extreme_off_flag": stablecoin["stablecoin_extreme_off_flag"].to_numpy(),
            "vix_close": vix["vix_close"].to_numpy(),
            "vix_elevated_flag": vix["elevated_vix_flag"].to_numpy(),
            "vix_extreme_flag": vix["extreme_vix_flag"].to_numpy(),
        }
    ).to_csv(equity_csv, index=False)

    payload = {
        "latest_cross_signal": latest_payload,
        "latest_cross_2d_candidate": latest_payload_2d,
        "latest_stablecoin_hint": {
            "stablecoin_30d_z": float(stablecoin["stablecoin_30d_z"].iloc[-1]) if pd.notna(stablecoin["stablecoin_30d_z"].iloc[-1]) else None,
            "stablecoin_risk_off_flag": bool(stablecoin["stablecoin_risk_off_flag"].iloc[-1]),
            "stablecoin_extreme_off_flag": bool(stablecoin["stablecoin_extreme_off_flag"].iloc[-1]),
        },
        "latest_vix_hint": {
            "vix_close": float(vix["vix_close"].iloc[-1]) if pd.notna(vix["vix_close"].iloc[-1]) else None,
            "vix_z_90d": float(vix["vix_z_90d"].iloc[-1]) if pd.notna(vix["vix_z_90d"].iloc[-1]) else None,
            "elevated_vix_flag": bool(vix["elevated_vix_flag"].iloc[-1]),
            "extreme_vix_flag": bool(vix["extreme_vix_flag"].iloc[-1]),
        },
        "btc_v2_stats": btc_stats,
        "cross_stats": cross_stats,
        "cross_2d_stats": cross_2d_stats,
        "combo_stats": combo_stats,
        "combo_2d_stats": combo_2d_stats,
        "combo_stablecoin_stats": combo_stablecoin_stats,
        "combo_vix_stats": combo_vix_stats,
        "dsr_diagnostics": dsr_diag.to_dict(orient="records"),
        "pbo_summary": pbo_summary,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Cross-Sectional Shadow Dashboard",
        "",
        "## Latest Cross Signal",
        f"- date: {latest_payload['date']}",
        f"- selected_assets: {', '.join(latest_payload['selected_assets']) or 'none'}",
        f"- portfolio_turnover: {latest_payload['portfolio_turnover']:.4f}",
        "",
        "## Latest Cross 2d Candidate",
        f"- date: {latest_payload_2d['date']}",
        f"- selected_assets: {', '.join(latest_payload_2d['selected_assets']) or 'none'}",
        f"- portfolio_turnover: {latest_payload_2d['portfolio_turnover']:.4f}",
        "",
        "## Latest Stablecoin Hint",
        f"- stablecoin_30d_z: {payload['latest_stablecoin_hint']['stablecoin_30d_z']:.4f}" if payload["latest_stablecoin_hint"]["stablecoin_30d_z"] is not None else "- stablecoin_30d_z: n/a",
        f"- stablecoin_risk_off_flag: {payload['latest_stablecoin_hint']['stablecoin_risk_off_flag']}",
        f"- stablecoin_extreme_off_flag: {payload['latest_stablecoin_hint']['stablecoin_extreme_off_flag']}",
        "",
        "## Latest VIX Hint",
        f"- vix_close: {payload['latest_vix_hint']['vix_close']:.4f}" if payload["latest_vix_hint"]["vix_close"] is not None else "- vix_close: n/a",
        f"- vix_z_90d: {payload['latest_vix_hint']['vix_z_90d']:.4f}" if payload["latest_vix_hint"]["vix_z_90d"] is not None else "- vix_z_90d: n/a",
        f"- elevated_vix_flag: {payload['latest_vix_hint']['elevated_vix_flag']}",
        f"- extreme_vix_flag: {payload['latest_vix_hint']['extreme_vix_flag']}",
        "",
        "## Test Stats",
        f"- btc_v2 test_sharpe: {btc_stats['test_sharpe']:.4f}",
        f"- cross_1d test_sharpe: {cross_stats['test_sharpe']:.4f}",
        f"- cross_2d test_sharpe: {cross_2d_stats['test_sharpe']:.4f}",
        f"- combo test_sharpe: {combo_stats['test_sharpe']:.4f}",
        f"- combo_2d_candidate test_sharpe: {combo_2d_stats['test_sharpe']:.4f}",
        f"- combo_stablecoin_filter test_sharpe: {combo_stablecoin_stats['test_sharpe']:.4f}",
        f"- combo_vix_filter test_sharpe: {combo_vix_stats['test_sharpe']:.4f}",
        f"- btc_v2 test_calmar: {btc_stats['test_calmar']:.4f}",
        f"- cross_1d test_calmar: {cross_stats['test_calmar']:.4f}",
        f"- cross_2d test_calmar: {cross_2d_stats['test_calmar']:.4f}",
        f"- combo test_calmar: {combo_stats['test_calmar']:.4f}",
        f"- combo_2d_candidate test_calmar: {combo_2d_stats['test_calmar']:.4f}",
        f"- combo_stablecoin_filter test_calmar: {combo_stablecoin_stats['test_calmar']:.4f}",
        f"- combo_vix_filter test_calmar: {combo_vix_stats['test_calmar']:.4f}",
        "",
        "## DSR and PBO Gate",
    ]
    if not dsr_diag.empty:
        cross_1d_dsr = dsr_diag.loc[dsr_diag["strategy"] == "cross_1d", "dsr"]
        cross_2d_dsr = dsr_diag.loc[dsr_diag["strategy"] == "cross_2d_candidate", "dsr"]
        lines.extend(
            [
                f"- cross_1d dsr: {float(cross_1d_dsr.iloc[0]):.4f}" if not cross_1d_dsr.empty else "- cross_1d dsr: n/a",
                f"- cross_2d_candidate dsr: {float(cross_2d_dsr.iloc[0]):.4f}" if not cross_2d_dsr.empty else "- cross_2d_candidate dsr: n/a",
            ]
        )
    if pbo_summary:
        lines.extend(
            [
                f"- pbo: {float(pbo_summary.get('pbo', float('nan'))):.4f}" if pd.notna(pbo_summary.get("pbo")) else "- pbo: n/a",
                f"- median_oos_rank_pct: {float(pbo_summary.get('median_oos_rank_pct', float('nan'))):.4f}" if pd.notna(pbo_summary.get("median_oos_rank_pct")) else "- median_oos_rank_pct: n/a",
            ]
        )
    lines += [
        "",
        "## Full Sample",
        f"- btc_v2 sharpe: {btc_stats['sharpe']:.4f}",
        f"- cross_1d sharpe: {cross_stats['sharpe']:.4f}",
        f"- cross_2d sharpe: {cross_2d_stats['sharpe']:.4f}",
        f"- combo sharpe: {combo_stats['sharpe']:.4f}",
        f"- combo_2d_candidate sharpe: {combo_2d_stats['sharpe']:.4f}",
        f"- combo_stablecoin_filter sharpe: {combo_stablecoin_stats['sharpe']:.4f}",
        f"- combo_vix_filter sharpe: {combo_vix_stats['sharpe']:.4f}",
    ]
    dashboard_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {dashboard_md}")


if __name__ == "__main__":
    main()
