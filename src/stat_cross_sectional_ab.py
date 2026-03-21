"""Secondary pillar diagnostics: cost stress, universe drift, and combo A/B."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.stat_cross_sectional import (
    OUTPUT_ROOT as CROSS_OUTPUT_ROOT,
    build_panel,
    build_positions,
    load_btc_v2_returns,
    perf_stats,
    run_portfolio_backtest,
)
from src.stat_system_backtest import calmar_ratio, tail_ratio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cross_sectional" / "ab"
TRAIN_END = pd.Timestamp("2023-12-31")


def summarise_returns(ret: pd.Series, benchmark: pd.Series) -> dict[str, float]:
    eq = (1.0 + ret.fillna(0.0)).cumprod()
    stats = perf_stats(eq)
    test_mask = eq.index > TRAIN_END
    test_eq = eq.loc[test_mask]
    test_stats = perf_stats(test_eq)
    aligned = ret.align(benchmark, join="inner")
    test_aligned = aligned[0].loc[aligned[0].index > TRAIN_END].align(
        aligned[1].loc[aligned[1].index > TRAIN_END],
        join="inner",
    )
    return {
        "total_return": float(stats["total_return"]),
        "cagr": float(stats["ann_return"]),
        "sharpe": float(stats["sharpe"]),
        "max_drawdown": float(stats["max_drawdown"]),
        "calmar": float(calmar_ratio(stats["ann_return"], stats["max_drawdown"])),
        "tail_ratio": float(tail_ratio(ret)),
        "test_sharpe": float(test_stats["sharpe"]),
        "test_calmar": float(calmar_ratio(test_stats["ann_return"], test_stats["max_drawdown"])),
        "corr_vs_btc_v2": float(aligned[0].corr(aligned[1])) if len(aligned[0]) >= 20 else float("nan"),
        "test_corr_vs_btc_v2": float(test_aligned[0].corr(test_aligned[1])) if len(test_aligned[0]) >= 20 else float("nan"),
    }


def build_cross_sectional_returns(config: pd.Series) -> tuple[pd.Series, pd.Series]:
    assets = tuple(str(config["universe_assets"]).split(","))
    close_df = build_panel(assets)[list(assets)]
    positions = build_positions(
        close_df=close_df,
        lookback=int(config["lookback"]),
        top_k=int(config["top_k"]),
        rebalance_days=int(config["rebalance_days"]),
        ranking_method=str(config["ranking_method"]),
        weighting_method=str(config["weighting_method"]),
    )
    _, turnover, pnl = run_portfolio_backtest(
        positions,
        close_df,
        fee=float(config["fee"]),
        slip=float(config["slip"]),
    )
    return pnl, turnover


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(CROSS_OUTPUT_ROOT / "summary.csv")
    best = summary.sort_values(["test_sharpe", "test_calmar"], ascending=[False, False]).iloc[0]
    btc_v2_ret = load_btc_v2_returns()

    best_pnl, best_turnover = build_cross_sectional_returns(best)

    cost_rows: list[dict[str, float | str]] = []
    for _, row in summary[
        ["fee", "slip", "universe_size", "liquidity_threshold", "universe_assets", "lookback", "top_k", "rebalance_days", "ranking_method", "weighting_method"]
    ].drop_duplicates().iterrows():
        if (
            int(row["universe_size"]) == int(best["universe_size"])
            and float(row["liquidity_threshold"]) == float(best["liquidity_threshold"])
            and str(row["universe_assets"]) == str(best["universe_assets"])
            and int(row["lookback"]) == int(best["lookback"])
            and int(row["top_k"]) == int(best["top_k"])
            and int(row["rebalance_days"]) == int(best["rebalance_days"])
            and str(row["ranking_method"]) == str(best["ranking_method"])
            and str(row["weighting_method"]) == str(best["weighting_method"])
        ):
            cfg = row.copy()
            pnl, turnover = build_cross_sectional_returns(cfg)
            metrics = summarise_returns(pnl, btc_v2_ret)
            metrics.update(
                {
                    "scenario": "cost_stress",
                    "fee": float(cfg["fee"]),
                    "slip": float(cfg["slip"]),
                    "avg_turnover": float(turnover.mean()),
                }
            )
            cost_rows.append(metrics)

    drift_rows: list[dict[str, float | str]] = []
    drift_candidates = summary[
        (
            (summary["lookback"] == best["lookback"])
            & (summary["top_k"] == best["top_k"])
            & (summary["rebalance_days"] == best["rebalance_days"])
            & (summary["ranking_method"] == best["ranking_method"])
            & (summary["weighting_method"] == best["weighting_method"])
            & (summary["fee"] == best["fee"])
            & (summary["slip"] == best["slip"])
        )
    ][["liquidity_threshold", "universe_size", "universe_assets"]].drop_duplicates()
    for _, row in drift_candidates.iterrows():
        cfg = best.copy()
        cfg["liquidity_threshold"] = row["liquidity_threshold"]
        cfg["universe_size"] = row["universe_size"]
        cfg["universe_assets"] = row["universe_assets"]
        pnl, turnover = build_cross_sectional_returns(cfg)
        metrics = summarise_returns(pnl, btc_v2_ret)
        metrics.update(
            {
                "scenario": "universe_drift",
                "fee": float(best["fee"]),
                "slip": float(best["slip"]),
                "liquidity_threshold": float(cfg["liquidity_threshold"]),
                "universe_size": int(cfg["universe_size"]),
                "universe_assets": str(cfg["universe_assets"]),
                "avg_turnover": float(turnover.mean()),
            }
        )
        drift_rows.append(metrics)

    combo_weights = [
        ("btc_only", 1.0, 0.0),
        ("cross_only", 0.0, 1.0),
        ("combo_75_25", 0.75, 0.25),
        ("combo_50_50", 0.50, 0.50),
        ("combo_25_75", 0.25, 0.75),
    ]
    combo_rows: list[dict[str, float | str]] = []
    for name, btc_w, cross_w in combo_weights:
        combined = btc_v2_ret.mul(btc_w, fill_value=0.0).add(best_pnl.mul(cross_w, fill_value=0.0), fill_value=0.0)
        metrics = summarise_returns(combined, btc_v2_ret)
        metrics.update(
            {
                "portfolio": name,
                "btc_weight": btc_w,
                "cross_weight": cross_w,
            }
        )
        combo_rows.append(metrics)

    cost_df = pd.DataFrame(cost_rows).sort_values(["test_sharpe", "test_calmar"], ascending=[False, False])
    drift_df = pd.DataFrame(drift_rows).sort_values(["test_sharpe", "test_calmar"], ascending=[False, False])
    combo_df = pd.DataFrame(combo_rows).sort_values(["test_sharpe", "test_calmar"], ascending=[False, False])

    cost_df.to_csv(OUTPUT_ROOT / "cost_stress.csv", index=False)
    drift_df.to_csv(OUTPUT_ROOT / "universe_drift.csv", index=False)
    combo_df.to_csv(OUTPUT_ROOT / "combo_ab.csv", index=False)

    report = [
        "# Cross-Sectional Extension Report",
        "",
        "## Best Cross-Sectional Candidate",
        f"- universe_assets: {best['universe_assets']}",
        f"- lookback: {int(best['lookback'])}",
        f"- top_k: {int(best['top_k'])}",
        f"- rebalance_days: {int(best['rebalance_days'])}",
        f"- ranking_method: {best['ranking_method']}",
        f"- weighting_method: {best['weighting_method']}",
        f"- fee/slip: {best['fee']:.4f}/{best['slip']:.4f}",
        f"- test_sharpe: {best['test_sharpe']:.4f}",
        f"- test_corr_vs_btc_v2: {best['test_corr_vs_btc_v2']:.4f}",
        "",
        "## Cost Stress",
        f"- best scenario test_sharpe: {cost_df.iloc[0]['test_sharpe']:.4f}",
        f"- worst scenario test_sharpe: {cost_df.iloc[-1]['test_sharpe']:.4f}",
        f"- cost rows: {len(cost_df)}",
        "",
        "## Universe Drift",
        f"- best drift universe: {drift_df.iloc[0]['universe_assets']}",
        f"- best drift test_sharpe: {drift_df.iloc[0]['test_sharpe']:.4f}",
        f"- worst drift test_sharpe: {drift_df.iloc[-1]['test_sharpe']:.4f}",
        f"- drift rows: {len(drift_df)}",
        "",
        "## Combo A/B",
        f"- best portfolio: {combo_df.iloc[0]['portfolio']}",
        f"- best portfolio test_sharpe: {combo_df.iloc[0]['test_sharpe']:.4f}",
        f"- best portfolio test_calmar: {combo_df.iloc[0]['test_calmar']:.4f}",
        f"- best portfolio test_corr_vs_btc_v2: {combo_df.iloc[0]['test_corr_vs_btc_v2']:.4f}",
    ]
    (OUTPUT_ROOT / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
