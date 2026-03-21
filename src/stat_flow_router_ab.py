"""A/B test flow-alignment router overlays against uptrend-only momentum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.market_structure import MarketStructureSpec, build_market_structure_dataset, classify_regimes
from src.research_v1 import ResearchSpec, build_base_df_for_asset, build_v1_features
from src.stat_system_backtest import calmar_ratio, run_position_backtest, summarize_model, tail_ratio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "system_backtest" / "flow_router"
ASSETS = ("BTC", "ETH", "SOL")
TRAIN_END = "2023-12-31"
FEE = 0.0004
FLOW_FEATURE = "spot_taker_buy_7d_rs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run flow-router A/B tests.")
    parser.add_argument("--asset", default="ALL")
    parser.add_argument("--frequency", default="1d", choices=["1d"])
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def load_flow_series(asset: str) -> pd.Series:
    research_spec = ResearchSpec(asset=asset, frequency="1d")
    base = build_base_df_for_asset(research_spec)
    features = build_v1_features(base, research_spec)
    if FLOW_FEATURE not in features.columns:
        raise KeyError(f"Missing {FLOW_FEATURE} for {asset}")
    return pd.Series(features[FLOW_FEATURE].to_numpy(), index=features.index)


def build_flow_router_positions(score: pd.Series, regime: pd.Series, flow_signal: pd.Series) -> dict[str, pd.Series]:
    score = score.fillna(0.0).clip(-1.0, 1.0)
    flow_signal = flow_signal.reindex(score.index)
    long_only = score.where(score > 0, 0.0)
    uptrend = regime == "uptrend"
    aligned = flow_signal > 0

    base = long_only.where(uptrend, 0.0)
    aligned_only = long_only.where(uptrend & aligned, 0.0)

    weighted = pd.Series(0.0, index=score.index)
    weighted.loc[uptrend & aligned] = long_only.loc[uptrend & aligned]
    weighted.loc[uptrend & ~aligned] = 0.5 * long_only.loc[uptrend & ~aligned]

    confirm_and_reject = pd.Series(0.0, index=score.index)
    confirm_and_reject.loc[uptrend & aligned] = 1.0 * long_only.loc[uptrend & aligned]
    confirm_and_reject.loc[uptrend & ~aligned] = 0.0

    return {
        "Baseline_uptrend_only": base,
        "FlowA_aligned_only": aligned_only,
        "FlowB_misaligned_half": weighted.clip(lower=0.0, upper=1.0),
        "FlowC_confirm_or_flat": confirm_and_reject,
    }


def run_one(asset: str, export: bool) -> pd.DataFrame:
    spec = MarketStructureSpec(asset=asset, frequency="1d")
    dataset = build_market_structure_dataset(spec)
    classified, thresholds = classify_regimes(dataset, spec)
    feature = classified["primary_feature_name"].dropna().iloc[0]

    idx = pd.to_datetime(classified["date"])
    ret = pd.Series(classified["ret"].to_numpy(), index=idx)
    score = pd.Series(classified[feature].to_numpy(), index=idx)
    regime = pd.Series(classified["regime"].to_numpy(), index=idx)
    flow_signal = load_flow_series(asset)

    positions = build_flow_router_positions(score, regime, flow_signal)
    rows = []
    eq_map: dict[str, pd.Series] = {}
    pnl_map: dict[str, pd.Series] = {}
    turnover_map: dict[str, pd.Series] = {}
    position_map: dict[str, pd.Series] = {}

    for name, position in positions.items():
        eq, turnover, pnl = run_position_backtest(position, ret, FEE)
        row = summarize_model(name, eq, pnl, turnover, position, TRAIN_END)
        row["flow_coverage_ratio"] = float(flow_signal.reindex(position.index).notna().mean())
        rows.append(row)
        eq_map[name] = eq
        pnl_map[name] = pnl
        turnover_map[name] = turnover
        position_map[name] = position

    summary = pd.DataFrame(rows).sort_values(["test_sharpe", "test_calmar"], ascending=[False, False])
    summary.insert(0, "frequency", "1d")
    summary.insert(0, "asset", asset)

    if export:
        out_dir = OUTPUT_ROOT / asset / "1d"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(out_dir / "flow_router_summary.csv", index=False)
        pd.DataFrame(eq_map).to_csv(out_dir / "flow_router_equity.csv", index_label="date")
        pd.DataFrame(pnl_map).to_csv(out_dir / "flow_router_pnl.csv", index_label="date")
        pd.DataFrame(turnover_map).to_csv(out_dir / "flow_router_turnover.csv", index_label="date")
        pd.DataFrame(position_map).to_csv(out_dir / "flow_router_position.csv", index_label="date")
        (out_dir / "flow_router_metadata.json").write_text(
            json.dumps(
                {
                    "asset": asset,
                    "frequency": "1d",
                    "fee": FEE,
                    "train_end": TRAIN_END,
                    "primary_feature": feature,
                    "flow_feature": FLOW_FEATURE,
                    **thresholds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        write_report(out_dir, asset, summary)
    return summary


def write_report(out_dir: Path, asset: str, summary: pd.DataFrame) -> None:
    best = summary.iloc[0]
    base = summary[summary["model"] == "Baseline_uptrend_only"].iloc[0]
    lines = [
        "# Flow Router A/B",
        "",
        "## Setup",
        f"- asset: {asset}",
        "- frequency: 1d",
        f"- flow feature: {FLOW_FEATURE}",
        "",
        "## Snapshot",
        f"- best model: {best['model']} (test_sharpe={best['test_sharpe']:.4f}, test_calmar={best['test_calmar']:.4f})",
        "",
        "## Delta vs Baseline",
    ]
    for _, row in summary.iterrows():
        if row["model"] == "Baseline_uptrend_only":
            continue
        lines.append(
            f"- {row['model']}: "
            f"test_sharpe_delta={row['test_sharpe'] - base['test_sharpe']:.4f}, "
            f"test_calmar_delta={row['test_calmar'] - base['test_calmar']:.4f}, "
            f"max_dd_delta={row['max_drawdown'] - base['max_drawdown']:.4f}, "
            f"turnover_delta={row['avg_turnover'] - base['avg_turnover']:.4f}"
        )
    lines.extend(["", "## Models"])
    for _, row in summary.iterrows():
        lines.extend(
            [
                f"### {row['model']}",
                f"- sharpe: {row['sharpe']:.4f}",
                f"- test_sharpe: {row['test_sharpe']:.4f}",
                f"- test_calmar: {row['test_calmar']:.4f}",
                f"- max_drawdown: {row['max_drawdown']:.4f}",
                f"- avg_turnover: {row['avg_turnover']:.4f}",
                f"- avg_abs_exposure: {row['avg_abs_exposure']:.4f}",
            ]
        )
    (out_dir / "flow_router_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cross_asset(summary: pd.DataFrame) -> None:
    out_dir = OUTPUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "cross_asset_flow_router.csv", index=False)
    lines = [
        "# Cross-Asset Flow Router Summary",
        "",
        "| asset | model | sharpe | test_sharpe | test_calmar | max_drawdown | avg_turnover | avg_abs_exposure |",
        "|:------|:------|-------:|------------:|------------:|-------------:|-------------:|-----------------:|",
    ]
    ordered = summary.sort_values(["asset", "test_sharpe"], ascending=[True, False])
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['asset']} | {row['model']} | {row['sharpe']:.4f} | {row['test_sharpe']:.4f} | "
            f"{row['test_calmar']:.4f} | {row['max_drawdown']:.4f} | {row['avg_turnover']:.4f} | {row['avg_abs_exposure']:.4f} |"
        )
    winners = ordered.groupby("asset", as_index=False).first()
    lines.extend(["", "## Winners"])
    for _, row in winners.iterrows():
        lines.append(
            f"- {row['asset']}: {row['model']} "
            f"(test_sharpe={row['test_sharpe']:.4f}, test_calmar={row['test_calmar']:.4f})"
        )
    (out_dir / "cross_asset_flow_router.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    assets = ASSETS if args.asset.upper() == "ALL" else (args.asset.upper(),)
    frames = [run_one(asset, not args.no_export) for asset in assets]
    summary = pd.concat(frames, ignore_index=True)
    if not args.no_export:
        write_cross_asset(summary)
    top = summary.sort_values(["test_sharpe", "test_calmar"], ascending=[False, False]).iloc[0]
    print(
        f"best_asset={top['asset']} best_model={top['model']} "
        f"test_sharpe={top['test_sharpe']:.4f} test_calmar={top['test_calmar']:.4f}"
    )


if __name__ == "__main__":
    main()
