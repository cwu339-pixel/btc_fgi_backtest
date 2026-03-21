"""A/B test recovery-aware router variants against the BTC 1d production candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.market_structure import MarketStructureSpec, build_market_structure_dataset, classify_regimes
from src.research import perf_slice, perf_stats
from src.stat_system_backtest import calmar_ratio, run_position_backtest, tail_ratio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "system_backtest" / "recovery_router"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run recovery-aware router A/B tests.")
    parser.add_argument("--asset", default="BTC")
    parser.add_argument("--frequency", default="1d")
    parser.add_argument("--fee", type=float, default=0.0004)
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def _base_long_only(score: pd.Series, regime: pd.Series) -> pd.Series:
    return score.fillna(0.0).clip(-1.0, 1.0).where((score > 0) & (regime == "uptrend"), 0.0)


def build_recovery_positions(score: pd.Series, regime: pd.Series) -> dict[str, pd.Series]:
    score = score.fillna(0.0).clip(-1.0, 1.0)
    long_only = score.where(score > 0, 0.0)
    prev_regime = regime.shift(1)
    stress_to_uptrend = (prev_regime == "stress") & (regime == "uptrend")

    base = _base_long_only(score, regime)

    # Variant A: keep the production candidate, but give recovery bars the top-of-family uptrend weight.
    recovery_boost = base.copy()
    recovery_boost.loc[stress_to_uptrend] = (long_only.loc[stress_to_uptrend] * 1.25).clip(upper=1.0)

    # Variant B: allow the recovery bar and the next 2 bars to retain uptrend exposure through chop only.
    recovery_flag = (
        stress_to_uptrend.astype(int)
        .rolling(window=3, min_periods=1)
        .max()
        .fillna(0)
        .astype(bool)
    )
    recovery_hold = base.copy()
    hold_mask = recovery_flag & regime.isin(["uptrend", "chop"])
    recovery_hold.loc[hold_mask] = long_only.loc[hold_mask]
    recovery_hold.loc[regime.isin(["downtrend", "stress"])] = 0.0

    # Variant C: combine the two ideas, but keep all risk changes inside the approved narrow family.
    recovery_combo = base.copy()
    recovery_combo.loc[hold_mask] = long_only.loc[hold_mask] * np.where(
        stress_to_uptrend.loc[hold_mask],
        1.25,
        0.75,
    )
    recovery_combo = recovery_combo.clip(lower=0.0, upper=1.0)
    recovery_combo.loc[regime.isin(["downtrend", "stress"])] = 0.0

    return {
        "ProductionCandidate_uptrend_only": base,
        "RecoveryA_boost_first_bar": recovery_boost,
        "RecoveryB_hold_through_chop_3bar": recovery_hold,
        "RecoveryC_boost_and_decay": recovery_combo,
    }


def summarize_model(
    name: str,
    eq: pd.Series,
    pnl: pd.Series,
    turnover: pd.Series,
    position: pd.Series,
    train_end: str,
) -> dict[str, float | str]:
    full = perf_stats(eq)
    train = perf_slice(eq, eq.index.min().strftime("%Y-%m-%d"), train_end)
    test_start = (pd.Timestamp(train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    test = perf_slice(eq, test_start, eq.index.max().strftime("%Y-%m-%d"))
    train_mask = eq.index <= pd.Timestamp(train_end)
    test_mask = eq.index > pd.Timestamp(train_end)

    return {
        "model": name,
        "total_return": float(full["total_return"]),
        "cagr": float(full["ann_return"]),
        "max_drawdown": float(full["max_drawdown"]),
        "calmar": float(calmar_ratio(full["ann_return"], full["max_drawdown"])),
        "vol": float(full["vol"]),
        "sharpe": float(full["sharpe"]),
        "tail_ratio": float(tail_ratio(pnl)),
        "avg_turnover": float(turnover.mean()),
        "avg_abs_exposure": float(position.abs().mean()),
        "train_sharpe": float(train["sharpe"]),
        "test_sharpe": float(test["sharpe"]),
        "train_calmar": float(calmar_ratio(train["ann_return"], train["max_drawdown"])),
        "test_calmar": float(calmar_ratio(test["ann_return"], test["max_drawdown"])),
        "train_tail_ratio": float(tail_ratio(pnl.loc[train_mask])),
        "test_tail_ratio": float(tail_ratio(pnl.loc[test_mask])),
    }


def write_report(export_dir: Path, asset: str, frequency: str, feature: str, summary: pd.DataFrame) -> None:
    best_test = summary.sort_values(["test_sharpe", "test_calmar"], ascending=[False, False]).iloc[0]
    base = summary[summary["model"] == "ProductionCandidate_uptrend_only"].iloc[0]
    lines = [
        "# Recovery-Aware Router A/B",
        "",
        "## Setup",
        f"- asset: {asset}",
        f"- frequency: {frequency}",
        f"- alpha core: {feature}",
        f"- baseline: ProductionCandidate_uptrend_only",
        "",
        "## Snapshot",
        f"- best test Sharpe: {best_test['model']} ({best_test['test_sharpe']:.4f})",
        "",
        "## Delta vs Baseline",
    ]
    for _, row in summary.iterrows():
        if row["model"] == "ProductionCandidate_uptrend_only":
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
                f"- calmar: {row['calmar']:.4f}",
                f"- test_calmar: {row['test_calmar']:.4f}",
                f"- max_drawdown: {row['max_drawdown']:.4f}",
                f"- avg_turnover: {row['avg_turnover']:.4f}",
                f"- avg_abs_exposure: {row['avg_abs_exposure']:.4f}",
            ]
        )
    (export_dir / "recovery_router_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    asset = args.asset.upper()
    spec = MarketStructureSpec(asset=asset, frequency=args.frequency)
    dataset = build_market_structure_dataset(spec)
    feature = dataset["primary_feature_name"].dropna().iloc[0]
    classified, thresholds = classify_regimes(dataset, spec)

    date_index = pd.to_datetime(classified["date"])
    ret = pd.Series(classified["ret"].to_numpy(), index=date_index)
    score = pd.Series(classified[feature].to_numpy(), index=date_index)
    regime = pd.Series(classified["regime"].to_numpy(), index=date_index)

    positions = build_recovery_positions(score, regime)
    rows = []
    eq_map: dict[str, pd.Series] = {}
    pnl_map: dict[str, pd.Series] = {}
    turnover_map: dict[str, pd.Series] = {}
    position_map: dict[str, pd.Series] = {}

    for name, position in positions.items():
        eq, turnover, pnl = run_position_backtest(position, ret, args.fee)
        rows.append(summarize_model(name, eq, pnl, turnover, position, args.train_end))
        eq_map[name] = eq
        pnl_map[name] = pnl
        turnover_map[name] = turnover
        position_map[name] = position

    summary = pd.DataFrame(rows).sort_values(["test_sharpe", "test_calmar"], ascending=[False, False])

    if not args.no_export:
        export_dir = OUTPUT_ROOT / asset / args.frequency
        export_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(export_dir / "recovery_router_summary.csv", index=False)
        pd.DataFrame(eq_map).to_csv(export_dir / "recovery_router_equity.csv", index_label="date")
        pd.DataFrame(pnl_map).to_csv(export_dir / "recovery_router_pnl.csv", index_label="date")
        pd.DataFrame(turnover_map).to_csv(export_dir / "recovery_router_turnover.csv", index_label="date")
        pd.DataFrame(position_map).to_csv(export_dir / "recovery_router_position.csv", index_label="date")
        (export_dir / "recovery_router_metadata.json").write_text(
            json.dumps(
                {
                    "asset": asset,
                    "frequency": args.frequency,
                    "fee": args.fee,
                    "train_end": args.train_end,
                    "primary_feature": feature,
                    **thresholds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        write_report(export_dir, asset, args.frequency, feature, summary)

    top = summary.iloc[0]
    print(
        f"asset={asset} frequency={args.frequency} best_model={top['model']} "
        f"test_sharpe={top['test_sharpe']:.4f} test_calmar={top['test_calmar']:.4f}"
    )


if __name__ == "__main__":
    main()
