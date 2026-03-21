"""System-level backtest for regime engine + router + sizing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.market_structure import MarketStructureSpec, build_market_structure_dataset, classify_regimes
from src.research import perf_slice, perf_stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "system_backtest"
ASSETS = ("BTC", "ETH", "SOL")
FREQUENCIES = ("1d", "4h")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run system-level regime backtests.")
    parser.add_argument("--asset", default="BTC")
    parser.add_argument("--frequency", default="1d")
    parser.add_argument("--fee", type=float, default=0.0004)
    parser.add_argument("--train-end", default="2023-12-31")
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def run_position_backtest(position: pd.Series, ret: pd.Series, fee: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    pos = position.fillna(0.0).clip(-1.0, 1.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    pnl = pos.shift(1).fillna(0.0) * ret.fillna(0.0) - fee * turnover
    eq = (1.0 + pnl).cumprod()
    eq.index = pos.index
    turnover.index = pos.index
    pnl.index = pos.index
    return eq, turnover, pnl


def build_model2_position(
    score: pd.Series,
    regime: pd.Series,
    *,
    uptrend_weight: float = 1.0,
    chop_weight: float = 0.5,
    downtrend_weight: float = 0.0,
    stress_weight: float = 0.0,
) -> pd.Series:
    long_only = score.fillna(0.0).clip(-1.0, 1.0).where(score > 0, 0.0)
    weights = pd.Series(0.0, index=score.index)
    weights.loc[regime == "uptrend"] = uptrend_weight
    weights.loc[regime == "chop"] = chop_weight
    weights.loc[regime == "downtrend"] = downtrend_weight
    weights.loc[regime == "stress"] = stress_weight
    return long_only * weights


def build_model3_position(
    score: pd.Series,
    regime: pd.Series,
    *,
    uptrend_long_weight: float = 1.0,
    chop_long_weight: float = 0.25,
    downtrend_short_weight: float = 0.5,
    stress_short_weight: float = 0.0,
) -> pd.Series:
    score = score.fillna(0.0).clip(-1.0, 1.0)
    long_only = score.where(score > 0, 0.0)
    short_only = score.where(score < 0, 0.0)
    router = pd.Series(0.0, index=score.index)
    router.loc[regime == "uptrend"] = uptrend_long_weight * long_only.loc[regime == "uptrend"]
    router.loc[regime == "chop"] = chop_long_weight * long_only.loc[regime == "chop"]
    router.loc[regime == "downtrend"] = downtrend_short_weight * short_only.loc[regime == "downtrend"]
    router.loc[regime == "stress"] = stress_short_weight * short_only.loc[regime == "stress"]
    return router


def tail_ratio(pnl: pd.Series) -> float:
    pnl = pnl.dropna()
    if len(pnl) < 30:
        return float("nan")
    q95 = float(pnl.quantile(0.95))
    q05 = float(pnl.quantile(0.05))
    if q05 >= 0:
        return float("nan")
    return q95 / abs(q05) if abs(q05) > 0 else float("nan")


def calmar_ratio(ann_return: float, max_drawdown: float) -> float:
    if not np.isfinite(ann_return) or not np.isfinite(max_drawdown) or max_drawdown <= 0:
        return float("nan")
    return ann_return / max_drawdown


def build_system_positions(dataset: pd.DataFrame, feature: str) -> dict[str, pd.Series]:
    score = dataset[feature].fillna(0.0).clip(-1.0, 1.0)
    long_only = score.where(score > 0, 0.0)
    regime = dataset["regime"]

    return {
        "Baseline_always_on_momentum": long_only,
        "Model1_exclude_stress": long_only.where(regime != "stress", 0.0),
        "Model2_regime_aware_sizing": build_model2_position(score, regime),
        "Model3_router_plus_sizing": build_model3_position(score, regime),
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
        "n_days": int(full["n_days"]),
        "avg_turnover": float(turnover.mean()),
        "avg_abs_exposure": float(position.abs().mean()),
        "train_sharpe": float(train["sharpe"]),
        "test_sharpe": float(test["sharpe"]),
        "train_calmar": float(calmar_ratio(train["ann_return"], train["max_drawdown"])),
        "test_calmar": float(calmar_ratio(test["ann_return"], test["max_drawdown"])),
        "train_tail_ratio": float(tail_ratio(pnl.loc[train_mask])),
        "test_tail_ratio": float(tail_ratio(pnl.loc[test_mask])),
    }


def write_report(
    export_dir: Path,
    asset: str,
    frequency: str,
    feature: str,
    rows: pd.DataFrame,
) -> None:
    best_test = rows.sort_values("test_sharpe", ascending=False).iloc[0]
    best_calmar = rows.sort_values("calmar", ascending=False).iloc[0]
    lines = [
        "# System-Level Backtest",
        "",
        "## Setup",
        f"- asset: {asset}",
        f"- frequency: {frequency}",
        f"- alpha core: {feature}",
        f"- models: Baseline / Model1 / Model2 / Model3",
        "",
        "## Snapshot",
        f"- best test Sharpe: {best_test['model']} ({best_test['test_sharpe']:.4f})",
        f"- best full Calmar: {best_calmar['model']} ({best_calmar['calmar']:.4f})",
        "",
        "## Models",
    ]
    for _, row in rows.iterrows():
        lines.extend(
            [
                f"### {row['model']}",
                f"- cagr: {row['cagr']:.4f}",
                f"- sharpe: {row['sharpe']:.4f}",
                f"- test_sharpe: {row['test_sharpe']:.4f}",
                f"- max_drawdown: {row['max_drawdown']:.4f}",
                f"- calmar: {row['calmar']:.4f}",
                f"- tail_ratio: {row['tail_ratio']:.4f}",
                f"- avg_turnover: {row['avg_turnover']:.4f}",
                f"- avg_abs_exposure: {row['avg_abs_exposure']:.4f}",
            ]
        )
    (export_dir / "system_backtest_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(asset: str, frequency: str, fee: float, train_end: str, export: bool) -> pd.DataFrame:
    spec = MarketStructureSpec(asset=asset, frequency=frequency)
    dataset = build_market_structure_dataset(spec)
    feature = dataset["primary_feature_name"].dropna().iloc[0]
    classified, thresholds = classify_regimes(dataset, spec)

    date_index = pd.to_datetime(classified["date"])
    ret = pd.Series(classified["ret"].to_numpy(), index=date_index)
    score = pd.Series(classified[feature].to_numpy(), index=date_index)
    regime = pd.Series(classified["regime"].to_numpy(), index=date_index)
    frame = pd.DataFrame({"ret": ret, feature: score, "regime": regime})

    positions = build_system_positions(frame, feature)
    rows = []
    eq_map: dict[str, pd.Series] = {}
    pnl_map: dict[str, pd.Series] = {}
    turnover_map: dict[str, pd.Series] = {}
    position_map: dict[str, pd.Series] = {}
    for name, position in positions.items():
        eq, turnover, pnl = run_position_backtest(position, ret, fee)
        rows.append(summarize_model(name, eq, pnl, turnover, position, train_end))
        eq_map[name] = eq
        pnl_map[name] = pnl
        turnover_map[name] = turnover
        position_map[name] = position

    summary = pd.DataFrame(rows).sort_values(["test_sharpe", "calmar"], ascending=[False, False])
    summary.insert(0, "frequency", frequency)
    summary.insert(0, "asset", asset)

    if export:
        export_dir = OUTPUT_ROOT / asset / frequency
        export_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(export_dir / "system_backtest_summary.csv", index=False)
        pd.DataFrame(eq_map).to_csv(export_dir / "system_backtest_equity.csv", index_label="date")
        pd.DataFrame(pnl_map).to_csv(export_dir / "system_backtest_pnl.csv", index_label="date")
        pd.DataFrame(turnover_map).to_csv(export_dir / "system_backtest_turnover.csv", index_label="date")
        pd.DataFrame(position_map).to_csv(export_dir / "system_backtest_position.csv", index_label="date")
        (export_dir / "system_backtest_metadata.json").write_text(
            json.dumps(
                {
                    "asset": asset,
                    "frequency": frequency,
                    "fee": fee,
                    "train_end": train_end,
                    "primary_feature": feature,
                    **thresholds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        write_report(export_dir, asset, frequency, feature, summary)
    return summary


def write_cross_asset(summary: pd.DataFrame) -> None:
    out_csv = OUTPUT_ROOT / "cross_asset_system_backtest.csv"
    out_md = OUTPUT_ROOT / "cross_asset_system_backtest.md"
    summary.to_csv(out_csv, index=False)
    lines = [
        "# Cross-Asset System Backtest Summary",
        "",
        "| frequency | asset | model | cagr | sharpe | test_sharpe | max_drawdown | calmar | tail_ratio | avg_turnover | avg_abs_exposure |",
        "|:----------|:------|:------|-----:|-------:|------------:|-------------:|-------:|-----------:|-------------:|-----------------:|",
    ]
    ordered = summary.sort_values(["frequency", "asset", "test_sharpe"], ascending=[True, True, False])
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['frequency']} | {row['asset']} | {row['model']} | {row['cagr']:.4f} | "
            f"{row['sharpe']:.4f} | {row['test_sharpe']:.4f} | {row['max_drawdown']:.4f} | "
            f"{row['calmar']:.4f} | {row['tail_ratio']:.4f} | {row['avg_turnover']:.4f} | {row['avg_abs_exposure']:.4f} |"
        )
    winners = ordered.groupby(["frequency", "asset"], as_index=False).first()
    lines.extend(["", "## Winners"])
    for _, row in winners.iterrows():
        lines.append(
            f"- {row['frequency']} / {row['asset']}: {row['model']} "
            f"(test_sharpe={row['test_sharpe']:.4f}, calmar={row['calmar']:.4f})"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    asset_arg = args.asset.upper()
    freq_arg = args.frequency

    if asset_arg == "ALL" or freq_arg == "all":
        assets = ASSETS if asset_arg == "ALL" else (asset_arg,)
        frequencies = FREQUENCIES if freq_arg == "all" else (freq_arg,)
        frames = []
        for frequency in frequencies:
            for asset in assets:
                frames.append(run_one(asset, frequency, args.fee, args.train_end, not args.no_export))
        summary = pd.concat(frames, ignore_index=True)
        if not args.no_export:
            write_cross_asset(summary)
        top = summary.sort_values(["test_sharpe", "calmar"], ascending=[False, False]).iloc[0]
        print(
            f"cross_asset best={top['frequency']}/{top['asset']}/{top['model']} "
            f"test_sharpe={top['test_sharpe']:.4f} calmar={top['calmar']:.4f}"
        )
        return

    summary = run_one(asset_arg, freq_arg, args.fee, args.train_end, not args.no_export)
    top = summary.iloc[0]
    print(
        f"asset={asset_arg} frequency={freq_arg} best_model={top['model']} "
        f"test_sharpe={top['test_sharpe']:.4f} calmar={top['calmar']:.4f}"
    )


if __name__ == "__main__":
    main()
