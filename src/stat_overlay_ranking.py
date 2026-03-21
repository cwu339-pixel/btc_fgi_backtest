"""Rank BTC 1d overlay variants on one table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.market_structure import MarketStructureSpec, build_market_structure_dataset, classify_regimes
from src.research_v1 import ResearchSpec, build_base_df_for_asset, build_v1_features
from src.stat_flow_router_ab import FLOW_FEATURE
from src.stat_recovery_router_ab import build_recovery_positions
from src.stat_system_backtest import run_position_backtest, summarize_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "system_backtest" / "overlay_ranking" / "BTC" / "1d"
ASSET = "BTC"
FREQUENCY = "1d"
FEE = 0.0004
TRAIN_END = "2023-12-31"


def load_flow_series() -> pd.Series:
    spec = ResearchSpec(asset=ASSET, frequency=FREQUENCY)
    base = build_base_df_for_asset(spec)
    features = build_v1_features(base, spec)
    if FLOW_FEATURE not in features.columns:
        raise KeyError(f"Missing {FLOW_FEATURE}")
    return pd.Series(features[FLOW_FEATURE].to_numpy(), index=features.index)


def build_overlay_positions(score: pd.Series, regime: pd.Series, flow_signal: pd.Series) -> dict[str, pd.Series]:
    recovery_positions = build_recovery_positions(score, regime)
    base = recovery_positions["ProductionCandidate_uptrend_only"]
    recovery_c = recovery_positions["RecoveryC_boost_and_decay"]

    flow_signal = flow_signal.reindex(score.index)
    aligned = flow_signal > 0
    uptrend = regime == "uptrend"
    long_only = score.fillna(0.0).clip(-1.0, 1.0).where(score > 0, 0.0)

    flow_b = pd.Series(0.0, index=score.index)
    flow_b.loc[uptrend & aligned] = long_only.loc[uptrend & aligned]
    flow_b.loc[uptrend & ~aligned] = 0.5 * long_only.loc[uptrend & ~aligned]
    flow_b = flow_b.clip(lower=0.0, upper=1.0)

    combo = recovery_c.copy()
    combo.loc[uptrend & aligned] = recovery_c.loc[uptrend & aligned]
    combo.loc[uptrend & ~aligned] = 0.5 * recovery_c.loc[uptrend & ~aligned]
    combo = combo.clip(lower=0.0, upper=1.0)

    return {
        "Baseline_uptrend_only": base,
        "RecoveryC_overlay": recovery_c,
        "FlowB_overlay": flow_b,
        "RecoveryC_plus_FlowB": combo,
    }


def write_report(out_dir: Path, summary: pd.DataFrame) -> None:
    best = summary.iloc[0]
    base = summary[summary["model"] == "Baseline_uptrend_only"].iloc[0]
    lines = [
        "# BTC 1d Overlay Ranking",
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
    lines.extend(["", "## Ranking"])
    for _, row in summary.iterrows():
        lines.append(
            f"- {row['model']}: test_sharpe={row['test_sharpe']:.4f}, "
            f"test_calmar={row['test_calmar']:.4f}, max_drawdown={row['max_drawdown']:.4f}, "
            f"avg_turnover={row['avg_turnover']:.4f}, avg_abs_exposure={row['avg_abs_exposure']:.4f}"
        )
    (out_dir / "overlay_ranking_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    spec = MarketStructureSpec(asset=ASSET, frequency=FREQUENCY)
    dataset = build_market_structure_dataset(spec)
    classified, thresholds = classify_regimes(dataset, spec)
    feature = classified["primary_feature_name"].dropna().iloc[0]

    idx = pd.to_datetime(classified["date"])
    ret = pd.Series(classified["ret"].to_numpy(), index=idx)
    score = pd.Series(classified[feature].to_numpy(), index=idx)
    regime = pd.Series(classified["regime"].to_numpy(), index=idx)
    flow_signal = load_flow_series()

    positions = build_overlay_positions(score, regime, flow_signal)
    rows = []
    eq_map: dict[str, pd.Series] = {}
    pnl_map: dict[str, pd.Series] = {}
    turnover_map: dict[str, pd.Series] = {}
    position_map: dict[str, pd.Series] = {}

    for model_name, position in positions.items():
        eq, turnover, pnl = run_position_backtest(position, ret, FEE)
        rows.append(summarize_model(model_name, eq, pnl, turnover, position, TRAIN_END))
        eq_map[model_name] = eq
        pnl_map[model_name] = pnl
        turnover_map[model_name] = turnover
        position_map[model_name] = position

    summary = pd.DataFrame(rows).sort_values(["test_sharpe", "test_calmar"], ascending=[False, False])
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_ROOT / "overlay_ranking_summary.csv", index=False)
    pd.DataFrame(eq_map).to_csv(OUTPUT_ROOT / "overlay_ranking_equity.csv", index_label="date")
    pd.DataFrame(pnl_map).to_csv(OUTPUT_ROOT / "overlay_ranking_pnl.csv", index_label="date")
    pd.DataFrame(turnover_map).to_csv(OUTPUT_ROOT / "overlay_ranking_turnover.csv", index_label="date")
    pd.DataFrame(position_map).to_csv(OUTPUT_ROOT / "overlay_ranking_position.csv", index_label="date")
    (OUTPUT_ROOT / "overlay_ranking_metadata.json").write_text(
        json.dumps(
            {
                "asset": ASSET,
                "frequency": FREQUENCY,
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
    write_report(OUTPUT_ROOT, summary)
    top = summary.iloc[0]
    print(
        f"best_model={top['model']} test_sharpe={top['test_sharpe']:.4f} "
        f"test_calmar={top['test_calmar']:.4f}"
    )


if __name__ == "__main__":
    main()
