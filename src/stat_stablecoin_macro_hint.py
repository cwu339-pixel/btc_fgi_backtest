"""Test global stablecoin liquidity as a combo risk hint."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.stat_cross_sectional_ab import summarise_returns
from src.stat_cross_sectional import load_btc_v2_returns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "stablecoin_macro_hint" / "v0"
STABLECOIN_PATH = PROJECT_ROOT / "data" / "onchain" / "stablecoin_liquidity_global.csv"
COMBO_PATH = PROJECT_ROOT / "outputs" / "cross_sectional" / "shadow_validation" / "cross_sectional_shadow_equity.csv"


def rolling_zscore(series: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return (series - mean) / std.replace(0.0, pd.NA)


def build_dataset() -> pd.DataFrame:
    df = pd.read_csv(STABLECOIN_PATH, parse_dates=["date"]).sort_values("date")
    ds = pd.DataFrame({"date": pd.to_datetime(df["date"])})
    ds["stablecoin_total_usd"] = pd.to_numeric(df["stablecoin_total_usd"], errors="coerce")
    ds["stablecoin_flow_usd"] = pd.to_numeric(df["stablecoin_flow_usd"], errors="coerce")
    ds["stablecoin_flow_pct"] = pd.to_numeric(df["stablecoin_flow_pct"], errors="coerce")
    ds["stablecoin_30d_pct"] = ds["stablecoin_total_usd"].pct_change(30, fill_method=None)
    ds["stablecoin_30d_z"] = rolling_zscore(ds["stablecoin_30d_pct"])
    ds["stablecoin_risk_off_flag"] = (ds["stablecoin_30d_z"] <= -0.75).astype(float)
    ds["stablecoin_extreme_off_flag"] = (ds["stablecoin_30d_z"] <= -1.25).astype(float)
    return ds


def load_stablecoin_macro_filter() -> pd.DataFrame:
    path = OUTPUT_ROOT / "dataset.csv"
    if path.exists():
        ds = pd.read_csv(path, parse_dates=["date"])
    else:
        ds = build_dataset()
    keep = [
        "date",
        "stablecoin_total_usd",
        "stablecoin_30d_pct",
        "stablecoin_30d_z",
        "stablecoin_risk_off_flag",
        "stablecoin_extreme_off_flag",
    ]
    return ds[keep].sort_values("date").reset_index(drop=True)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ds = build_dataset()
    ds.to_csv(OUTPUT_ROOT / "dataset.csv", index=False)

    combo = pd.read_csv(COMBO_PATH, parse_dates=["date"]).sort_values("date")
    combo_ret = pd.Series(combo["combo_pnl"].to_numpy(), index=pd.to_datetime(combo["date"]))
    btc_ret = load_btc_v2_returns()

    filt = ds[["date", "stablecoin_total_usd", "stablecoin_30d_pct", "stablecoin_30d_z", "stablecoin_risk_off_flag", "stablecoin_extreme_off_flag"]].copy()
    filt["date"] = pd.to_datetime(filt["date"])
    filt = filt.set_index("date").reindex(combo_ret.index).ffill()
    risk_off_lag1 = filt["stablecoin_risk_off_flag"].shift(1).fillna(0.0)
    extreme_off_lag1 = filt["stablecoin_extreme_off_flag"].shift(1).fillna(0.0)

    variants = {
        "baseline_combo": combo_ret,
        "half_on_stablecoin_risk_off": combo_ret * (1.0 - 0.5 * risk_off_lag1),
        "flat_on_stablecoin_extreme_off": combo_ret * (1.0 - 1.0 * extreme_off_lag1),
        "half_on_risk_off_flat_on_extreme": combo_ret * (1.0 - 0.5 * risk_off_lag1 - 0.5 * extreme_off_lag1).clip(lower=0.0),
    }

    rows = []
    for name, ret in variants.items():
        metrics = summarise_returns(ret, btc_ret)
        metrics["variant"] = name
        rows.append(metrics)
    out = pd.DataFrame(rows).sort_values(["test_sharpe", "test_calmar"], ascending=[False, False])
    out.to_csv(OUTPUT_ROOT / "summary.csv", index=False)

    sample = filt.dropna(subset=["stablecoin_30d_z"]).copy()
    risk_off = sample[sample["stablecoin_risk_off_flag"] > 0]
    normal = sample[sample["stablecoin_risk_off_flag"] == 0]
    bucket_summary = pd.DataFrame(
        [
            {
                "bucket": "risk_off",
                "n_obs": int(len(risk_off)),
                "avg_stablecoin_30d_z": float(risk_off["stablecoin_30d_z"].mean()) if len(risk_off) else float("nan"),
                "avg_combo_next_1d": float(combo_ret.reindex(risk_off.index).mean()) if len(risk_off) else float("nan"),
            },
            {
                "bucket": "normal",
                "n_obs": int(len(normal)),
                "avg_stablecoin_30d_z": float(normal["stablecoin_30d_z"].mean()) if len(normal) else float("nan"),
                "avg_combo_next_1d": float(combo_ret.reindex(normal.index).mean()) if len(normal) else float("nan"),
            },
        ]
    )
    bucket_summary.to_csv(OUTPUT_ROOT / "bucket_summary.csv", index=False)

    best = out.iloc[0]
    lines = [
        "# Stablecoin Macro Hint v0",
        "",
        "## Setup",
        "- source: DefiLlama stablecoin history (local CSV)",
        "- role: combo filter / label",
        "- signal: stablecoin_total_usd 30d change with rolling z-score",
        "",
        "## Bucket Summary",
    ]
    for _, row in bucket_summary.iterrows():
        lines.extend(
            [
                f"### {row['bucket']}",
                f"- n_obs: {int(row['n_obs'])}",
                f"- avg_stablecoin_30d_z: {row['avg_stablecoin_30d_z']:.4f}",
                f"- avg_combo_next_1d: {row['avg_combo_next_1d']:.4f}",
                "",
            ]
        )
    lines.append("## A/B Ranking")
    for _, row in out.iterrows():
        lines.extend(
            [
                f"### {row['variant']}",
                f"- test_sharpe: {row['test_sharpe']:.4f}",
                f"- test_calmar: {row['test_calmar']:.4f}",
                f"- max_drawdown: {row['max_drawdown']:.4f}",
                f"- test_corr_vs_btc_v2: {row['test_corr_vs_btc_v2']:.4f}",
                "",
            ]
        )
    lines.extend(
        [
            "## Conclusion",
            f"- best_variant: {best['variant']}",
            "- This line only passes if it improves drawdown / Calmar without clearly damaging Sharpe.",
        ]
    )
    (OUTPUT_ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUTPUT_ROOT / 'summary.csv'}")
    print(f"Wrote {OUTPUT_ROOT / 'report.md'}")


if __name__ == "__main__":
    main()
