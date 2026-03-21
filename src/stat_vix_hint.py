"""Test VIX as a combo filter."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.fetch_vix_history import OUTPUT_DIR as MACRO_DIR, fetch_vix_history
from src.stat_cross_sectional import load_btc_v2_returns
from src.stat_cross_sectional_ab import summarise_returns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "vix_hint" / "v0"
COMBO_PATH = PROJECT_ROOT / "outputs" / "cross_sectional" / "shadow_validation" / "cross_sectional_shadow_equity.csv"


def rolling_zscore(series: pd.Series, window: int = 90, min_periods: int = 30) -> pd.Series:
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return (series - mean) / std.replace(0.0, pd.NA)


def rolling_quantile(series: pd.Series, window: int, quantile: float, min_periods: int) -> pd.Series:
    return series.rolling(window, min_periods=min_periods).quantile(quantile)


def build_dataset() -> pd.DataFrame:
    MACRO_DIR.mkdir(parents=True, exist_ok=True)
    path = MACRO_DIR / "VIX_1d.csv"
    if not path.exists() or path.stat().st_size == 0:
        fetch_vix_history().to_csv(path, index=False)
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    ds = pd.DataFrame({"date": pd.to_datetime(df["date"])})
    ds["vix_close"] = pd.to_numeric(df["vix_close"], errors="coerce")
    ds["vix_z_90d"] = rolling_zscore(ds["vix_close"])
    ds["vix_q70_60d"] = rolling_quantile(ds["vix_close"], window=60, quantile=0.70, min_periods=30)
    ds["vix_q90_60d"] = rolling_quantile(ds["vix_close"], window=60, quantile=0.90, min_periods=30)
    ds["elevated_vix_flag"] = (ds["vix_close"] >= ds["vix_q70_60d"]).astype(float)
    ds["extreme_vix_flag"] = (ds["vix_close"] >= ds["vix_q90_60d"]).astype(float)
    return ds


def load_vix_filter() -> pd.DataFrame:
    path = OUTPUT_ROOT / "dataset.csv"
    if path.exists():
        ds = pd.read_csv(path, parse_dates=["date"])
    else:
        ds = build_dataset()
    keep = [
        "date",
        "vix_close",
        "vix_z_90d",
        "vix_q70_60d",
        "vix_q90_60d",
        "elevated_vix_flag",
        "extreme_vix_flag",
    ]
    return ds[keep].sort_values("date").reset_index(drop=True)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ds = build_dataset()
    ds.to_csv(OUTPUT_ROOT / "dataset.csv", index=False)

    combo = pd.read_csv(COMBO_PATH, parse_dates=["date"]).sort_values("date")
    combo_ret = pd.Series(combo["combo_pnl"].to_numpy(), index=pd.to_datetime(combo["date"]))
    btc_ret = load_btc_v2_returns()

    filt = ds.set_index("date").reindex(combo_ret.index).ffill()
    elevated_lag1 = filt["elevated_vix_flag"].shift(1).fillna(0.0)
    extreme_lag1 = filt["extreme_vix_flag"].shift(1).fillna(0.0)

    variants = {
        "baseline_combo": combo_ret,
        "half_on_elevated_vix": combo_ret * (1.0 - 0.5 * elevated_lag1),
        "flat_on_extreme_vix": combo_ret * (1.0 - 1.0 * extreme_lag1),
        "half_on_elevated_flat_on_extreme": combo_ret * (1.0 - 0.5 * elevated_lag1 - 0.5 * extreme_lag1).clip(lower=0.0),
    }

    rows = []
    for name, ret in variants.items():
        metrics = summarise_returns(ret, btc_ret)
        metrics["variant"] = name
        rows.append(metrics)
    out = pd.DataFrame(rows).sort_values(["test_sharpe", "test_calmar"], ascending=[False, False])
    out.to_csv(OUTPUT_ROOT / "summary.csv", index=False)

    elevated = filt[filt["elevated_vix_flag"] > 0]
    extreme = filt[filt["extreme_vix_flag"] > 0]
    normal = filt[filt["elevated_vix_flag"] == 0]
    bucket_summary = pd.DataFrame(
        [
            {
                "bucket": "elevated_vix",
                "n_obs": int(len(elevated)),
                "avg_vix": float(elevated["vix_close"].mean()) if len(elevated) else float("nan"),
                "avg_combo_next_1d": float(combo_ret.reindex(elevated.index).mean()) if len(elevated) else float("nan"),
            },
            {
                "bucket": "extreme_vix",
                "n_obs": int(len(extreme)),
                "avg_vix": float(extreme["vix_close"].mean()) if len(extreme) else float("nan"),
                "avg_combo_next_1d": float(combo_ret.reindex(extreme.index).mean()) if len(extreme) else float("nan"),
            },
            {
                "bucket": "normal",
                "n_obs": int(len(normal)),
                "avg_vix": float(normal["vix_close"].mean()) if len(normal) else float("nan"),
                "avg_combo_next_1d": float(combo_ret.reindex(normal.index).mean()) if len(normal) else float("nan"),
            },
        ]
    )
    bucket_summary.to_csv(OUTPUT_ROOT / "bucket_summary.csv", index=False)

    best = out.iloc[0]
    lines = [
        "# VIX Hint v0",
        "",
        "## Setup",
        "- source: CBOE public CSV",
        "- role: combo filter / label",
        "- signal: elevated / extreme VIX flags",
        "",
        "## Bucket Summary",
    ]
    for _, row in bucket_summary.iterrows():
        lines.extend(
            [
                f"### {row['bucket']}",
                f"- n_obs: {int(row['n_obs'])}",
                f"- avg_vix: {row['avg_vix']:.4f}",
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
        ]
    )
    (OUTPUT_ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {OUTPUT_ROOT / 'summary.csv'}")
    print(f"Wrote {OUTPUT_ROOT / 'report.md'}")


if __name__ == "__main__":
    main()
