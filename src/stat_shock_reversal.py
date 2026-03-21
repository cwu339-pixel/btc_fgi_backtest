"""Deeper research pass for a shock-driven reversal third-pillar candidate."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from src.research import perf_slice, perf_stats
from src.research_v1 import ALT_DATA_ROOT
from src.stat_system_backtest import calmar_ratio, tail_ratio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKET_DIR = PROJECT_ROOT / "data" / "market"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "shock_reversal" / "v0"
COMBO_BENCHMARK_PATH = (
    PROJECT_ROOT / "outputs" / "cross_sectional" / "shadow_validation" / "cross_sectional_shadow_equity.csv"
)
ASSETS = ("BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT")
TRAIN_END = "2023-12-31"
COSTS = ((0.0010, 0.0010), (0.0010, 0.0020))
STRESS_THRESHOLDS = (1.25, 1.50, 1.75, 2.00)
HOLD_DAYS = (1, 3, 5, 7)
TOP_KS = (0, 1, 2)
TRIGGERS = ("next_day_green", "midpoint_reclaim", "ranked_rebound")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run shock reversal v0 research.")
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


@lru_cache(maxsize=None)
def load_asset_ohlcv(asset: str) -> pd.DataFrame:
    path = MARKET_DIR / f"binance_spot_daily_{asset}USDT.csv"
    if path.exists() and path.stat().st_size > 0:
        try:
            df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
            if {"open", "high", "low", "close", "volume"}.issubset(df.columns):
                out = df[["date", "open", "high", "low", "close", "volume"]].copy()
                out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.tz_localize(None)
                return out.set_index("date")
        except EmptyDataError:
            pass
    feather = ALT_DATA_ROOT / f"{asset}_USDT-1d.feather"
    if feather.exists():
        df = pd.read_feather(feather)
        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("date") or cols.get("timestamp") or df.columns[0]
        req = {k: cols.get(k) for k in ("open", "high", "low", "close", "volume")}
        if all(req.values()):
            out = pd.DataFrame(
                {
                    "date": pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.tz_localize(None),
                    "open": pd.to_numeric(df[req["open"]], errors="coerce"),
                    "high": pd.to_numeric(df[req["high"]], errors="coerce"),
                    "low": pd.to_numeric(df[req["low"]], errors="coerce"),
                    "close": pd.to_numeric(df[req["close"]], errors="coerce"),
                    "volume": pd.to_numeric(df[req["volume"]], errors="coerce"),
                }
            )
            return out.set_index("date").sort_index()
    raise FileNotFoundError(asset)


@lru_cache(maxsize=1)
def discovered_assets() -> tuple[str, ...]:
    ok = []
    for asset in ASSETS:
        try:
            df = load_asset_ohlcv(asset)
        except Exception:
            continue
        if len(df.dropna()) >= 500:
            ok.append(asset)
    return tuple(ok)


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assets = discovered_assets()
    opens = pd.DataFrame({a: load_asset_ohlcv(a)["open"] for a in assets}).sort_index()
    highs = pd.DataFrame({a: load_asset_ohlcv(a)["high"] for a in assets}).sort_index()
    lows = pd.DataFrame({a: load_asset_ohlcv(a)["low"] for a in assets}).sort_index()
    closes = pd.DataFrame({a: load_asset_ohlcv(a)["close"] for a in assets}).sort_index()
    volumes = pd.DataFrame({a: load_asset_ohlcv(a)["volume"] for a in assets}).sort_index()
    return opens, highs, lows, closes, volumes


def zscore_last(frame: pd.DataFrame, window: int = 60, min_periods: int = 20) -> pd.DataFrame:
    mean = frame.rolling(window, min_periods=min_periods).mean()
    std = frame.rolling(window, min_periods=min_periods).std()
    return (frame - mean) / std.replace(0.0, np.nan)


def load_combo_benchmark() -> pd.Series:
    df = pd.read_csv(COMBO_BENCHMARK_PATH, parse_dates=["date"]).sort_values("date")
    return pd.Series(df["combo_pnl"].to_numpy(), index=pd.to_datetime(df["date"]))


def run_backtest(position: pd.DataFrame, ret_df: pd.DataFrame, fee: float, slip: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    turnover = position.diff().abs().sum(axis=1).fillna(position.abs().sum(axis=1))
    pnl = (position.shift(1).fillna(0.0) * ret_df.fillna(0.0)).sum(axis=1) - (fee + slip) * turnover
    eq = (1.0 + pnl).cumprod()
    return eq, turnover, pnl


def summarize(eq: pd.Series, pnl: pd.Series, turnover: pd.Series, benchmark: pd.Series, train_end: str) -> dict[str, float]:
    full = perf_stats(eq)
    test_start = (pd.Timestamp(train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    test = perf_slice(eq, test_start, eq.index.max().strftime("%Y-%m-%d"))
    aligned = pnl.align(benchmark, join="inner")
    test_mask = aligned[0].index > pd.Timestamp(train_end)
    test_aligned = aligned[0].loc[test_mask].align(aligned[1].loc[test_mask], join="inner")
    return {
        "total_return": float(full["total_return"]),
        "cagr": float(full["ann_return"]),
        "sharpe": float(full["sharpe"]),
        "max_drawdown": float(full["max_drawdown"]),
        "calmar": float(calmar_ratio(full["ann_return"], full["max_drawdown"])),
        "tail_ratio": float(tail_ratio(pnl)),
        "avg_turnover": float(turnover.mean()),
        "test_sharpe": float(test["sharpe"]),
        "test_calmar": float(calmar_ratio(test["ann_return"], test["max_drawdown"])),
        "corr_vs_combo": float(aligned[0].corr(aligned[1])) if len(aligned[0]) >= 20 else float("nan"),
        "test_corr_vs_combo": float(test_aligned[0].corr(test_aligned[1])) if len(test_aligned[0]) >= 20 else float("nan"),
    }


def build_shock_signals(
    opens: pd.DataFrame,
    highs: pd.DataFrame,
    lows: pd.DataFrame,
    closes: pd.DataFrame,
    volumes: pd.DataFrame,
    threshold: float,
    trigger: str,
    top_k: int,
    hold_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ret_df = closes.pct_change(fill_method=None).fillna(0.0)
    intraday_range = (highs - lows) / closes.replace(0.0, np.nan)
    ret_z = zscore_last(ret_df.abs())
    range_z = zscore_last(intraday_range)
    vol_z = zscore_last(np.log(volumes.replace(0.0, np.nan)))
    stress = 0.5 * ret_z + 0.3 * range_z + 0.2 * vol_z
    shock_flag = ((stress > threshold) & (ret_df < 0)).astype(bool)

    prev_shock = shock_flag.shift(1, fill_value=False)
    midpoint = (opens.shift(1) + closes.shift(1)) / 2.0
    next_day_green = prev_shock & (ret_df > 0)
    midpoint_reclaim = prev_shock & (closes > midpoint)
    ranked_rebound = prev_shock & (ret_df > 0)

    if trigger == "next_day_green":
        trigger_mask = next_day_green
        rank_score = ret_df.where(trigger_mask)
    elif trigger == "midpoint_reclaim":
        trigger_mask = midpoint_reclaim
        rank_score = ((closes / midpoint) - 1.0).where(trigger_mask)
    elif trigger == "ranked_rebound":
        trigger_mask = ranked_rebound
        rank_score = (ret_df / (stress.shift(1).abs() + 1e-9)).where(trigger_mask)
    else:
        raise ValueError(f"Unknown trigger: {trigger}")

    if top_k == 0:
        raw = trigger_mask.astype(float)
        positions = raw.div(raw.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    else:
        positions = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
        for dt in positions.index:
            scores = rank_score.loc[dt].dropna().sort_values(ascending=False)
            winners = scores.head(top_k).index.tolist()
            if winners:
                positions.loc[dt, winners] = 1.0 / len(winners)

    if hold_days > 1:
        held = positions.copy()
        for lag in range(1, hold_days):
            held = held.add(positions.shift(lag).fillna(0.0), fill_value=0.0)
        positions = held.div(held.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
    return positions, stress


def walk_forward_table(position: pd.DataFrame, ret_df: pd.DataFrame, fee: float, slip: float) -> pd.DataFrame:
    eq, turnover, pnl = run_backtest(position, ret_df, fee, slip)
    windows: list[dict[str, float | str]] = []
    start = pd.Timestamp("2023-01-01")
    end = eq.index.max()
    current = start
    while current + pd.Timedelta(days=180) <= end:
        test_end = min(current + pd.Timedelta(days=180), end)
        mask = (eq.index >= current) & (eq.index <= test_end)
        if mask.sum() < 30:
            break
        eq_slice = eq.loc[mask]
        pnl_slice = pnl.loc[mask]
        turn_slice = turnover.loc[mask]
        stats = perf_stats(eq_slice)
        windows.append(
            {
                "window_start": current.strftime("%Y-%m-%d"),
                "window_end": test_end.strftime("%Y-%m-%d"),
                "sharpe": float(stats["sharpe"]),
                "calmar": float(calmar_ratio(stats["ann_return"], stats["max_drawdown"])),
                "max_drawdown": float(stats["max_drawdown"]),
                "avg_turnover": float(turn_slice.mean()),
            }
        )
        current = current + pd.Timedelta(days=30)
    return pd.DataFrame(windows)


def main() -> None:
    args = parse_args()
    opens, highs, lows, closes, volumes = build_panel()
    ret_df = closes.pct_change(fill_method=None).fillna(0.0)
    benchmark = load_combo_benchmark()

    rows: list[dict[str, float | int | str]] = []
    position_cache: dict[tuple[float, str, int, int, float, float], pd.DataFrame] = {}
    wf_cache: dict[tuple[float, str, int, int, float, float], pd.DataFrame] = {}

    for threshold in STRESS_THRESHOLDS:
        for trigger in TRIGGERS:
            for top_k in TOP_KS:
                for hold_days in HOLD_DAYS:
                    for fee, slip in COSTS:
                        positions, _ = build_shock_signals(
                            opens=opens,
                            highs=highs,
                            lows=lows,
                            closes=closes,
                            volumes=volumes,
                            threshold=threshold,
                            trigger=trigger,
                            top_k=top_k,
                            hold_days=hold_days,
                        )
                        eq, turnover, pnl = run_backtest(positions, ret_df, fee, slip)
                        row = {
                            "threshold": threshold,
                            "trigger": trigger,
                            "top_k": top_k,
                            "hold_days": hold_days,
                            "fee": fee,
                            "slip": slip,
                        }
                        row.update(summarize(eq, pnl, turnover, benchmark, args.train_end))
                        rows.append(row)
                        key = (threshold, trigger, top_k, hold_days, fee, slip)
                        position_cache[key] = positions
                        wf_cache[key] = walk_forward_table(positions, ret_df, fee, slip)

    summary = pd.DataFrame(rows).sort_values(
        ["test_sharpe", "test_calmar", "test_corr_vs_combo"],
        ascending=[False, False, True],
    )

    if not args.no_export:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        summary.to_csv(OUTPUT_ROOT / "summary.csv", index=False)
        best = summary.iloc[0]
        key = (
            float(best["threshold"]),
            str(best["trigger"]),
            int(best["top_k"]),
            int(best["hold_days"]),
            float(best["fee"]),
            float(best["slip"]),
        )
        wf_cache[key].to_csv(OUTPUT_ROOT / "walk_forward.csv", index=False)
        report_lines = [
            "# Shock Reversal v0 Report",
            "",
            "## Setup",
            f"- assets: {discovered_assets()}",
            f"- thresholds: {STRESS_THRESHOLDS}",
            f"- triggers: {TRIGGERS}",
            f"- top_k: {TOP_KS}",
            f"- hold_days: {HOLD_DAYS}",
            f"- cost_scenarios: {COSTS}",
            "",
            "## Best Candidate",
            f"- threshold: {best['threshold']:.2f}",
            f"- trigger: {best['trigger']}",
            f"- top_k: {int(best['top_k'])}",
            f"- hold_days: {int(best['hold_days'])}",
            f"- fee/slip: {best['fee']:.4f}/{best['slip']:.4f}",
            f"- test_sharpe: {best['test_sharpe']:.4f}",
            f"- test_calmar: {best['test_calmar']:.4f}",
            f"- max_drawdown: {best['max_drawdown']:.4f}",
            f"- avg_turnover: {best['avg_turnover']:.4f}",
            f"- test_corr_vs_combo: {best['test_corr_vs_combo']:.4f}",
            "",
            "## Gate Check",
            f"- go_test_sharpe_gt_0_50: {bool(best['test_sharpe'] > 0.50)}",
            f"- go_corr_lt_0_50: {bool(best['test_corr_vs_combo'] < 0.50)}",
            f"- go_both: {bool((best['test_sharpe'] > 0.50) and (best['test_corr_vs_combo'] < 0.50))}",
        ]
        (OUTPUT_ROOT / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    best = summary.iloc[0]
    print(
        f"best threshold={best['threshold']:.2f} trigger={best['trigger']} top_k={int(best['top_k'])} "
        f"hold={int(best['hold_days'])} test_sharpe={best['test_sharpe']:.4f} "
        f"test_corr_vs_combo={best['test_corr_vs_combo']:.4f}"
    )


if __name__ == "__main__":
    main()
