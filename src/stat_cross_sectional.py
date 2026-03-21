"""Minimal cross-sectional multi-asset ranking backtest."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from src.research import perf_slice, perf_stats
from src.stat_system_backtest import calmar_ratio, tail_ratio
from src.research_v1 import ALT_DATA_ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MARKET_DIR = PROJECT_ROOT / "data" / "market"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "cross_sectional" / "v0"
ASSETS = ("BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT")
LOOKBACKS = (7, 14, 21, 28)
TOP_K = (1, 2)
REBALANCE_DAYS = (1, 7)
COSTS = ((0.0010, 0.0010), (0.0010, 0.0020))
RANKING_METHODS = ("raw_momentum", "risk_adjusted")
WEIGHTING_METHODS = ("equal", "inv_vol")
TRAIN_END = "2023-12-31"
LIQUIDITY_THRESHOLDS = (1.0e7, 5.0e7, 1.0e8)
UNIVERSE_SIZES = (3, 5, 7, 9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cross-sectional Top-K ranking baseline.")
    parser.add_argument("--train-end", default=TRAIN_END)
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


@lru_cache(maxsize=None)
def load_asset_close(asset: str) -> pd.Series:
    path = MARKET_DIR / f"binance_spot_daily_{asset}USDT.csv"
    if path.exists() and path.stat().st_size > 0:
        try:
            df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
            dt = pd.to_datetime(df["date"], utc=True, errors="coerce").dt.tz_localize(None)
            return pd.Series(df["close"].to_numpy(), index=dt, name=asset)
        except EmptyDataError:
            pass
    feather_path = ALT_DATA_ROOT / f"{asset}_USDT-1d.feather"
    if feather_path.exists():
        df = pd.read_feather(feather_path)
        cols = {c.lower(): c for c in df.columns}
        date_col = cols.get("date") or cols.get("timestamp") or df.columns[0]
        close_col = cols.get("close")
        if close_col is None:
            raise KeyError(f"Missing close column in {feather_path}")
        dt = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.tz_localize(None)
        return pd.Series(pd.to_numeric(df[close_col], errors="coerce").to_numpy(), index=dt, name=asset)
    raise FileNotFoundError(f"No local market source found for {asset}")


@lru_cache(maxsize=None)
def asset_liquidity(asset: str) -> float:
    path = MARKET_DIR / f"binance_spot_daily_{asset}USDT.csv"
    if path.exists() and path.stat().st_size > 0:
        try:
            df = pd.read_csv(path)
        except EmptyDataError:
            df = pd.DataFrame()
        if {"close", "volume"}.issubset(df.columns) and len(df) >= 60:
            quote_volume = pd.to_numeric(df["close"], errors="coerce") * pd.to_numeric(df["volume"], errors="coerce")
            return float(quote_volume.tail(60).median())
    feather_path = ALT_DATA_ROOT / f"{asset}_USDT-1d.feather"
    if feather_path.exists():
        df = pd.read_feather(feather_path)
        cols = {c.lower(): c for c in df.columns}
        close_col = cols.get("close")
        vol_col = cols.get("volume")
        if close_col and vol_col and len(df) >= 60:
            quote_volume = pd.to_numeric(df[close_col], errors="coerce") * pd.to_numeric(df[vol_col], errors="coerce")
            return float(quote_volume.tail(60).median())
    return float("nan")


@lru_cache(maxsize=1)
def discover_assets() -> tuple[str, ...]:
    assets: list[tuple[str, float]] = []
    for asset in ASSETS:
        try:
            series = load_asset_close(asset)
        except Exception:
            continue
        liq = asset_liquidity(asset)
        if len(series.dropna()) >= 500 and np.isfinite(liq):
            assets.append((asset, liq))
    assets.sort(key=lambda x: x[1], reverse=True)
    return tuple(asset for asset, _ in assets)


@lru_cache(maxsize=None)
def select_universe(liquidity_threshold: float, universe_size: int) -> tuple[str, ...]:
    ranked = []
    for asset in discover_assets():
        liq = asset_liquidity(asset)
        if np.isfinite(liq) and liq >= liquidity_threshold:
            ranked.append((asset, liq))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return tuple(asset for asset, _ in ranked[:universe_size])


@lru_cache(maxsize=None)
def build_panel(assets: tuple[str, ...]) -> pd.DataFrame:
    closes: dict[str, pd.Series] = {}
    for asset in assets:
        try:
            closes[asset] = load_asset_close(asset)
        except FileNotFoundError:
            continue
    if not closes:
        raise FileNotFoundError("No local market source found for configured assets")
    close_df = pd.DataFrame(closes).sort_index()
    close_df = close_df.dropna(how="all")
    ret_df = close_df.pct_change(fill_method=None)
    return close_df.join(ret_df.add_suffix("_ret"))


def build_positions(
    close_df: pd.DataFrame,
    lookback: int,
    top_k: int,
    rebalance_days: int,
    ranking_method: str,
    weighting_method: str,
) -> pd.DataFrame:
    momentum = close_df.pct_change(lookback, fill_method=None)
    ret = close_df.pct_change(fill_method=None)
    realized_vol = ret.rolling(20, min_periods=10).std() * np.sqrt(20)
    if ranking_method == "risk_adjusted":
        score_frame = momentum / realized_vol.replace(0.0, np.nan)
    else:
        score_frame = momentum
    eligible = momentum.notna()
    positions = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)
    rebalance_mask = np.arange(len(close_df)) % rebalance_days == 0
    current = pd.Series(0.0, index=close_df.columns)
    for i, dt in enumerate(close_df.index):
        if rebalance_mask[i]:
            scores = score_frame.loc[dt].where(eligible.loc[dt]).dropna().sort_values(ascending=False)
            winners = scores.head(top_k).index.tolist()
            current[:] = 0.0
            if winners:
                if weighting_method == "inv_vol":
                    vol_slice = realized_vol.loc[dt, winners].replace(0.0, np.nan)
                    inv = (1.0 / vol_slice).replace([np.inf, -np.inf], np.nan).dropna()
                    if not inv.empty and inv.sum() > 0:
                        current.loc[inv.index] = inv / inv.sum()
                    else:
                        current.loc[winners] = 1.0 / len(winners)
                else:
                    current.loc[winners] = 1.0 / len(winners)
        positions.loc[dt] = current
    return positions


def run_portfolio_backtest(positions: pd.DataFrame, close_df: pd.DataFrame, fee: float, slip: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    ret = close_df.pct_change(fill_method=None).fillna(0.0)
    turnover = positions.diff().abs().sum(axis=1).fillna(positions.abs().sum(axis=1))
    portfolio_ret = (positions.shift(1).fillna(0.0) * ret).sum(axis=1)
    pnl = portfolio_ret - (fee + slip) * turnover
    eq = (1.0 + pnl).cumprod()
    return eq, turnover, pnl


def load_btc_v2_returns() -> pd.Series:
    path = PROJECT_ROOT / "outputs" / "system_backtest" / "shadow_validation" / "btc_1d_shadow_equity.csv"
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    eq = pd.Series(df["v2_equity"].to_numpy(), index=pd.to_datetime(df["date"]))
    return eq.pct_change().fillna(0.0)


def summarize(eq: pd.Series, pnl: pd.Series, turnover: pd.Series, train_end: str, benchmark_ret: pd.Series) -> dict[str, float]:
    full = perf_stats(eq)
    test_start = (pd.Timestamp(train_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    test = perf_slice(eq, test_start, eq.index.max().strftime("%Y-%m-%d"))
    test_mask = pd.Series(eq.index > pd.Timestamp(train_end), index=eq.index)
    aligned = pnl.align(benchmark_ret, join="inner")
    corr = float(aligned[0].corr(aligned[1])) if len(aligned[0]) >= 20 else float("nan")
    if len(aligned[0]) >= 20:
        aligned_mask = test_mask.reindex(aligned[0].index, fill_value=False)
        test_corr = float(aligned[0][aligned_mask].corr(aligned[1][aligned_mask]))
        rolling_corr = aligned[0].rolling(60, min_periods=20).corr(aligned[1])
        rolling_corr_median = float(rolling_corr.median())
    else:
        test_corr = float("nan")
        rolling_corr_median = float("nan")
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
        "corr_vs_btc_v2": corr,
        "test_corr_vs_btc_v2": test_corr,
        "rolling60_corr_median": rolling_corr_median,
    }


def write_report(summary: pd.DataFrame) -> None:
    best = summary.sort_values(["test_sharpe", "test_calmar"], ascending=[False, False]).iloc[0]
    lines = [
        "# Cross-Sectional v0 Report",
        "",
        "## Setup",
        f"- available_assets: {', '.join(discover_assets())}",
        f"- lookbacks: {LOOKBACKS}",
        f"- top_k: {TOP_K}",
        f"- rebalance_days: {REBALANCE_DAYS}",
        f"- ranking_methods: {RANKING_METHODS}",
        f"- weighting_methods: {WEIGHTING_METHODS}",
        f"- liquidity_thresholds: {LIQUIDITY_THRESHOLDS}",
        f"- universe_sizes: {UNIVERSE_SIZES}",
        f"- cost_scenarios: {COSTS}",
        "",
        "## Best Candidate",
        f"- lookback: {int(best['lookback'])}",
        f"- top_k: {int(best['top_k'])}",
        f"- rebalance_days: {int(best['rebalance_days'])}",
        f"- universe_size: {int(best['universe_size'])}",
        f"- liquidity_threshold: {float(best['liquidity_threshold']):.0f}",
        f"- universe_assets: {best['universe_assets']}",
        f"- ranking_method: {best['ranking_method']}",
        f"- weighting_method: {best['weighting_method']}",
        f"- fee: {best['fee']:.4f}",
        f"- slip: {best['slip']:.4f}",
        f"- test_sharpe: {best['test_sharpe']:.4f}",
        f"- test_calmar: {best['test_calmar']:.4f}",
        f"- test_corr_vs_btc_v2: {best['test_corr_vs_btc_v2']:.4f}",
        f"- rolling60_corr_median: {best['rolling60_corr_median']:.4f}",
    ]
    (OUTPUT_ROOT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def walk_forward_table(
    close_df: pd.DataFrame,
    benchmark_ret: pd.Series,
    lookback: int,
    top_k: int,
    rebalance_days: int,
    ranking_method: str,
    weighting_method: str,
    fee: float,
    slip: float,
) -> pd.DataFrame:
    position = build_positions(
        close_df=close_df,
        lookback=lookback,
        top_k=top_k,
        rebalance_days=rebalance_days,
        ranking_method=ranking_method,
        weighting_method=weighting_method,
    )
    eq, turnover, pnl = run_portfolio_backtest(position, close_df, fee=fee, slip=slip)
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
        aligned = pnl_slice.align(benchmark_ret, join="inner")
        windows.append(
            {
                "window_start": current.strftime("%Y-%m-%d"),
                "window_end": test_end.strftime("%Y-%m-%d"),
                "sharpe": float(perf_stats(eq_slice)["sharpe"]),
                "calmar": float(calmar_ratio(perf_stats(eq_slice)["ann_return"], perf_stats(eq_slice)["max_drawdown"])),
                "avg_turnover": float(turn_slice.mean()),
                "corr_vs_btc_v2": float(aligned[0].corr(aligned[1])) if len(aligned[0]) >= 20 else float("nan"),
            }
        )
        current = current + pd.Timedelta(days=30)
    return pd.DataFrame(windows)


def main() -> None:
    args = parse_args()
    btc_v2_ret = load_btc_v2_returns()

    rows: list[dict[str, float | int]] = []
    for liquidity_threshold in LIQUIDITY_THRESHOLDS:
        for universe_size in UNIVERSE_SIZES:
            assets = select_universe(liquidity_threshold=liquidity_threshold, universe_size=universe_size)
            if len(assets) < 3:
                continue
            panel = build_panel(assets)
            close_df = panel[list(assets)]
            for lookback in LOOKBACKS:
                for top_k in TOP_K:
                    if top_k >= len(assets):
                        continue
                    for rebalance_days in REBALANCE_DAYS:
                        for ranking_method in RANKING_METHODS:
                            for weighting_method in WEIGHTING_METHODS:
                                pos = build_positions(
                                    close_df,
                                    lookback=lookback,
                                    top_k=top_k,
                                    rebalance_days=rebalance_days,
                                    ranking_method=ranking_method,
                                    weighting_method=weighting_method,
                                )
                                for fee, slip in COSTS:
                                    eq, turnover, pnl = run_portfolio_backtest(pos, close_df, fee=fee, slip=slip)
                                    row = {
                                        "liquidity_threshold": liquidity_threshold,
                                        "universe_size": len(assets),
                                        "universe_assets": ",".join(assets),
                                        "lookback": lookback,
                                        "top_k": top_k,
                                        "rebalance_days": rebalance_days,
                                        "ranking_method": ranking_method,
                                        "weighting_method": weighting_method,
                                        "fee": fee,
                                        "slip": slip,
                                    }
                                    row.update(summarize(eq, pnl, turnover, args.train_end, btc_v2_ret))
                                    rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["test_sharpe", "test_calmar"], ascending=[False, False])
    if not args.no_export:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        summary.to_csv(OUTPUT_ROOT / "summary.csv", index=False)
        write_report(summary)
        best = summary.iloc[0]
        wf = walk_forward_table(
            close_df=build_panel(tuple(str(x) for x in str(best["universe_assets"]).split(",")))[list(str(best["universe_assets"]).split(","))],
            benchmark_ret=btc_v2_ret,
            lookback=int(best["lookback"]),
            top_k=int(best["top_k"]),
            rebalance_days=int(best["rebalance_days"]),
            ranking_method=str(best["ranking_method"]),
            weighting_method=str(best["weighting_method"]),
            fee=float(best["fee"]),
            slip=float(best["slip"]),
        )
        wf.to_csv(OUTPUT_ROOT / "walk_forward.csv", index=False)
    top = summary.iloc[0]
    print(
        f"best lookback={int(top['lookback'])} top_k={int(top['top_k'])} "
        f"rebalance={int(top['rebalance_days'])} ranking={top['ranking_method']} "
        f"weighting={top['weighting_method']} test_sharpe={top['test_sharpe']:.4f} "
        f"test_corr_vs_btc_v2={top['test_corr_vs_btc_v2']:.4f}"
    )


if __name__ == "__main__":
    main()
