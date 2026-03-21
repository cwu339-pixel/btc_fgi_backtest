"""Minimal alpha research entrypoint for BTC daily factors."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data_source import (
    BITCOIN_PATH,
    FGI_JSON_PATH,
    default_bitcoin_path,
    load_fgi,
    load_price,
)
from src.data_source_deriv import (
    DEFAULT_FUNDING_PATH,
    DEFAULT_OI_PATH,
    load_funding_daily,
    load_oi_daily,
)
from src.data_source_flow import (
    MARKET_DIR,
    load_etf_netflow_daily,
    load_exchange_netflow_daily,
    load_premium_index_daily,
    load_stablecoin_liquidity_daily,
    load_spot_taker_flow_daily,
)
from src.research import (
    build_factors,
    build_funding_factors,
    build_oi_factors,
    long_only_with_turnover,
    perf_slice,
    perf_stats,
    robust_scale,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "research_v1"
ALT_DATA_ROOT = PROJECT_ROOT.parent / "llm-btc-trader" / "user_data" / "data" / "binance"


@dataclass(frozen=True)
class ResearchSpec:
    asset: str = "BTC"
    frequency: str = "1d"
    horizons: tuple[int, ...] = (1, 3, 7)
    quantiles: int = 5
    primary_horizon: int = 3
    export: bool = True
    required_features: tuple[str, ...] = (
        "mom_20d",
        "mom_60d",
        "vol_20d",
        "funding_carry_7d_rs",
        "oi_change_7d_rs",
        "oi_zscore_30d_rs",
    )
    optional_features: tuple[str, ...] = (
        "exchange_netflow_7d_rs",
        "etf_netflow_7d_rs",
        "spot_taker_buy_7d_rs",
        "premium_basis_7d_rs",
        "stablecoin_flow_7d_rs",
        "top_trader_count_ls_7d_rs",
        "top_trader_sum_ls_7d_rs",
        "taker_ls_vol_7d_rs",
    )
    signal_v2_components: tuple[str, ...] = (
        "mom_20d",
        "mom_60d",
        "oi_change_7d_rs",
    )
    signal_v3_components: tuple[str, ...] = (
        "mom_20d",
        "mom_60d",
        "oi_change_7d_rs",
        "funding_carry_7d_rs",
    )
    train_end: str = "2023-12-31"
    walk_forward_train_years: int = 2
    walk_forward_test_years: int = 1
    kill_test_weight_step: float = 0.25
    kill_test_fees: tuple[float, ...] = (0.0004, 0.0010, 0.0020)
    momentum_rebalance_days: tuple[int, ...] = (1, 3, 7)


def default_asset_paths(asset: str) -> dict[str, Path]:
    asset = asset.upper()
    price_candidates = [
        PROJECT_ROOT / "data" / f"{asset.lower()}.xlsx",
        PROJECT_ROOT / "data" / f"{asset.lower()}.csv",
        ALT_DATA_ROOT / f"{asset}_USDT-1d.feather",
    ]
    funding_candidates = [
        PROJECT_ROOT / "data" / "derivatives" / f"binance_funding_{asset}USDT.csv",
        ALT_DATA_ROOT / f"{asset}_USDT-derivatives-1h.feather",
    ]
    oi_candidates = [
        PROJECT_ROOT / "data" / "derivatives" / f"binance_oi_{asset}USDT.csv",
        ALT_DATA_ROOT / f"{asset}_USDT-derivatives-1h.feather",
    ]
    netflow_candidates = [
        PROJECT_ROOT / "data" / "onchain" / f"exchange_netflow_{asset}.csv",
    ]
    etf_flow_candidates = [
        PROJECT_ROOT / "data" / "onchain" / f"etf_netflow_{asset}.csv",
    ]
    spot_flow_candidates = [
        MARKET_DIR / f"binance_spot_daily_{asset}USDT.csv",
    ]
    premium_candidates = [
        MARKET_DIR / f"binance_premium_daily_{asset}USDT.csv",
    ]
    stablecoin_candidates = [
        PROJECT_ROOT / "data" / "onchain" / "stablecoin_liquidity_global.csv",
    ]

    defaults = {
        "price_path": default_bitcoin_path() if asset == "BTC" else next((p for p in price_candidates if p.exists()), price_candidates[0]),
        "fgi_path": FGI_JSON_PATH,
        "funding_path": DEFAULT_FUNDING_PATH if asset == "BTC" else next((p for p in funding_candidates if p.exists()), funding_candidates[0]),
        "oi_path": DEFAULT_OI_PATH if asset == "BTC" else next((p for p in oi_candidates if p.exists()), oi_candidates[0]),
        "netflow_path": netflow_candidates[0],
        "etf_flow_path": etf_flow_candidates[0],
        "spot_flow_path": spot_flow_candidates[0],
        "premium_path": premium_candidates[0],
        "stablecoin_path": stablecoin_candidates[0],
    }
    return defaults


def build_base_df_for_asset(spec: ResearchSpec) -> pd.DataFrame:
    paths = default_asset_paths(spec.asset)
    price_df = load_price(paths["price_path"])
    fgi_df = load_fgi(paths["fgi_path"])
    df = price_df.join(fgi_df, how="left")

    try:
        funding_df = load_funding_daily(paths["funding_path"])
    except FileNotFoundError:
        funding_df = pd.DataFrame(index=df.index, columns=["funding_rate"])
    df = df.join(funding_df, how="left")

    try:
        oi_df = load_oi_daily(paths["oi_path"])
    except FileNotFoundError:
        oi_df = pd.DataFrame(index=df.index, columns=["open_interest"])
    else:
        oi_df = oi_df.rename(columns={"oi": "open_interest"})
    df = df.join(oi_df, how="left")

    close_col = None
    for candidate in ["close", "priceClose", "priceclose"]:
        if candidate in df.columns:
            close_col = df[candidate]
            break
    if close_col is None:
        raise KeyError("Price dataframe must contain close or priceClose column")

    ret = close_col.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["ret"] = ret
    df["eq_bh"] = (1 + ret).cumprod()
    return df


def build_targets(df: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    """Build simple forward-return labels."""
    out = pd.DataFrame(index=df.index)
    ret = df["ret"].fillna(0.0)
    for horizon in horizons:
        gross = (1.0 + ret).shift(-1).rolling(window=horizon).apply(np.prod, raw=True)
        out[f"fwd_ret_{horizon}d"] = gross - 1.0
    return out


def build_v1_features(df: pd.DataFrame, spec: ResearchSpec) -> pd.DataFrame:
    """Construct the first-pass feature set for daily BTC research."""
    enriched = build_factors(df.copy())
    features = pd.DataFrame(index=enriched.index)
    features["mom_20d"] = robust_scale(enriched["priceClose"].pct_change(20))
    features["mom_60d"] = enriched["fac_mom_60"]
    features["vol_20d"] = robust_scale(-enriched["ret"].rolling(20).std())

    if "funding_rate" in enriched.columns and enriched["funding_rate"].notna().any():
        funding = build_funding_factors(enriched)
        features = features.join(funding[["funding_carry_7d_rs"]], how="left")

    if "open_interest" in enriched.columns and enriched["open_interest"].notna().any():
        oi = build_oi_factors(enriched)
        features = features.join(
            oi[["oi_change_7d_rs", "oi_zscore_30d_rs"]],
            how="left",
        )
    if "top_trader_count_long_short_ratio" in enriched.columns:
        ratio = np.log(enriched["top_trader_count_long_short_ratio"].replace(0, np.nan))
        features["top_trader_count_ls_7d_rs"] = robust_scale(
            ratio.rolling(7, min_periods=3).mean()
        )
    if "top_trader_sum_long_short_ratio" in enriched.columns:
        ratio = np.log(enriched["top_trader_sum_long_short_ratio"].replace(0, np.nan))
        features["top_trader_sum_ls_7d_rs"] = robust_scale(
            ratio.rolling(7, min_periods=3).mean()
        )
    if "taker_long_short_vol_ratio" in enriched.columns:
        ratio = np.log(enriched["taker_long_short_vol_ratio"].replace(0, np.nan))
        features["taker_ls_vol_7d_rs"] = robust_scale(
            ratio.rolling(7, min_periods=3).mean()
        )

    try:
        netflow = load_exchange_netflow_daily(default_asset_paths(spec.asset)["netflow_path"])
    except FileNotFoundError:
        pass
    else:
        netflow = netflow.reindex(features.index)
        features["exchange_netflow_7d_rs"] = robust_scale(
            netflow["exchange_netflow"].rolling(7, min_periods=3).mean()
        )
    try:
        etf_flow = load_etf_netflow_daily(default_asset_paths(spec.asset)["etf_flow_path"])
    except FileNotFoundError:
        pass
    else:
        etf_flow = etf_flow.reindex(features.index)
        features["etf_netflow_7d_rs"] = robust_scale(
            etf_flow["etf_netflow"].rolling(7, min_periods=3).mean()
        )
    try:
        spot_flow = load_spot_taker_flow_daily(default_asset_paths(spec.asset)["spot_flow_path"])
    except FileNotFoundError:
        pass
    else:
        spot_flow = spot_flow.reindex(features.index)
        features["spot_taker_buy_7d_rs"] = robust_scale(
            (spot_flow["spot_taker_buy_ratio"] - 0.5).rolling(7, min_periods=3).mean()
        )
    try:
        premium = load_premium_index_daily(default_asset_paths(spec.asset)["premium_path"])
    except FileNotFoundError:
        pass
    else:
        premium = premium.reindex(features.index)
        features["premium_basis_7d_rs"] = robust_scale(
            premium["premium_index_close"].rolling(7, min_periods=3).mean()
        )
    try:
        stablecoin = load_stablecoin_liquidity_daily(
            default_asset_paths(spec.asset)["stablecoin_path"]
        )
    except FileNotFoundError:
        pass
    else:
        stablecoin = stablecoin.reindex(features.index)
        features["stablecoin_flow_7d_rs"] = robust_scale(
            stablecoin["stablecoin_liquidity"].rolling(7, min_periods=3).mean()
        )

    return features


def compute_ic_table(
    features: pd.DataFrame,
    targets: pd.DataFrame,
) -> pd.DataFrame:
    """Compute Spearman rank IC between each feature and each target."""
    rows: list[dict[str, float | str]] = []
    for feature_name in features.columns:
        feature = features[feature_name]
        for target_name in targets.columns:
            target = targets[target_name]
            valid = feature.notna() & target.notna()
            if valid.sum() < 20:
                ic = np.nan
                n_obs = int(valid.sum())
            else:
                ic = feature[valid].corr(target[valid], method="spearman")
                n_obs = int(valid.sum())
            rows.append(
                {
                    "feature": feature_name,
                    "target": target_name,
                    "spearman_ic": ic,
                    "n_obs": n_obs,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["target", "spearman_ic"],
        ascending=[True, False],
        na_position="last",
    )


def compute_quantile_return_table(
    features: pd.DataFrame,
    target: pd.Series,
    quantiles: int,
) -> pd.DataFrame:
    """Average forward returns by feature quantile for one target horizon."""
    frames: list[pd.DataFrame] = []
    for feature_name in features.columns:
        sample = pd.DataFrame({"feature": features[feature_name], "target": target}).dropna()
        if len(sample) < quantiles * 10:
            continue
        try:
            sample["bucket"] = pd.qcut(
                sample["feature"],
                q=quantiles,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            continue
        summary = sample.groupby("bucket")["target"].agg(["mean", "count"]).reset_index()
        summary.insert(0, "feature", feature_name)
        frames.append(summary.rename(columns={"mean": "avg_fwd_return"}))
    if not frames:
        return pd.DataFrame(columns=["feature", "bucket", "avg_fwd_return", "count"])
    return pd.concat(frames, ignore_index=True)


def build_composite_signal(
    features: pd.DataFrame,
    components: tuple[str, ...],
    signal_name: str,
) -> pd.Series:
    """Build a simple composite score from the selected feature components."""
    available = [col for col in components if col in features.columns]
    if not available:
        raise ValueError(f"No {signal_name} components are available in the feature set.")
    score = features[available].mean(axis=1, skipna=True)
    return robust_scale(score).rename(signal_name)


def backtest_signal(signal: pd.Series, ret: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Run a simple long-only backtest using positive composite score as exposure."""
    return long_only_with_turnover(signal, ret, fee=0.0004, max_leverage=1.0)


def backtest_signal_with_fee(
    signal: pd.Series,
    ret: pd.Series,
    fee: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Run a long-only backtest with configurable transaction cost."""
    return long_only_with_turnover(signal, ret, fee=fee, max_leverage=1.0)


def backtest_signal_rebalanced(
    signal: pd.Series,
    ret: pd.Series,
    fee: float,
    rebalance_every: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Backtest a signal while only updating target exposure every N days."""
    score = signal.clip(-1, 1)
    target = score.where(score > 0, 0.0).clip(0.0, 1.0)
    if rebalance_every <= 1:
        position = target
    else:
        mask = np.arange(len(target)) % rebalance_every == 0
        position = target.where(mask).ffill().fillna(0.0)
    turnover = position.diff().abs().fillna(0.0)
    ret_net = position.shift(1).fillna(0.0) * ret.fillna(0.0) - fee * turnover
    equity = (1 + ret_net).cumprod()
    equity.index = signal.index
    turnover.index = signal.index
    position.index = signal.index
    return equity, turnover, position


def summarize_backtests(
    base_df: pd.DataFrame,
    features: pd.DataFrame,
    extra_signals: dict[str, pd.Series],
    spec: ResearchSpec,
) -> pd.DataFrame:
    """Compare buy-and-hold, base momentum, and composite signals on the common sample."""
    compare_cols = {
        "ret": base_df["ret"],
        "eq_bh": base_df["eq_bh"],
        "mom_20d": features.get("mom_20d"),
    }
    compare_cols.update(extra_signals)
    compare_df = pd.DataFrame(compare_cols, index=base_df.index)
    required = ["mom_20d", *extra_signals.keys()]
    compare_df = compare_df.dropna(subset=required)
    eq_mom20, _, _ = backtest_signal(compare_df["mom_20d"], compare_df["ret"])
    eq_bh = compare_df["eq_bh"] / compare_df["eq_bh"].iloc[0]

    strategies = {
        "BuyHold": eq_bh,
        "Mom20": eq_mom20,
    }
    for signal_name in extra_signals:
        eq_signal, _, _ = backtest_signal(compare_df[signal_name], compare_df["ret"])
        strategies[signal_name] = eq_signal
    rows: list[dict[str, float | str]] = []
    train_end = pd.Timestamp(spec.train_end)
    split_start = max(compare_df.index.min(), train_end + pd.Timedelta(days=1))
    split_end = compare_df.index.max()
    for name, equity in strategies.items():
        if name == "BuyHold":
            trade_events = 1
            avg_exposure = 1.0
            avg_turnover = 0.0
        elif name == "Mom20":
            _, turnover, position = backtest_signal(compare_df["mom_20d"], compare_df["ret"])
            trade_events = int(position.diff().fillna(position).gt(0).sum())
            avg_exposure = float(position.mean())
            avg_turnover = float(turnover.mean())
        else:
            _, turnover, position = backtest_signal(compare_df[name], compare_df["ret"])
            trade_events = int(position.diff().fillna(position).gt(0).sum())
            avg_exposure = float(position.mean())
            avg_turnover = float(turnover.mean())
        full = perf_stats(equity)
        train = perf_slice(equity, compare_df.index.min(), train_end)
        recent = perf_slice(equity, split_start, split_end)
        rows.append(
            {
                "strategy": name,
                "trade_events": trade_events,
                "avg_exposure": avg_exposure,
                "avg_turnover": avg_turnover,
                "total_return": full["total_return"],
                "ann_return": full["ann_return"],
                "max_drawdown": full["max_drawdown"],
                "sharpe": full["sharpe"],
                "train_return": train["total_return"],
                "train_sharpe": train["sharpe"],
                "oos_return": recent["total_return"],
                "oos_sharpe": recent["sharpe"],
            }
        )
    return pd.DataFrame(rows)


def walk_forward_summary(
    base_df: pd.DataFrame,
    features: pd.DataFrame,
    signals: dict[str, pd.Series],
    spec: ResearchSpec,
) -> pd.DataFrame:
    """Evaluate strategies on rolling train/test windows."""
    compare_cols = {"ret": base_df["ret"], "mom_20d": features.get("mom_20d")}
    compare_cols.update(signals)
    df = pd.DataFrame(compare_cols, index=base_df.index).dropna()
    if df.empty:
        return pd.DataFrame(
            columns=["window", "strategy", "train_sharpe", "test_sharpe", "test_return"]
        )

    start = df.index.min()
    end = df.index.max()
    rows: list[dict[str, float | str]] = []
    train_offset = pd.DateOffset(years=spec.walk_forward_train_years)
    test_offset = pd.DateOffset(years=spec.walk_forward_test_years)
    window_start = start
    strategies = ["mom_20d", *signals.keys()]

    while True:
        train_start = window_start
        train_end = train_start + train_offset - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        if test_start > end:
            break
        test_end = min(test_start + test_offset - pd.Timedelta(days=1), end)

        for strategy in strategies:
            equity, _, _ = backtest_signal(df[strategy], df["ret"])
            train_stats = perf_slice(equity, train_start, train_end)
            test_stats = perf_slice(equity, test_start, test_end)
            rows.append(
                {
                    "window": f"{train_start.date()}->{test_end.date()}",
                    "strategy": "Mom20" if strategy == "mom_20d" else strategy,
                    "train_sharpe": train_stats["sharpe"],
                    "test_sharpe": test_stats["sharpe"],
                    "test_return": test_stats["total_return"],
                }
            )
        window_start = window_start + test_offset

    return pd.DataFrame(rows)


def generate_weight_grid(step: float) -> list[tuple[float, float, float]]:
    """Generate simple convex weights for three-component SignalV2."""
    values = np.round(np.arange(0.0, 1.0 + 1e-9, step), 10)
    combos: list[tuple[float, float, float]] = []
    for w1, w2 in itertools.product(values, repeat=2):
        w3 = 1.0 - w1 - w2
        if w3 < 0 or w3 > 1:
            continue
        combos.append((float(w1), float(w2), float(round(w3, 10))))
    return combos


def build_weighted_signal_v2(
    features: pd.DataFrame,
    weights: tuple[float, float, float],
) -> pd.Series:
    """Build a weighted version of SignalV2 for robustness testing."""
    components = ["mom_20d", "mom_60d", "oi_change_7d_rs"]
    available = [c for c in components if c in features.columns]
    if len(available) != 3:
        raise ValueError("SignalV2 weighted kill test requires all three base components.")
    score = sum(features[col] * weight for col, weight in zip(components, weights))
    return robust_scale(score).rename("signal_v2_weighted")


def run_signal_v2_kill_test(
    base_df: pd.DataFrame,
    features: pd.DataFrame,
    spec: ResearchSpec,
) -> pd.DataFrame:
    """Stress-test SignalV2 across weights and fees."""
    components = ["mom_20d", "mom_60d", "oi_change_7d_rs"]
    sample = pd.DataFrame(
        {"ret": base_df["ret"], **{name: features.get(name) for name in components}},
        index=base_df.index,
    ).dropna()
    if sample.empty:
        return pd.DataFrame(
            columns=[
                "weights",
                "fee",
                "train_sharpe",
                "oos_sharpe",
                "oos_return",
                "max_drawdown",
            ]
        )

    train_end = pd.Timestamp(spec.train_end)
    test_start = train_end + pd.Timedelta(days=1)
    rows: list[dict[str, float | str]] = []
    for weights in generate_weight_grid(spec.kill_test_weight_step):
        signal = build_weighted_signal_v2(sample, weights)
        for fee in spec.kill_test_fees:
            equity, _, _ = backtest_signal_with_fee(signal, sample["ret"], fee=fee)
            train_stats = perf_slice(equity, sample.index.min(), train_end)
            test_stats = perf_slice(equity, test_start, sample.index.max())
            rows.append(
                {
                    "weights": "/".join(f"{w:.2f}" for w in weights),
                    "fee": fee,
                    "train_sharpe": train_stats["sharpe"],
                    "oos_sharpe": test_stats["sharpe"],
                    "oos_return": test_stats["total_return"],
                    "max_drawdown": test_stats["max_drawdown"],
                }
            )
    result = pd.DataFrame(rows)
    return result.sort_values(["oos_sharpe", "oos_return"], ascending=[False, False])


def run_momentum_core_test(
    base_df: pd.DataFrame,
    features: pd.DataFrame,
    spec: ResearchSpec,
) -> pd.DataFrame:
    """Stress-test pure momentum combinations under weights, fees, and rebalance schedules."""
    sample = pd.DataFrame(
        {
            "ret": base_df["ret"],
            "mom_20d": features.get("mom_20d"),
            "mom_60d": features.get("mom_60d"),
        },
        index=base_df.index,
    ).dropna()
    if sample.empty:
        return pd.DataFrame(
            columns=[
                "weights",
                "fee",
                "rebalance_days",
                "train_sharpe",
                "oos_sharpe",
                "oos_return",
                "max_drawdown",
            ]
        )

    train_end = pd.Timestamp(spec.train_end)
    test_start = train_end + pd.Timedelta(days=1)
    weight_pairs = []
    values = np.round(np.arange(0.0, 1.0 + 1e-9, spec.kill_test_weight_step), 10)
    for w20 in values:
        w60 = 1.0 - w20
        weight_pairs.append((float(w20), float(round(w60, 10))))

    rows: list[dict[str, float | str | int]] = []
    for w20, w60 in weight_pairs:
        signal = robust_scale(sample["mom_20d"] * w20 + sample["mom_60d"] * w60)
        signal.name = "momentum_core"
        for fee in spec.kill_test_fees:
            for rebalance_days in spec.momentum_rebalance_days:
                equity, _, _ = backtest_signal_rebalanced(
                    signal,
                    sample["ret"],
                    fee=fee,
                    rebalance_every=rebalance_days,
                )
                _, turnover, position = backtest_signal_rebalanced(
                    signal,
                    sample["ret"],
                    fee=fee,
                    rebalance_every=rebalance_days,
                )
                train_stats = perf_slice(equity, sample.index.min(), train_end)
                test_stats = perf_slice(equity, test_start, sample.index.max())
                rows.append(
                    {
                        "weights": f"{w20:.2f}/{w60:.2f}",
                        "fee": fee,
                        "rebalance_days": rebalance_days,
                        "trade_events": int(position.diff().fillna(position).gt(0).sum()),
                        "avg_exposure": float(position.mean()),
                        "avg_turnover": float(turnover.mean()),
                        "train_sharpe": train_stats["sharpe"],
                        "oos_sharpe": test_stats["sharpe"],
                        "oos_return": test_stats["total_return"],
                        "max_drawdown": test_stats["max_drawdown"],
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["oos_sharpe", "oos_return"],
        ascending=[False, False],
    )


def select_momentum_candidates(
    momentum_core_table: pd.DataFrame,
    top_n: int = 2,
) -> pd.DataFrame:
    """Pick a small set of distinct momentum candidates for execution follow-up."""
    if momentum_core_table.empty:
        return pd.DataFrame(
            columns=[
                "candidate",
                "weights",
                "fee",
                "rebalance_days",
                "oos_sharpe",
                "oos_return",
                "max_drawdown",
            ]
        )

    selected_rows: list[dict[str, object]] = []
    seen_rebalance: set[int] = set()
    for _, row in momentum_core_table.iterrows():
        rebalance_days = int(row["rebalance_days"])
        if rebalance_days in seen_rebalance:
            continue
        selected_rows.append(row.to_dict())
        seen_rebalance.add(rebalance_days)
        if len(selected_rows) >= top_n:
            break

    if len(selected_rows) < top_n:
        for _, row in momentum_core_table.iterrows():
            if len(selected_rows) >= top_n:
                break
            if any(
                existing["weights"] == row["weights"]
                and int(existing["rebalance_days"]) == int(row["rebalance_days"])
                and float(existing["fee"]) == float(row["fee"])
                for existing in selected_rows
            ):
                continue
            selected_rows.append(row.to_dict())

    candidates = pd.DataFrame(selected_rows).reset_index(drop=True)
    if candidates.empty:
        return candidates
    candidates.insert(0, "candidate", [f"MomentumCore_{i+1}" for i in range(len(candidates))])
    return candidates


def print_research_header(spec: ResearchSpec) -> None:
    print("=== Research V1 ===")
    print(f"asset={spec.asset} | frequency={spec.frequency}")
    print(f"horizons={list(spec.horizons)} | quantiles={spec.quantiles}")
    print("feature families=momentum, volatility, funding, open-interest, netflow(optional)")


def print_coverage(features: pd.DataFrame, targets: pd.DataFrame) -> None:
    coverage = pd.DataFrame(
        {
            "feature_non_null": features.notna().sum(),
            "feature_coverage": features.notna().mean(),
        }
    )
    print("\n=== Feature coverage ===")
    print(coverage.to_string(float_format=lambda x: f"{x:0.2%}" if x <= 1 else f"{x:.0f}"))

    print("\n=== Target summary ===")
    print(targets.describe().T[["mean", "std", "min", "max"]].to_string(float_format=lambda x: f"{x:.4f}"))


def print_feature_availability(spec: ResearchSpec, features: pd.DataFrame) -> None:
    available = [name for name in spec.required_features if name in features.columns]
    missing_required = [name for name in spec.required_features if name not in features.columns]
    available_optional = [name for name in spec.optional_features if name in features.columns]

    print("\n=== Feature availability ===")
    print(f"required_available={available}")
    if missing_required:
        print(f"required_missing={missing_required}")
    if available_optional:
        print(f"optional_available={available_optional}")


def print_strategy_summary(summary: pd.DataFrame) -> None:
    print("\n=== Strategy comparison ===")
    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )


def print_walk_forward_summary(summary: pd.DataFrame) -> None:
    print("\n=== Walk-forward summary ===")
    if summary.empty:
        print("No walk-forward windows available.")
        return
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def print_kill_test_summary(summary: pd.DataFrame, top_n: int = 10) -> None:
    print("\n=== SignalV2 Kill Test ===")
    if summary.empty:
        print("No kill test results available.")
        return
    print(summary.head(top_n).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def print_momentum_core_summary(summary: pd.DataFrame, top_n: int = 10) -> None:
    print("\n=== Momentum Core Test ===")
    if summary.empty:
        print("No momentum core results available.")
        return
    print(summary.head(top_n).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def print_momentum_candidates(candidates: pd.DataFrame) -> None:
    print("\n=== Momentum Candidates ===")
    if candidates.empty:
        print("No momentum candidates selected.")
        return
    print(candidates.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def export_results(
    spec: ResearchSpec,
    dataset: pd.DataFrame,
    ic_table: pd.DataFrame,
    quantile_table: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    walk_forward_table: pd.DataFrame,
    kill_test_table: pd.DataFrame,
    momentum_core_table: pd.DataFrame,
    momentum_candidates: pd.DataFrame,
) -> Path:
    export_dir = OUTPUT_ROOT / spec.asset.upper() / spec.frequency
    export_dir.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(export_dir / "dataset.csv", index=True)
    ic_table.to_csv(export_dir / "ic_table.csv", index=False)
    quantile_table.to_csv(export_dir / "quantile_returns.csv", index=False)
    strategy_summary.to_csv(export_dir / "strategy_summary.csv", index=False)
    walk_forward_table.to_csv(export_dir / "walk_forward_summary.csv", index=False)
    kill_test_table.to_csv(export_dir / "signal_v2_kill_test.csv", index=False)
    momentum_core_table.to_csv(export_dir / "momentum_core_test.csv", index=False)
    momentum_candidates.to_csv(export_dir / "momentum_candidates.csv", index=False)

    top_ic_rows = (
        ic_table.dropna(subset=["spearman_ic"])
        .sort_values(["target", "spearman_ic"], ascending=[True, False])
        .groupby("target", as_index=False)
        .head(1)
    )
    top_ic_records = top_ic_rows.to_dict(orient="records")

    metadata = {
        "asset": spec.asset,
        "frequency": spec.frequency,
        "horizons": list(spec.horizons),
        "quantiles": spec.quantiles,
        "primary_horizon": spec.primary_horizon,
        "rows": int(len(dataset)),
        "columns": list(dataset.columns),
        "top_ic": top_ic_records,
        "strategy_summary": strategy_summary.to_dict(orient="records"),
        "walk_forward_rows": int(len(walk_forward_table)),
        "kill_test_rows": int(len(kill_test_table)),
        "momentum_core_rows": int(len(momentum_core_table)),
        "momentum_candidates": momentum_candidates.to_dict(orient="records"),
    }
    with (export_dir / "run_metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)

    summary_lines = [
        f"# Research Summary: {spec.asset} {spec.frequency}",
        "",
        f"- horizons: {list(spec.horizons)}",
        f"- quantiles: {spec.quantiles}",
        f"- rows: {len(dataset)}",
        "",
        "## Top IC By Horizon",
    ]
    if top_ic_records:
        for row in top_ic_records:
            summary_lines.append(
                f"- {row['target']}: {row['feature']} (Spearman IC={row['spearman_ic']:.4f}, n={row['n_obs']})"
            )
    else:
        summary_lines.append("- No valid IC results.")

    summary_lines.extend(
        [
            "",
            "## Strategy Comparison",
        ]
    )
    for row in strategy_summary.to_dict(orient="records"):
        summary_lines.append(
            f"- {row['strategy']}: total_return={row['total_return']:.4f}, "
            f"sharpe={row['sharpe']:.4f}, train_sharpe={row['train_sharpe']:.4f}, "
            f"oos_return={row['oos_return']:.4f}, "
            f"oos_sharpe={row['oos_sharpe']:.4f}"
        )

    if not walk_forward_table.empty:
        summary_lines.extend(
            [
                "",
                "## Walk-Forward Mean Test Sharpe",
            ]
        )
        wf_mean = (
            walk_forward_table.groupby("strategy", as_index=False)["test_sharpe"]
            .mean()
            .sort_values("test_sharpe", ascending=False)
        )
        for row in wf_mean.to_dict(orient="records"):
            summary_lines.append(
                f"- {row['strategy']}: mean_test_sharpe={row['test_sharpe']:.4f}"
            )

    if not kill_test_table.empty:
        summary_lines.extend(
            [
                "",
                "## Kill Test Top Configs",
            ]
        )
        for row in kill_test_table.head(5).to_dict(orient="records"):
            summary_lines.append(
                f"- weights={row['weights']}, fee={row['fee']:.4f}, "
                f"oos_sharpe={row['oos_sharpe']:.4f}, oos_return={row['oos_return']:.4f}"
            )

    if not momentum_core_table.empty:
        summary_lines.extend(
            [
                "",
                "## Momentum Core Top Configs",
            ]
        )
        for row in momentum_core_table.head(5).to_dict(orient="records"):
            summary_lines.append(
                f"- weights={row['weights']}, fee={row['fee']:.4f}, "
                f"rebalance_days={int(row['rebalance_days'])}, "
                f"oos_sharpe={row['oos_sharpe']:.4f}, oos_return={row['oos_return']:.4f}"
            )

    if not momentum_candidates.empty:
        summary_lines.extend(
            [
                "",
                "## Recommended Momentum Candidates",
            ]
        )
        for row in momentum_candidates.to_dict(orient="records"):
            summary_lines.append(
                f"- {row['candidate']}: weights={row['weights']}, fee={row['fee']:.4f}, "
                f"rebalance_days={int(row['rebalance_days'])}, "
                f"oos_sharpe={row['oos_sharpe']:.4f}, oos_return={row['oos_return']:.4f}"
            )

    summary_lines.extend(
        [
            "",
            "## Files",
            "- dataset.csv",
            "- ic_table.csv",
            "- quantile_returns.csv",
            "- strategy_summary.csv",
            "- walk_forward_summary.csv",
            "- signal_v2_kill_test.csv",
            "- momentum_core_test.csv",
            "- momentum_candidates.csv",
            "- run_metadata.json",
        ]
    )
    (export_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return export_dir


def run_research(spec: ResearchSpec | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spec = spec or ResearchSpec()
    base_df = build_base_df_for_asset(spec)
    features = build_v1_features(base_df, spec)
    targets = build_targets(base_df, spec.horizons)

    print_research_header(spec)
    print_feature_availability(spec, features)
    print_coverage(features, targets)

    ic_table = compute_ic_table(features, targets)
    print("\n=== IC table ===")
    print(ic_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    primary_target = targets[f"fwd_ret_{spec.primary_horizon}d"]
    quantile_table = compute_quantile_return_table(
        features,
        primary_target,
        quantiles=spec.quantiles,
    )
    print(f"\n=== Quantile returns for fwd_ret_{spec.primary_horizon}d ===")
    if quantile_table.empty:
        print("No quantile table generated.")
    else:
        print(quantile_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    signal_v2 = build_composite_signal(features, spec.signal_v2_components, "SignalV2")
    signal_v3 = build_composite_signal(features, spec.signal_v3_components, "SignalV3")
    strategy_summary = summarize_backtests(
        base_df,
        features,
        {"SignalV2": signal_v2, "SignalV3": signal_v3},
        spec,
    )
    print_strategy_summary(strategy_summary)
    walk_forward_table = walk_forward_summary(
        base_df,
        features,
        {"SignalV2": signal_v2, "SignalV3": signal_v3},
        spec,
    )
    print_walk_forward_summary(walk_forward_table)
    kill_test_table = run_signal_v2_kill_test(base_df, features, spec)
    print_kill_test_summary(kill_test_table)
    momentum_core_table = run_momentum_core_test(base_df, features, spec)
    print_momentum_core_summary(momentum_core_table)
    momentum_candidates = select_momentum_candidates(momentum_core_table)
    print_momentum_candidates(momentum_candidates)

    dataset = base_df.join(features).join(signal_v2).join(signal_v3).join(targets)
    if spec.export:
        export_dir = export_results(
            spec,
            dataset,
            ic_table,
            quantile_table,
            strategy_summary,
            walk_forward_table,
            kill_test_table,
            momentum_core_table,
            momentum_candidates,
        )
        print(f"\n=== Exported results ===\n{export_dir}")
    return dataset, ic_table, quantile_table


def parse_args() -> ResearchSpec:
    parser = argparse.ArgumentParser(description="Run minimal crypto alpha research.")
    parser.add_argument("--asset", default="BTC", help="Asset ticker, e.g. BTC or ETH.")
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Do not write output files under outputs/research_v1.",
    )
    args = parser.parse_args()
    return ResearchSpec(asset=args.asset.upper(), export=not args.no_export)


def main() -> None:
    spec = parse_args()
    try:
        run_research(spec)
    except FileNotFoundError as exc:
        missing = Path(str(exc))
        print(f"Missing required local data file for asset {spec.asset}: {missing}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
