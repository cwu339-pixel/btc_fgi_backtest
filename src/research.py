"""Research utilities to construct BTC FGI strategies."""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from src.data_source import build_base_df

MOM_MIX_WEIGHTS = {
    "fac_mom_60": 0.4,
    "fac_mom_90": 0.4,
    "fac_mom_180": 0.2,
}


# ---------- Stats helpers ----------
def robust_scale(s: pd.Series, clip: float = 3.0) -> pd.Series:
    """Robust z-score style scaling clipped to [-1, 1] using median/MAD fallback."""
    x = s.to_numpy(dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        std = np.nanstd(x)
        if not np.isfinite(std) or std == 0:
            return pd.Series(0.0, index=s.index)
        z = (s - med) / (std * clip)
    else:
        z = (s - med) / (mad * clip)
    return z.clip(-1, 1)


def perf_stats(eq: pd.Series, freq: int = 365) -> Dict[str, float]:
    eq = eq.dropna()
    if len(eq) < 2:
        return {
            "total_return": np.nan,
            "ann_return": np.nan,
            "max_drawdown": np.nan,
            "vol": np.nan,
            "sharpe": np.nan,
            "n_days": 0,
        }
    rets = eq.pct_change().dropna()
    total_ret = eq.iloc[-1] / eq.iloc[0] - 1
    n_days = len(rets)
    ann_ret = (1 + total_ret) ** (freq / n_days) - 1 if n_days > 0 else np.nan
    roll_max = eq.cummax()
    drawdown = (roll_max - eq) / roll_max
    max_dd = drawdown.max()
    vol = rets.std() * np.sqrt(freq) if n_days > 0 else np.nan
    sharpe = ann_ret / vol if vol and vol > 0 else np.nan
    return {
        "total_return": total_ret,
        "ann_return": ann_ret,
        "max_drawdown": max_dd,
        "vol": vol,
        "sharpe": sharpe,
        "n_days": n_days,
    }


def perf_slice(eq: pd.Series, start: str, end: str) -> Dict[str, float]:
    if eq.empty:
        return perf_stats(eq)
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    sub = eq.loc[(eq.index >= s) & (eq.index <= e)]
    return perf_stats(sub)


def format_stats_line(name: str, stats: Dict[str, float]) -> str:
    return (
        f"{name:<9}: "
        f"总收益 {stats['total_return']*100:7.2f}% | "
        f"年化 {stats['ann_return']*100:7.2f}% | "
        f"最大回撤 {stats['max_drawdown']*100:6.2f}% | "
        f"波动 {stats['vol']*100:6.2f}% | "
        f"Sharpe {stats['sharpe']:5.2f} | "
        f"交易天数 {int(stats['n_days']):4d}"
    )


def best_strategy(
    stats_map: Dict[str, Dict[str, float]], key: str, prefer_min: bool = False
) -> tuple[str | None, float | None]:
    """Return the name/value pair with best metric by key (max by default, min if prefer_min)."""
    best_name = None
    best_val = None
    for name, stats in stats_map.items():
        val = stats.get(key)
        if val != val:
            continue
        if best_val is None:
            best_name, best_val = name, val
            continue
        if prefer_min:
            if val < best_val:
                best_name, best_val = name, val
        else:
            if val > best_val:
                best_name, best_val = name, val
    return best_name, best_val


def run_long_only(
    fac: pd.Series,
    ret: pd.Series,
    fee: float = 0.0004,
    max_leverage: float = 1.0,
) -> pd.Series:
    """Long-only strategy based on factor score."""
    score = fac.clip(-1, 1)
    pos = score.where(score > 0, 0.0).clip(0.0, max_leverage)
    turnover = pos.diff().abs().fillna(0.0)
    ret_net = pos.shift(1).fillna(0.0) * ret - fee * turnover
    eq = (1 + ret_net).cumprod()
    eq.index = fac.index
    return eq


def long_only_with_turnover(
    fac: pd.Series,
    ret: pd.Series,
    fee: float = 0.0004,
    max_leverage: float = 1.0,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Return equity curve and turnover series for long-only strategy."""
    score = fac.clip(-1, 1)
    pos = score.where(score > 0, 0.0).clip(0.0, max_leverage)
    turnover = pos.diff().abs().fillna(0.0)
    ret_net = pos.shift(1).fillna(0.0) * ret - fee * turnover
    eq = (1 + ret_net).cumprod()
    eq.index = fac.index
    turnover.index = fac.index
    pos.index = fac.index
    return eq, turnover, pos


def run_mom_mix_long_only(
    df: pd.DataFrame,
    fee: float = 0.0004,
    max_leverage: float = 1.0,
) -> pd.DataFrame:
    """Execute long-only strategy using the fac_mom_mix factor."""
    fac = df["fac_mom_mix"]
    score = fac.clip(-1, 1)
    position = score.where(score > 0, 0.0).clip(0.0, max_leverage)
    turnover = position.diff().abs().fillna(0.0)
    ret_series = df["ret"].fillna(0.0)
    ret_net = position.shift(1).fillna(0.0) * ret_series - fee * turnover
    eq = (1 + ret_net).cumprod()
    return pd.DataFrame(
        {
            "fac_mom_mix": fac,
            "position": position,
            "turnover": turnover,
            "ret_net": ret_net,
            "eq_mom_mix": eq,
        },
        index=df.index,
    )


def run_combo_mom_clean(
    ret: pd.Series,
    pos_mom_mix: pd.Series,
    pos_clean: pd.Series,
    fee: float = 0.0004,
) -> pd.DataFrame:
    """Combined strategy using MomMix offense gated by Clean signal."""

    def gate_signal(pos: pd.Series) -> pd.Series:
        gate = pd.Series(0.0, index=pos.index)
        gate = gate.mask(pos >= 0.2, 0.5)
        gate = gate.mask(pos >= 0.8, 1.0)
        gate[pos < 0.2] = 0.0
        return gate

    gate = gate_signal(pos_clean)
    final_pos = pos_mom_mix.clip(0.0, 1.0) * gate
    turnover = final_pos.diff().abs().fillna(0.0)
    ret_series = ret.fillna(0.0)
    ret_net = final_pos.shift(1).fillna(0.0) * ret_series - fee * turnover
    eq = (1 + ret_net).cumprod()
    return pd.DataFrame(
        {
            "gate": gate,
            "final_position": final_pos,
            "turnover": turnover,
            "ret_net": ret_net,
            "eq_combo": eq,
        },
        index=ret.index,
    )


def run_mom120_funding_filter(df: pd.DataFrame, fee: float = 0.0004) -> pd.Series:
    """Mom120 long-only with funding-rate crowding filter."""
    if "fac_mom_120" not in df.columns:
        raise KeyError("fac_mom_120 is required for funding filter strategy.")
    if "funding_carry_7d_rs" not in df.columns:
        raise KeyError("funding_carry_7d_rs is required for funding filter strategy.")
    mom_signal = df["fac_mom_120"]
    funding_penalty = df["funding_carry_7d_rs"].clip(lower=0).fillna(0.0)
    gate = (1.0 - funding_penalty).clip(0.0, 1.0)
    filtered_signal = mom_signal * gate
    return run_long_only(filtered_signal, df["ret"], fee=fee)


def run_mom120_oi_filter(df: pd.DataFrame, fee: float = 0.0004) -> pd.Series:
    """Mom120 long-only gated by OI crowding filter."""
    if "fac_mom_120" not in df.columns:
        raise KeyError("fac_mom_120 is required for OI filter strategy.")
    if "oi_change_7d_rs" not in df.columns or "oi_zscore_30d_rs" not in df.columns:
        raise KeyError("OI factors are required for OI filter strategy.")
    oi_filter = build_oi_filter(df["oi_change_7d_rs"], df["oi_zscore_30d_rs"])
    filtered_signal = df["fac_mom_120"] * oi_filter
    return run_long_only(filtered_signal, df["ret"], fee=fee)


def build_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Augment df with slow/fast momentum and vol-based filters."""
    df = df.copy()
    close = df["priceClose"]
    horizons = [30, 60, 90, 120, 180, 240]
    for h in horizons:
        df[f"fac_mom_{h}"] = robust_scale(close.pct_change(h))
    df["fac_mom_120"] = df["fac_mom_120"]  # ensure existing reference
    mix = sum(df[col] * weight for col, weight in MOM_MIX_WEIGHTS.items() if col in df)
    if isinstance(mix, pd.Series):
        df["fac_mom_mix"] = mix.clip(-1, 1)
    else:
        df["fac_mom_mix"] = 0.0
    df["fac_trend"] = robust_scale(np.log(close / close.rolling(200).mean()))
    df["fac_lowvol"] = robust_scale(-df["ret"].rolling(20).std())

    log_vol = np.log(df["volume"].clip(lower=1e-12))
    vol_mean = log_vol.rolling(20).mean()
    vol_std = log_vol.rolling(20).std()
    df["fac_vol"] = robust_scale((log_vol - vol_mean) / vol_std)

    df["fac_mom_60"] = robust_scale(close.pct_change(60))
    df["fac_vol_30"] = robust_scale(-df["ret"].rolling(30).std())
    return df


def build_funding_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Construct funding carry signals from funding_rate."""
    if "funding_rate" not in df.columns:
        raise KeyError("Expected 'funding_rate' column in dataframe for funding factors.")
    funding = df["funding_rate"].astype(float)
    carry_7d = funding.rolling(window=7, min_periods=3).mean()
    carry_30d = funding.rolling(window=30, min_periods=10).mean()
    factors = pd.DataFrame(
        {
            "funding_raw": funding,
            "funding_carry_7d": carry_7d,
            "funding_carry_30d": carry_30d,
        },
        index=df.index,
    )
    factors["funding_carry_7d_rs"] = robust_scale(factors["funding_carry_7d"])
    factors["funding_carry_30d_rs"] = robust_scale(factors["funding_carry_30d"])
    return factors


def print_funding_factor_summary(factors: pd.DataFrame, rows: int = 5) -> None:
    """Quick sanity print for funding factors."""
    cols = [
        "funding_raw",
        "funding_carry_7d",
        "funding_carry_30d",
        "funding_carry_7d_rs",
        "funding_carry_30d_rs",
    ]
    available = [c for c in cols if c in factors.columns]
    if not available:
        print("No funding factors to summarize.")
        return
    print("\n=== Funding factors sample ===")
    print(factors[available].tail(rows))


def build_oi_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Build open interest carry/crowding factors."""
    if "open_interest" not in df.columns:
        raise KeyError("Expected 'open_interest' column in dataframe for OI factors.")
    oi = df["open_interest"].astype(float)
    oi_change_7d = oi.pct_change(periods=7)
    oi_mean_30d = oi.rolling(window=30, min_periods=10).mean()
    oi_std_30d = oi.rolling(window=30, min_periods=10).std()
    oi_zscore_30d = (oi - oi_mean_30d) / oi_std_30d
    factors = pd.DataFrame(
        {
            "oi_raw": oi,
            "oi_change_7d": oi_change_7d,
            "oi_zscore_30d": oi_zscore_30d,
        },
        index=df.index,
    )
    factors["oi_change_7d_rs"] = robust_scale(factors["oi_change_7d"])
    factors["oi_zscore_30d_rs"] = robust_scale(factors["oi_zscore_30d"])
    return factors


def print_oi_factor_summary(oi_factors: pd.DataFrame) -> None:
    """Quick sanity print for open-interest factors."""
    cols = ["oi_raw", "oi_change_7d", "oi_zscore_30d"]
    available = [c for c in cols if c in oi_factors.columns]
    if not available:
        print("No OI factors to summarize.")
        return
    print("\n=== OI factors sample ===")
    print(oi_factors[available].head())
    print("\n=== OI factors describe ===")
    print(
        oi_factors[available]
        .describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
    )


def build_oi_filter(
    oi_change_rs: pd.Series,
    oi_zscore_rs: pd.Series,
    thr_change: float = 0.5,
    thr_zscore: float = 0.5,
) -> pd.Series:
    """Binary filter that disables longs when OI looks crowded."""
    safe_mask = (oi_change_rs <= thr_change) & (oi_zscore_rs <= thr_zscore)
    filt = safe_mask.astype(float)
    both_nan = oi_change_rs.isna() & oi_zscore_rs.isna()
    filt = filt.where(~both_nan, other=1.0)
    return filt


def analyze_oi_signal(
    df: pd.DataFrame,
    factors: pd.DataFrame,
    horizon: int = 30,
    n_bins: int = 5,
) -> None:
    """Diagnostic for OI factor predictive power."""
    if "ret" not in df.columns:
        raise KeyError("Expected 'ret' column in df for forward returns.")

    gross = (1.0 + df["ret"]).shift(-1).rolling(window=horizon).apply(
        lambda x: float(np.prod(x)), raw=True
    )
    fwd_ret = gross - 1.0
    fwd_ret = fwd_ret.reindex(df.index)

    train_mask = (df.index >= pd.Timestamp("2017-01-01")) & (
        df.index <= pd.Timestamp("2020-12-31")
    )
    test_mask = df.index >= pd.Timestamp("2021-01-01")

    oi_factor_names = ["oi_change_7d_rs", "oi_zscore_30d_rs"]
    for name in oi_factor_names:
        if name not in factors.columns:
            print(f"\n[analyze_oi_signal] Skip {name}: not in factors.")
            continue

        fac = factors[name]
        print(f"\n=== OI factor: {name} (horizon={horizon}d, bins={n_bins}) ===")

        for label, mask in [("TRAIN", train_mask), ("TEST", test_mask)]:
            fac_sub = fac[mask]
            fwd_sub = fwd_ret[mask]

            valid = fac_sub.notna() & fwd_sub.notna()
            fac_valid = fac_sub[valid]
            fwd_valid = fwd_sub[valid]

            if len(fac_valid) < n_bins * 5:
                print(f"  [{label}] not enough data (n={len(fac_valid)})")
                continue

            try:
                bins = pd.qcut(fac_valid, q=n_bins, labels=False, duplicates="drop")
            except ValueError as exc:
                print(f"  [{label}] qcut failed for {name}: {exc}")
                continue

            summary = fwd_valid.groupby(bins).mean().to_frame(name="avg_fwd_ret")
            summary["count"] = fwd_valid.groupby(bins).size()

            print(f"\n  [{label}] avg forward {horizon}d return by {name} quantile:")
            print(summary)


def blended_recent_momentum(df: pd.DataFrame, start: str = "2023-01-01") -> pd.Series:
    """Blend fast momentum + vol filter for recent regime."""
    fac = df["fac_mom_120"].copy()
    mask = fac.index >= pd.Timestamp(start)
    if mask.any():
        recent = (
            0.5 * df["fac_mom_120"][mask]
            + 0.5 * df["fac_mom_60"][mask] * (1 + df["fac_vol_30"][mask].clip(lower=-0.5)) / 1.5
        )
        fac.loc[mask] = recent
    return fac


def generate_weight_grid(step: float = 0.1) -> Iterable[Tuple[float, float, float, float]]:
    """Yield feasible (trend, mom, lowvol, vol) weight tuples that sum to 1.0 or less."""
    values = np.round(np.arange(0.0, 1.0 + 1e-9, step), 10)
    for w_trend, w_mom, w_lowv in itertools.product(values, repeat=3):
        total = w_trend + w_mom + w_lowv
        if total > 1.0:
            continue
        w_vol = 1.0 - total
        yield (w_trend, w_mom, w_lowv, w_vol)


def optimize_clean_weights(
    df: pd.DataFrame,
    step: float = 0.1,
    max_leverage_options: Iterable[float] = (1.0, 1.25, 1.5),
    eval_start: str = "2021-01-01",
) -> Tuple[Tuple[float, float, float, float], float, Dict[str, float]]:
    best_weights = (0.3, 0.4, 0.2, 0.1)
    best_cap = 1.0
    best_stats = {"sharpe": -np.inf}
    ret = df["ret"]
    mask = df.index >= pd.Timestamp(eval_start)
    for weights in generate_weight_grid(step):
        combined = sum(
            w * df[col]
            for w, col in zip(weights, ["fac_trend", "fac_mom_120", "fac_lowvol", "fac_vol"])
        )
        for cap in max_leverage_options:
            eq = run_long_only(combined, ret, max_leverage=cap)
            stats = perf_slice(eq, eval_start, df.index.max())
            if np.isnan(stats["sharpe"]):
                continue
            if stats["sharpe"] > best_stats.get("sharpe", -np.inf):
                best_weights = tuple(float(w) for w in weights)
                best_cap = cap
                best_stats = stats
    return best_weights, best_cap, best_stats


def search_mom_horizon(df: pd.DataFrame, fee: float = 0.0004) -> None:
    """Print train/test Sharpe for each momentum horizon."""
    horizons = [30, 60, 90, 120, 180, 240]
    train_start, train_end = "2017-01-01", "2020-12-31"
    test_start, test_end = "2021-01-01", df.index.max()
    train_start_ts, train_end_ts = pd.to_datetime(train_start), pd.to_datetime(train_end)
    test_start_ts, test_end_ts = pd.to_datetime(test_start), pd.to_datetime(test_end)

    rows = []
    for h in horizons:
        col = f"fac_mom_{h}"
        if col not in df.columns:
            continue
        eq, turnover, _ = long_only_with_turnover(df[col], df["ret"], fee=fee)
        stats_train = perf_slice(eq, train_start, train_end)
        stats_test = perf_slice(eq, test_start, test_end)
        turnover_train = turnover.loc[
            (turnover.index >= train_start_ts) & (turnover.index <= train_end_ts)
        ].mean()
        turnover_test = turnover.loc[
            (turnover.index >= test_start_ts) & (turnover.index <= test_end_ts)
        ].mean()
        rows.append(
            {
                "horizon": h,
                "train_sharpe": stats_train["sharpe"],
                "test_sharpe": stats_test["sharpe"],
                "train_max_dd": stats_train["max_drawdown"],
                "test_max_dd": stats_test["max_drawdown"],
                "train_vol": stats_train["vol"],
                "test_vol": stats_test["vol"],
                "train_turnover": turnover_train,
                "test_turnover": turnover_test,
            }
        )

    if not rows:
        print("No momentum horizons available for search.")
        return

    df_results = pd.DataFrame(rows)
    df_results = df_results.sort_values("test_sharpe", ascending=False)
    best_train = df_results.loc[df_results["train_sharpe"].idxmax()]
    best_test = df_results.loc[df_results["test_sharpe"].idxmax()]

    print("\n=== Momentum horizon search (train/test metrics) ===")
    print(
        df_results.to_string(
            index=False,
            float_format=lambda x: f"{x:6.2f}",
        )
    )
    print(
        f"\nBest train Sharpe horizon: {int(best_train['horizon'])}d "
        f"({best_train['train_sharpe']:.2f})"
    )
    print(
        f"Best test Sharpe horizon: {int(best_test['horizon'])}d "
        f"({best_test['test_sharpe']:.2f})"
    )

    summaries = {}
    for key in [30, 180]:
        if key in df_results["horizon"].values:
            row = df_results.loc[df_results["horizon"] == key].iloc[0]
            summaries[key] = {
                "sharpe_delta": row["test_sharpe"] - row["train_sharpe"],
                "maxdd_delta": row["test_max_dd"] - row["train_max_dd"],
                "turnover_delta": row["test_turnover"] - row["train_turnover"],
                "train_sharpe": row["train_sharpe"],
                "test_sharpe": row["test_sharpe"],
            }

    for horizon, stats in summaries.items():
        print(
            f"\nHorizon {horizon}d: "
            f"Sharpe Δ (Test-Train)={stats['sharpe_delta']:.2f}, "
            f"maxDD Δ={stats['maxdd_delta']:.2f}, "
            f"turnover Δ={stats['turnover_delta']:.4f}"
        )

    # Commentary:
    # 30D momentum reacts fastest, so it keeps strong OOS Sharpe but runs hotter turnover,
    # whereas 180D stays ultra-stable in-sample with tiny turnover but collapses in later regimes.


def walk_forward_eval(
    df: pd.DataFrame,
    clean_weights: Tuple[float, float, float, float],
    max_leverage: float,
    step_years: int = 1,
    train_years: int = 3,
    fee: float = 0.0004,
) -> None:
    """Walk-forward evaluation of BH, 120D mom, and clean combo."""
    eq_bh = df["eq_bh"]
    eq_mom120 = run_long_only(df["fac_mom_120"], df["ret"], fee=fee)
    fac_clean = sum(
        w * df[col]
        for w, col in zip(clean_weights, ["fac_trend", "fac_mom_120", "fac_lowvol", "fac_vol"])
    )
    eq_clean = run_long_only(fac_clean, df["ret"], max_leverage=max_leverage, fee=fee)

    start = pd.Timestamp("2017-01-01")
    max_date = df.index.max()
    train_offset = pd.DateOffset(years=train_years)
    test_offset = pd.DateOffset(years=step_years)
    windows = []
    while True:
        train_start = start
        train_end = (train_start + train_offset) - pd.Timedelta(days=1)
        test_start = train_end + pd.Timedelta(days=1)
        if test_start > max_date:
            break
        test_end = (test_start + test_offset) - pd.Timedelta(days=1)
        test_end = min(test_end, max_date)
        stats_bh = perf_slice(eq_bh, test_start, test_end)
        stats_mom = perf_slice(eq_mom120, test_start, test_end)
        stats_clean = perf_slice(eq_clean, test_start, test_end)
        windows.append(
            {
                "label": f"{train_start.year}-{train_end.year} -> {test_start.year}",
                "test_range": f"{test_start.date()} ~ {test_end.date()}",
                "bh": stats_bh,
                "mom": stats_mom,
                "clean": stats_clean,
            }
        )
        start += test_offset

    if not windows:
        print("Walk-forward evaluation skipped (insufficient data).")
        return

    print(
        "\n=== Walk-forward eval (train 3y / test "
        f"{step_years}y, fixed clean weights) ==="
    )
    header = (
        "Window / Test Range             "
        " | BH Sharpe | BH DD | Mom Sharpe | Mom DD | Clean Sharpe | Clean DD"
    )
    print(header)
    for row in windows:
        def fmt(stats: Dict[str, float], key: str) -> str:
            val = stats.get(key)
            return f"{val:6.2f}" if val == val else "  nan "

        print(
            f"{row['label']:>12} {row['test_range']:>23}"
            f" | {fmt(row['bh'], 'sharpe')} | {fmt(row['bh'], 'max_drawdown')}"
            f" | {fmt(row['mom'], 'sharpe')} | {fmt(row['mom'], 'max_drawdown')}"
            f" | {fmt(row['clean'], 'sharpe')} | {fmt(row['clean'], 'max_drawdown')}"
        )


def describe_stats(title: str, eq: pd.Series) -> None:
    stats = perf_stats(eq)
    print(
        f"{title:>12}: "
        f"总收益 {stats['total_return']*100:7.2f}% | "
        f"年化 {stats['ann_return']*100:7.2f}% | "
        f"回撤 {stats['max_drawdown']*100:6.2f}% | "
        f"波动 {stats['vol']*100:6.2f}% | "
        f"Sharpe {stats['sharpe']:4.2f}"
    )


def main() -> None:
    df = build_base_df()
    df = build_factors(df)
    funding_factors = build_funding_factors(df)
    df = df.join(funding_factors)
    print_funding_factor_summary(funding_factors)
    oi_factors = build_oi_factors(df)
    df = df.join(oi_factors)
    print_oi_factor_summary(oi_factors)
    analyze_oi_signal(df, df)
    eq_bh = df["eq_bh"]
    eq_mom120 = run_long_only(df["fac_mom_120"], df["ret"])
    df_mom_mix = run_mom_mix_long_only(df)
    eq_mom_mix = df_mom_mix["eq_mom_mix"]
    df["pos_mom_mix"] = df_mom_mix["position"]

    weights, cap, stats = optimize_clean_weights(df, step=0.1)
    print(
        f"Optimized clean weights (trend, mom, lowvol, vol)={weights}, "
        f"max_leverage={cap}, eval_sharpe={stats['sharpe']:.2f}"
    )
    fac_clean = sum(
        w * df[col]
        for w, col in zip(weights, ["fac_trend", "fac_mom_120", "fac_lowvol", "fac_vol"])
    )
    eq_clean, _, pos_clean = long_only_with_turnover(
        fac_clean, df["ret"], max_leverage=cap
    )
    df["pos_clean"] = pos_clean
    combo_df = run_combo_mom_clean(df["ret"], df["pos_mom_mix"], df["pos_clean"])
    eq_combo = combo_df["eq_combo"]
    eq_mom_funding = run_mom120_funding_filter(df)
    eq_mom_oi = run_mom120_oi_filter(df)

    fac_mom_recent = blended_recent_momentum(df, start="2023-01-01")
    eq_mom_recent = run_long_only(fac_mom_recent, df["ret"])

    print("\n=== Full sample: Buy&Hold vs Mom strategies ===")
    header = (
        "Strategy      "
        "| TotalRet | AnnRet | MaxDD | Vol | Sharpe | Ndays"
    )
    print(header)
    for name, eq in [
        ("Buy&Hold", eq_bh),
        ("Mom120", eq_mom120),
        ("Clean", eq_clean),
        ("MomFund", eq_mom_funding),
        ("MomOI", eq_mom_oi),
        ("MomMix", eq_mom_mix),
        ("Combo", eq_combo),
    ]:
        stats_row = perf_stats(eq)
        print(
            f"{name:>10} | "
            f"{stats_row['total_return']*100:7.2f}% | "
            f"{stats_row['ann_return']*100:7.2f}% | "
            f"{stats_row['max_drawdown']*100:6.2f}% | "
            f"{stats_row['vol']*100:6.2f}% | "
            f"{stats_row['sharpe']:6.2f} | "
            f"{int(stats_row['n_days']):5d}"
        )

    print("\n=== Full sample stats ===")
    describe_stats("Buy&Hold", eq_bh)
    describe_stats("Mom120", eq_mom120)
    describe_stats("CleanOpt", eq_clean)
    describe_stats("MomFast", eq_mom_recent)

    print("\n=== OOS 2021+ stats ===")
    for name, eq in [
        ("Buy&Hold", eq_bh),
        ("Mom120", eq_mom120),
        ("CleanOpt", eq_clean),
        ("MomFund", eq_mom_funding),
        ("MomOI", eq_mom_oi),
        ("MomMix", eq_mom_mix),
        ("Combo", eq_combo),
        ("MomFast", eq_mom_recent),
    ]:
        stats = perf_slice(eq, "2021-01-01", df.index.max())
        print(
            f"{name:>12}: "
            f"总收益 {stats['total_return']*100:7.2f}% | "
            f"年化 {stats['ann_return']*100:7.2f}% | "
            f"回撤 {stats['max_drawdown']*100:6.2f}% | "
            f"Sharpe {stats['sharpe']:4.2f}"
        )

    strategies = [
        ("BH", eq_bh),
        ("Mom120", eq_mom120),
        ("Clean", eq_clean),
        ("MomFund", eq_mom_funding),
        ("MomOI", eq_mom_oi),
        ("MomMix", eq_mom_mix),
        ("Combo", eq_combo),
    ]

    segments = [
        ("熊+震荡 2018-01~2020-12", "2018-01-01", "2020-12-31"),
        ("大牛 2020-01~2021-12", "2020-01-01", "2021-12-31"),
        ("大熊 2022", "2022-01-01", "2022-12-31"),
        ("后震荡 2023-01~至今", "2023-01-01", "2100-01-01"),
    ]
    print("\n=== Regime breakdown ===")
    for label, start, end in segments:
        print(f"\n{label} ({start} ~ {end})")
        for name, eq in strategies:
            stats = perf_slice(eq, start, end)
            print("  " + format_stats_line(name, stats))

    train_range = ("2017-01-01", "2020-12-31")
    test_range = ("2021-01-01", df.index.max().strftime("%Y-%m-%d"))
    print(
        f"\n=== Train/Test split ===\n训练段 {train_range[0]} ~ {train_range[1]}"
    )
    train_stats = {}
    for name, eq in strategies:
        stats = perf_slice(eq, train_range[0], train_range[1])
        train_stats[name] = stats
        print("  " + format_stats_line(name, stats))

    print(f"\n测试段 {test_range[0]} ~ {test_range[1]}")
    test_stats = {}
    for name, eq in strategies:
        stats = perf_slice(eq, test_range[0], test_range[1])
        test_stats[name] = stats
        print("  " + format_stats_line(name, stats))

    train_best_sharpe = best_strategy(train_stats, "sharpe")
    train_low_dd = best_strategy(train_stats, "max_drawdown", prefer_min=True)
    test_best_sharpe = best_strategy(test_stats, "sharpe")
    test_low_dd = best_strategy(test_stats, "max_drawdown", prefer_min=True)

    print("\n=== Segment summary ===")
    if train_best_sharpe[0] is not None:
        print(
            f"Train Sharpe leader: {train_best_sharpe[0]} "
            f"({train_best_sharpe[1]:.2f})"
        )
    if train_low_dd[0] is not None:
        print(
            f"Train lowest maxDD: {train_low_dd[0]} "
            f"({train_low_dd[1]:.2%})"
        )
    if test_best_sharpe[0] is not None:
        print(
            f"Test Sharpe leader: {test_best_sharpe[0]} "
            f"({test_best_sharpe[1]:.2f})"
        )
    if test_low_dd[0] is not None:
        print(
            f"Test lowest maxDD: {test_low_dd[0]} "
            f"({test_low_dd[1]:.2%})"
        )

    search_mom_horizon(df)
    walk_forward_eval(df, weights, cap)


if __name__ == "__main__":
    main()
