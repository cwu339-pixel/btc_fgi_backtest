"""Market structure regime model utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_source import FGI_JSON_PATH, load_fgi, load_price
from src.research import robust_scale
from src.research_v1 import ALT_DATA_ROOT, ResearchSpec, build_base_df_for_asset, build_targets, build_v1_features


@dataclass(frozen=True)
class MarketStructureSpec:
    asset: str = "BTC"
    frequency: str = "1d"
    stress_window: int = 60
    trend_window: int = 20
    horizons: tuple[int, ...] = (1, 3, 7)
    stress_threshold_quantile: float = 0.90
    trend_threshold_quantile: float = 0.65
    primary_feature: str = "mom_20d"
    primary_target: str = "fwd_ret_3d"


def bars_per_day(frequency: str) -> int:
    if frequency == "1d":
        return 1
    if frequency == "4h":
        return 6
    raise ValueError(f"Unsupported frequency: {frequency}")


def equivalent_periods(days: int, frequency: str) -> int:
    return days * bars_per_day(frequency)


def market_price_path(asset: str, frequency: str) -> Path | None:
    asset = asset.upper()
    if frequency == "4h":
        candidate = ALT_DATA_ROOT / f"{asset}_USDT-4h.feather"
        return candidate if candidate.exists() else None
    return None


def load_market_base_df(spec: MarketStructureSpec) -> pd.DataFrame:
    if spec.frequency == "1d":
        research_spec = ResearchSpec(asset=spec.asset, frequency=spec.frequency, horizons=spec.horizons)
        return build_base_df_for_asset(research_spec).copy()

    price_path = market_price_path(spec.asset, spec.frequency)
    if price_path is None:
        raise FileNotFoundError(f"No local {spec.frequency} price path found for {spec.asset}")
    price_df = load_price(price_path).copy()
    price_df.index = pd.to_datetime(price_df.index, utc=False).tz_localize(None)
    fgi = load_fgi(FGI_JSON_PATH).copy()
    fgi.index = pd.to_datetime(fgi.index).tz_localize(None)
    intraday_index = pd.DataFrame(index=price_df.index).reset_index().rename(columns={"index": "date"})
    daily_fgi = fgi.reset_index().rename(columns={"index": "date"}).sort_values("date")
    merged = pd.merge_asof(
        intraday_index.sort_values("date"),
        daily_fgi.sort_values("date"),
        on="date",
        direction="backward",
    ).set_index("date")
    df = price_df.join(merged, how="left")
    close_col = _find_col(df, ["close", "priceClose", "priceclose"])
    ret = pd.to_numeric(df[close_col], errors="coerce").pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["ret"] = ret
    df["eq_bh"] = (1 + ret).cumprod()
    return df


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Could not find any of {candidates}")


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(20, window // 3)).mean()
    std = series.rolling(window, min_periods=max(20, window // 3)).std()
    z = (series - mean) / std.replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan)


def build_market_structure_dataset(spec: MarketStructureSpec) -> pd.DataFrame:
    base = load_market_base_df(spec)
    if spec.frequency == "1d":
        research_spec = ResearchSpec(asset=spec.asset, frequency=spec.frequency, horizons=spec.horizons)
        features = build_v1_features(base, research_spec)
        targets = build_targets(base, spec.horizons)
        primary_feature_name = spec.primary_feature
        primary_target_name = spec.primary_target
    else:
        mom_period = equivalent_periods(20, spec.frequency)
        target_horizons = (
            equivalent_periods(1, spec.frequency),
            equivalent_periods(3, spec.frequency),
            equivalent_periods(7, spec.frequency),
        )
        features = pd.DataFrame(index=base.index)
        features["mom_20d_eq"] = robust_scale(pd.to_numeric(base["close"], errors="coerce").pct_change(mom_period))
        targets = build_targets(base, target_horizons).rename(
            columns={
                f"fwd_ret_{target_horizons[0]}d": "fwd_ret_1d_eq",
                f"fwd_ret_{target_horizons[1]}d": "fwd_ret_3d_eq",
                f"fwd_ret_{target_horizons[2]}d": "fwd_ret_7d_eq",
            }
        )
        primary_feature_name = "mom_20d_eq"
        primary_target_name = "fwd_ret_3d_eq"

    close_col = _find_col(base, ["close", "priceClose", "priceclose"])
    open_col = _find_col(base, ["open", "priceOpen", "priceopen"])
    high_col = _find_col(base, ["high", "priceHigh", "pricehigh"])
    low_col = _find_col(base, ["low", "priceLow", "pricelow"])
    volume_col = _find_col(base, ["volume", "Volume"])

    dataset = pd.DataFrame(index=base.index)
    dataset["open"] = pd.to_numeric(base[open_col], errors="coerce")
    dataset["close"] = pd.to_numeric(base[close_col], errors="coerce")
    dataset["high"] = pd.to_numeric(base[high_col], errors="coerce")
    dataset["low"] = pd.to_numeric(base[low_col], errors="coerce")
    dataset["volume"] = pd.to_numeric(base[volume_col], errors="coerce")
    dataset["ret"] = pd.to_numeric(base["ret"], errors="coerce")
    dataset["abs_ret"] = dataset["ret"].abs()
    dataset["intraday_range"] = (dataset["high"] - dataset["low"]) / dataset["close"].replace(0, np.nan)
    dataset["log_volume"] = np.log(dataset["volume"].replace(0, np.nan))

    dataset["ret_z_60d"] = rolling_zscore(dataset["abs_ret"], spec.stress_window)
    dataset["range_z_60d"] = rolling_zscore(dataset["intraday_range"], spec.stress_window)
    dataset["vol_z_60d"] = rolling_zscore(dataset["log_volume"], spec.stress_window)
    dataset["stress_score"] = (
        0.5 * dataset["ret_z_60d"] + 0.3 * dataset["range_z_60d"] + 0.2 * dataset["vol_z_60d"]
    )

    trend_ret = dataset["close"].pct_change(spec.trend_window)
    trend_vol = dataset["ret"].rolling(spec.trend_window, min_periods=max(10, spec.trend_window // 2)).std()
    dataset["trend_strength"] = trend_ret / (trend_vol * np.sqrt(spec.trend_window))
    dataset["trend_strength_abs"] = dataset["trend_strength"].abs()

    dataset = dataset.join(base[["fgi"]], how="left")
    dataset = dataset.join(features, how="left")
    dataset = dataset.join(targets, how="left")
    dataset = dataset.reset_index().rename(columns={"index": "date"})
    dataset.insert(1, "asset", spec.asset.upper())
    dataset["primary_feature_name"] = primary_feature_name
    dataset["primary_target_name"] = primary_target_name
    return dataset


def classify_regimes(dataset: pd.DataFrame, spec: MarketStructureSpec) -> tuple[pd.DataFrame, dict[str, float]]:
    out = dataset.copy()
    stress_threshold = float(out["stress_score"].quantile(spec.stress_threshold_quantile))
    trend_threshold = float(out["trend_strength_abs"].quantile(spec.trend_threshold_quantile))

    out["regime"] = "chop"
    out.loc[(out["stress_score"] >= stress_threshold) & (out["ret"] < 0), "regime"] = "stress"
    trend_mask = (out["regime"] != "stress") & (out["trend_strength_abs"] >= trend_threshold)
    out.loc[trend_mask & (out["trend_strength"] >= 0), "regime"] = "uptrend"
    out.loc[trend_mask & (out["trend_strength"] < 0), "regime"] = "downtrend"

    thresholds = {
        "stress_threshold": stress_threshold,
        "trend_threshold": trend_threshold,
    }
    return out, thresholds


def build_regime_summary(
    dataset: pd.DataFrame,
    feature: str,
    target_col: str,
) -> pd.DataFrame:
    sample = dataset[["date", "regime", "stress_score", "trend_strength", feature, target_col]].dropna()
    sample["momentum_signed_return"] = np.sign(sample[feature]) * sample[target_col]
    sample["momentum_long_only_return"] = np.where(sample[feature] > 0, sample[target_col], 0.0)
    rows: list[dict[str, float | str]] = []
    for regime, frame in sample.groupby("regime"):
        rank_ic = float(frame[feature].corr(frame[target_col], method="spearman")) if len(frame) >= 20 else float("nan")
        directional_accuracy = float((np.sign(frame[feature]) == np.sign(frame[target_col])).mean()) if len(frame) >= 20 else float("nan")
        rows.append(
            {
                "regime": regime,
                "n_obs": int(len(frame)),
                "share_of_sample": float(len(frame) / len(sample)) if len(sample) else float("nan"),
                "avg_stress_score": float(frame["stress_score"].mean()),
                "avg_trend_strength": float(frame["trend_strength"].mean()),
                "feature_mean": float(frame[feature].mean()),
                "target_mean": float(frame[target_col].mean()),
                "momentum_rank_ic": rank_ic,
                "momentum_directional_accuracy": directional_accuracy,
                "momentum_signed_return": float(frame["momentum_signed_return"].mean()),
                "momentum_long_only_return": float(frame["momentum_long_only_return"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("regime")


def build_transition_table(dataset: pd.DataFrame) -> pd.DataFrame:
    sample = dataset[["date", "regime"]].dropna().copy()
    if sample.empty:
        return pd.DataFrame(columns=["from_regime", "to_regime", "count", "share"])
    sample["next_regime"] = sample["regime"].shift(-1)
    transitions = sample.dropna()
    counts = (
        transitions.groupby(["regime", "next_regime"])
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"regime": "from_regime", "next_regime": "to_regime"})
    )
    total = counts["count"].sum()
    counts["share"] = counts["count"] / total if total else np.nan
    return counts.sort_values(["from_regime", "count"], ascending=[True, False])


def build_state_trigger_summary(dataset: pd.DataFrame) -> pd.DataFrame:
    sample = dataset[["regime", "ret_z_60d", "range_z_60d", "vol_z_60d", "stress_score", "trend_strength"]].dropna()
    if sample.empty:
        return pd.DataFrame()
    return (
        sample.groupby("regime")[["ret_z_60d", "range_z_60d", "vol_z_60d", "stress_score", "trend_strength"]]
        .mean()
        .reset_index()
        .sort_values("regime")
    )


def build_conditional_strategy_table(
    dataset: pd.DataFrame,
    feature: str,
    target_col: str,
) -> pd.DataFrame:
    sample = dataset[["regime", feature, target_col]].dropna().copy()
    if sample.empty:
        return pd.DataFrame()

    sample["base_long_only"] = np.where(sample[feature] > 0, sample[target_col], 0.0)
    sample["base_signed"] = np.sign(sample[feature]) * sample[target_col]

    rows: list[dict[str, float | str]] = []
    configs = {
        "always_on_long_only": sample["base_long_only"],
        "exclude_stress": np.where(sample["regime"] != "stress", sample["base_long_only"], 0.0),
        "uptrend_only": np.where(sample["regime"] == "uptrend", sample["base_long_only"], 0.0),
        "uptrend_plus_chop": np.where(sample["regime"].isin(["uptrend", "chop"]), sample["base_long_only"], 0.0),
        "sign_aware_ex_stress": np.where(sample["regime"] != "stress", sample["base_signed"], 0.0),
    }
    for name, pnl in configs.items():
        active = pnl != 0
        rows.append(
            {
                "strategy": name,
                "avg_return": float(np.mean(pnl)),
                "active_share": float(np.mean(active)),
                "positive_share": float(np.mean(np.asarray(pnl) > 0)),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_return", ascending=False)
