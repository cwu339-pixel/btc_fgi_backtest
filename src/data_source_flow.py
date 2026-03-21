"""Optional on-chain flow data loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ONCHAIN_DIR = DATA_DIR / "onchain"
DEFAULT_NETFLOW_PATH = ONCHAIN_DIR / "exchange_netflow_BTC.csv"
DEFAULT_ETF_FLOW_PATH = ONCHAIN_DIR / "etf_netflow_BTC.csv"
DEFAULT_STABLECOIN_FLOW_PATH = ONCHAIN_DIR / "stablecoin_liquidity_global.csv"
MARKET_DIR = DATA_DIR / "market"


def _ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


def _coerce_datetime(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for col in candidates:
        if col not in df.columns:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            ts = pd.to_datetime(series, unit="ms", errors="coerce", utc=True)
        else:
            ts = pd.to_datetime(series, errors="coerce", utc=True)
        if ts.notna().any():
            return ts.dt.tz_localize(None).dt.normalize()
    raise KeyError(f"Could not find usable timestamp column in {list(df.columns)}")


def _coerce_numeric(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    for col in candidates:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().any():
            return series
    raise KeyError(f"Could not find usable numeric column among {list(candidates)}")


def _load_daily_flow_series(
    path: Path | str,
    value_candidates: Iterable[str],
    output_col: str,
) -> pd.DataFrame:
    path = Path(path)
    _ensure_exists(path)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=[output_col]).set_index(
            pd.Index([], name="date")
        )

    df = df.copy()
    df["date"] = _coerce_datetime(df, ["date", "timestamp", "time", "Time"])
    df[output_col] = _coerce_numeric(df, value_candidates)
    df = (
        df.groupby("date", as_index=True)[output_col]
        .sum()
        .sort_index()
        .to_frame()
    )
    return df


def load_exchange_netflow_daily(
    path: Path | str = DEFAULT_NETFLOW_PATH,
) -> pd.DataFrame:
    """Load optional exchange netflow data as a daily series."""
    return _load_daily_flow_series(
        path,
        [
            "exchange_netflow",
            "netflow",
            "net_flow",
            "netFlow",
            "value",
            "amount",
        ],
        "exchange_netflow",
    )


def load_etf_netflow_daily(
    path: Path | str = DEFAULT_ETF_FLOW_PATH,
) -> pd.DataFrame:
    """Load optional ETF netflow data as a daily series."""
    return _load_daily_flow_series(
        path,
        [
            "etf_netflow",
            "netflow",
            "net_flow",
            "netFlow",
            "change",
            "flow",
            "value",
            "amount",
        ],
        "etf_netflow",
    )


def load_spot_taker_flow_daily(path: Path | str) -> pd.DataFrame:
    """Load Binance spot taker-buy flow as a daily series."""
    path = Path(path)
    _ensure_exists(path)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["spot_taker_buy_ratio"]).set_index(pd.Index([], name="date"))
    df = df.copy()
    df["date"] = _coerce_datetime(df, ["date", "timestamp", "time", "Time"])
    ratio = None
    for col in ["taker_buy_quote_ratio", "taker_buy_base_ratio", "spot_taker_buy_ratio"]:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce")
            if series.notna().any():
                ratio = series
                break
    if ratio is None:
        if {"taker_buy_quote_volume", "quote_volume"}.issubset(df.columns):
            quote = pd.to_numeric(df["quote_volume"], errors="coerce")
            taker = pd.to_numeric(df["taker_buy_quote_volume"], errors="coerce")
            derived = taker / quote.replace(0, pd.NA)
            if derived.notna().any():
                ratio = derived
        elif {"taker_buy_base_volume", "volume"}.issubset(df.columns):
            base = pd.to_numeric(df["volume"], errors="coerce")
            taker = pd.to_numeric(df["taker_buy_base_volume"], errors="coerce")
            derived = taker / base.replace(0, pd.NA)
            if derived.notna().any():
                ratio = derived
    if ratio is None:
        return pd.DataFrame(columns=["spot_taker_buy_ratio"]).set_index(pd.Index([], name="date"))
    df["spot_taker_buy_ratio"] = ratio
    return (
        df.groupby("date", as_index=True)["spot_taker_buy_ratio"]
        .mean()
        .sort_index()
        .to_frame()
    )


def load_premium_index_daily(path: Path | str) -> pd.DataFrame:
    """Load Binance premium index klines as a daily basis proxy."""
    path = Path(path)
    _ensure_exists(path)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["premium_index_close"]).set_index(pd.Index([], name="date"))
    df = df.copy()
    df["date"] = _coerce_datetime(df, ["date", "open_time", "timestamp", "time", "Time"])
    df["premium_index_close"] = _coerce_numeric(
        df,
        [
            "close",
            "premium_close",
            "premium_index_close",
            "basis_rate_close",
        ],
    )
    return (
        df.groupby("date", as_index=True)["premium_index_close"]
        .last()
        .sort_index()
        .to_frame()
    )


def load_stablecoin_liquidity_daily(
    path: Path | str = DEFAULT_STABLECOIN_FLOW_PATH,
) -> pd.DataFrame:
    """Load global stablecoin liquidity history as a daily series."""
    return _load_daily_flow_series(
        path,
        [
            "stablecoin_flow_usd",
            "stablecoin_total_usd",
            "value",
            "amount",
        ],
        "stablecoin_liquidity",
    )
