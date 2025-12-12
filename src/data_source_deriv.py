"""Derivative data loaders for btc_fgi_backtest."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DERIV_DIR = DATA_DIR / "derivatives"
DEFAULT_FUNDING_PATH = DERIV_DIR / "binance_funding_BTCUSDT.csv"
DEFAULT_OI_PATH = DERIV_DIR / "binance_oi_BTCUSDT.csv"


def _ensure_exists(path: Path) -> None:
    """Raise if the given file path does not exist."""
    if not path.exists():
        raise FileNotFoundError(path)


def _coerce_datetime(
    df: pd.DataFrame, candidates: Iterable[str], *, normalize: bool = True
) -> pd.Series:
    """Return the first usable datetime column among candidates, normalized to date if requested."""
    for col in candidates:
        if col in df.columns:
            series = df[col]
            ts = None
            if pd.api.types.is_numeric_dtype(series):
                ts = pd.to_datetime(series, unit="ms", errors="coerce", utc=True)
            else:
                ts = pd.to_datetime(series, errors="coerce", utc=True)
            if ts.notna().any():
                ts = ts.dt.tz_localize(None)
                return ts.dt.normalize() if normalize else ts
    raise KeyError(f"Could not find timestamp column in {list(df.columns)}")


def _coerce_numeric(df: pd.DataFrame, candidates: Iterable[str]) -> pd.Series:
    """Return the first usable numeric column among candidates (stripping % if present)."""
    for col in candidates:
        if col in df.columns:
            series = df[col]
            if series.dtype == object:
                series = series.astype(str).str.replace("%", "", regex=False)
            series = pd.to_numeric(series, errors="coerce")
            if series.notna().any():
                return series
    raise KeyError(f"Could not find numeric column among {candidates}")


def load_funding_daily(path: Path | str = DEFAULT_FUNDING_PATH) -> pd.DataFrame:
    """Aggregate intraday funding rates into daily sums; returns date-indexed funding_rate."""
    path = Path(path)
    _ensure_exists(path)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["funding_rate"]).set_index(pd.Index([], name="date"))
    df = df.copy()
    df["date"] = _coerce_datetime(
        df, ["fundingTime", "timestamp", "time", "Time", "date", "Date"]
    )
    df["fundingRate"] = _coerce_numeric(
        df, ["fundingRate", "Funding Rate", "rate", "funding_rate"]
    )
    daily = (
        df.groupby("date")["fundingRate"]
        .sum()  # sum of all funding prints in a day to capture cumulative impact
        .rename("funding_rate")
    )
    return daily.to_frame()


def load_oi_daily(path: Path | str = DEFAULT_OI_PATH) -> pd.DataFrame:
    """Aggregate intraday open interest into end-of-day OI; returns date-indexed oi."""
    path = Path(path)
    _ensure_exists(path)
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["oi"]).set_index(pd.Index([], name="date"))
    df = df.copy()
    df["timestamp"] = _coerce_datetime(
        df, ["timestamp", "time", "Time", "date", "Date", "closeTime"], normalize=False
    )
    df["date"] = df["timestamp"].dt.normalize()
    df["oi"] = _coerce_numeric(df, ["openInterest", "Open Interest", "oi", "open_interest"])
    df = df.sort_values("timestamp")
    last_per_day = df.groupby("date").tail(1).set_index("date")
    return last_per_day[["oi"]]
