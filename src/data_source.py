"""Data source utilities for btc_fgi_backtest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_source_deriv import load_funding_daily, load_oi_daily

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
BITCOIN_PATH = DATA_DIR / "bitcoin.xlsx"
FGI_JSON_PATH = DATA_DIR / "fgi.json"


def _ensure_exists(path: Path) -> None:
    """Raise if the given file path does not exist."""
    if not path.exists():
        raise FileNotFoundError(path)


def load_price(path: Path = BITCOIN_PATH) -> pd.DataFrame:
    """Load BTC OHLCV daily data from Excel and return a date-indexed DataFrame."""
    _ensure_exists(path)
    df = pd.read_excel(path)
    lower_map = {col.lower(): col for col in df.columns}

    if "date" not in df.columns:
        if "timeclose" in lower_map:
            raw_col = lower_map["timeclose"]
            df["date"] = pd.to_datetime(df[raw_col], unit="ms", utc=True).dt.tz_localize(None)
        elif "timeopen" in lower_map:
            raw_col = lower_map["timeopen"]
            df["date"] = pd.to_datetime(df[raw_col], unit="ms", utc=True).dt.tz_localize(None)
        else:
            raise ValueError("bitcoin.xlsx must contain a 'date' or 'timeClose' column")

    if "priceClose" not in df.columns and "priceclose" in lower_map:
        df["priceClose"] = df[lower_map["priceclose"]]

    if "priceClose" not in df.columns and "close" in lower_map:
        df["priceClose"] = df[lower_map["close"]]

    if "priceClose" not in df.columns:
        raise ValueError("bitcoin.xlsx must contain a 'priceClose' column")

    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None).dt.normalize()
    df = df.sort_values("date").drop_duplicates(subset="date").set_index("date")
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df


def load_fgi(path: Path = FGI_JSON_PATH) -> pd.DataFrame:
    """Load Fear & Greed Index data from JSON and return a date-indexed DataFrame."""
    _ensure_exists(path)
    with path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    data = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame(columns=["fgi"]).set_index(pd.Index([], name="date"))
    df = df.rename(
        columns={"timestamp": "date", "value": "fgi", "value_classification": "fgi_label"}
    )
    if "date" not in df.columns or "fgi" not in df.columns:
        raise ValueError("FGI json must include 'timestamp'/'value' fields in entries")
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", dayfirst=True)
    df["fgi"] = pd.to_numeric(df["fgi"], errors="coerce")
    df = df.sort_values("date").drop_duplicates(subset="date")
    df = df.set_index("date")
    return df[["fgi", "fgi_label"]] if "fgi_label" in df.columns else df[["fgi"]]


def build_base_df() -> pd.DataFrame:
    """Return merged dataframe indexed by date with returns and buy-and-hold equity."""
    price_df = load_price()
    fgi_df = load_fgi()
    df = price_df.join(fgi_df, how="inner")
    # Join derivative metrics if available
    try:
        funding_df = load_funding_daily()
    except FileNotFoundError:
        funding_df = pd.DataFrame(index=df.index, columns=["funding_rate"])
    df = df.join(funding_df, how="left")
    try:
        oi_df = load_oi_daily()
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
