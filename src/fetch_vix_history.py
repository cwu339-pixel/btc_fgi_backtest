"""Fetch free VIX history from CBOE."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "macro"
VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"


def fetch_vix_history() -> pd.DataFrame:
    response = requests.get(VIX_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    df = pd.read_csv(pd.io.common.StringIO(response.text))
    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["DATE"], format="%m/%d/%Y", errors="coerce"),
            "vix_open": pd.to_numeric(df["OPEN"], errors="coerce"),
            "vix_high": pd.to_numeric(df["HIGH"], errors="coerce"),
            "vix_low": pd.to_numeric(df["LOW"], errors="coerce"),
            "vix_close": pd.to_numeric(df["CLOSE"], errors="coerce"),
        }
    ).dropna(subset=["date", "vix_close"])
    return out.sort_values("date").reset_index(drop=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "VIX_1d.csv"
    fetch_vix_history().to_csv(path, index=False)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
