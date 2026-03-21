from __future__ import annotations

import pandas as pd

from src import research_v1


def test_build_base_df_for_asset_allows_missing_fgi(monkeypatch) -> None:
    idx = pd.to_datetime(["2026-03-19", "2026-03-20"])
    price_df = pd.DataFrame({"priceClose": [84000.0, 85000.0]}, index=idx)

    monkeypatch.setattr(research_v1, "default_asset_paths", lambda _asset: {"price_path": "unused", "fgi_path": "missing", "funding_path": "missing", "oi_path": "missing"})
    monkeypatch.setattr(research_v1, "load_price", lambda _path: price_df)

    def _raise_missing(_path):
        raise FileNotFoundError("fgi missing")

    monkeypatch.setattr(research_v1, "load_fgi", _raise_missing)
    monkeypatch.setattr(research_v1, "load_funding_daily", lambda _path: pd.DataFrame(index=idx, columns=["funding_rate"]))
    monkeypatch.setattr(research_v1, "load_oi_daily", lambda _path: pd.DataFrame(index=idx, columns=["oi"]))

    out = research_v1.build_base_df_for_asset(research_v1.ResearchSpec(asset="BTC"))
    assert "ret" in out.columns
    assert "eq_bh" in out.columns
