from __future__ import annotations

import pandas as pd

from src import stat_cross_sectional


def test_build_panel_skips_assets_without_local_source(monkeypatch) -> None:
    idx = pd.to_datetime(["2026-03-20", "2026-03-21"])
    btc = pd.Series([84000.0, 85000.0], index=idx, name="BTC")

    def _load(asset: str) -> pd.Series:
        if asset == "BTC":
            return btc
        raise FileNotFoundError("missing")

    monkeypatch.setattr(stat_cross_sectional, "load_asset_close", _load)
    out = stat_cross_sectional.build_panel(("BTC", "ETH"))

    assert "BTC" in out.columns
    assert "ETH" not in out.columns
