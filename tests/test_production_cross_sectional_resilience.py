from __future__ import annotations

import pandas as pd

from src import production_cross_sectional


def test_build_cross_sectional_payloads_tolerates_partial_universe(monkeypatch) -> None:
    idx = pd.to_datetime(["2026-03-20", "2026-03-21"])
    panel = pd.DataFrame(
        {
            "BTC": [84000.0, 85000.0],
            "BTC_ret": [0.0, 0.0119],
        },
        index=idx,
    )
    positions = pd.DataFrame({"BTC": [0.0, 1.0]}, index=idx)
    eq = pd.Series([1.0, 1.01], index=idx)
    turnover = pd.Series([0.0, 1.0], index=idx)
    pnl = pd.Series([0.0, 0.01], index=idx)

    monkeypatch.setattr(production_cross_sectional, "build_panel", lambda _assets: panel)
    monkeypatch.setattr(
        production_cross_sectional,
        "build_positions",
        lambda **_kwargs: positions,
    )
    monkeypatch.setattr(
        production_cross_sectional,
        "run_portfolio_backtest",
        lambda *_args, **_kwargs: (eq, turnover, pnl),
    )

    _, payload = production_cross_sectional.build_cross_sectional_payloads(
        config={**production_cross_sectional.BEST_CONFIG, "universe_assets": ("BTC", "ETH")}
    )

    assert payload["selected_assets"] == ["BTC"]
