from __future__ import annotations

import pandas as pd

from src import daily_pipeline


def test_run_shadow_reports_continues_when_optional_reports_missing(monkeypatch) -> None:
    signal_payload = {
        "date": "2026-03-21",
        "regime": "chop",
        "momentum_score": 0.1,
        "flow_alignment": 0,
        "flow_signal": 0.0,
        "v1_target_exposure": 0.0,
        "v2_target_exposure": 0.0,
        "recovery_flag": 0,
        "recovery_hold_flag": 0,
        "v2_reason": "same_as_v1",
    }
    cross_payload = {
        "universe_assets": ["BTC"],
        "selected_assets": ["BTC"],
        "selected_weights": {"BTC": 1.0},
        "portfolio_turnover": 0.0,
    }
    shock_payload = {
        "threshold": 0.0,
        "trigger": 0.0,
        "shock_score": 0.0,
        "shock_flag": 0,
        "reclaim_flag": 0,
        "target_exposure": 0.0,
        "hold_days_remaining": 0,
        "reason": "inactive",
        "selected_assets": [],
        "selected_weights": {},
    }

    monkeypatch.setattr(
        daily_pipeline.production_signal,
        "build_signal_payloads",
        lambda: (pd.DataFrame(), signal_payload),
    )
    monkeypatch.setattr(
        daily_pipeline.production_cross_sectional,
        "build_cross_sectional_payloads",
        lambda **_kwargs: (pd.DataFrame(), cross_payload),
    )
    monkeypatch.setattr(
        daily_pipeline.production_shock_reversal,
        "build_shock_payloads",
        lambda: (pd.DataFrame(), shock_payload),
    )

    monkeypatch.setattr(daily_pipeline.production_signal, "main", lambda: None)
    monkeypatch.setattr(daily_pipeline.production_cross_sectional, "main", lambda: None)
    monkeypatch.setattr(daily_pipeline.production_shock_reversal, "main", lambda: None)

    def _optional_missing() -> None:
        raise FileNotFoundError("optional input missing")

    monkeypatch.setattr(daily_pipeline.stat_stablecoin_macro_hint, "main", _optional_missing)
    monkeypatch.setattr(daily_pipeline.stat_vix_hint, "main", lambda: None)
    monkeypatch.setattr(daily_pipeline.stat_shadow_divergence, "main", lambda: None)
    monkeypatch.setattr(daily_pipeline.stat_shadow_dashboard, "main", lambda: None)
    monkeypatch.setattr(daily_pipeline.stat_cross_sectional_shadow, "main", lambda: None)
    monkeypatch.setattr(daily_pipeline.stat_stablecoin_shadow, "main", lambda: None)
    monkeypatch.setattr(daily_pipeline.stat_stablecoin_shadow_gate, "main", lambda: None)
    monkeypatch.setattr(daily_pipeline.stat_shock_shadow, "main", lambda: None)
    monkeypatch.setattr(daily_pipeline.stat_risk_budget_sheet, "main", lambda: None)
    monkeypatch.setattr(daily_pipeline.stat_capacity_sheet, "main", lambda: None)
    monkeypatch.setattr(daily_pipeline.stat_weekly_attribution_sheet, "main", lambda: None)

    result = daily_pipeline.run_shadow_reports()
    assert len(result) == 7
