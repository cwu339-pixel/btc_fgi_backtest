from __future__ import annotations

from src import daily_pipeline


def test_build_combo_stablecoin_candidate_handles_missing_filter(monkeypatch) -> None:
    monkeypatch.setattr(
        daily_pipeline,
        "load_stablecoin_macro_filter",
        lambda: (_ for _ in ()).throw(FileNotFoundError("missing stablecoin data")),
    )
    combo = {
        "policy_name": "x",
        "btc_weight": 0.25,
        "cross_weight": 0.75,
        "target_weights": {"BTC": 0.75},
        "net_exposure": 0.75,
        "gross_exposure": 0.75,
        "risk_action": "none",
        "risk_reason": "",
        "risk_halt": False,
    }

    out = daily_pipeline.build_combo_stablecoin_candidate(combo, "2026-03-21")
    assert out["candidate_action"] == "same_as_combo"


def test_build_combo_vix_candidate_handles_missing_filter(monkeypatch) -> None:
    monkeypatch.setattr(
        daily_pipeline,
        "load_vix_filter",
        lambda: (_ for _ in ()).throw(FileNotFoundError("missing vix data")),
    )
    combo = {
        "policy_name": "x",
        "btc_weight": 0.25,
        "cross_weight": 0.75,
        "target_weights": {"BTC": 0.75},
        "net_exposure": 0.75,
        "gross_exposure": 0.75,
        "risk_action": "none",
        "risk_reason": "",
        "risk_halt": False,
    }

    out = daily_pipeline.build_combo_vix_candidate(combo, "2026-03-21")
    assert out["candidate_action"] == "same_as_combo"
