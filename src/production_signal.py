"""Generate daily production signal logs for BTC 1d Production Candidate v1 and v2."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.market_structure import MarketStructureSpec, build_market_structure_dataset, classify_regimes
from src.stat_system_backtest import build_model2_position
from src.stat_overlay_ranking import build_overlay_positions, load_flow_series


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "production_signals"


def build_signal_payloads() -> tuple[pd.DataFrame, dict]:
    spec = MarketStructureSpec(asset="BTC", frequency="1d")
    dataset = build_market_structure_dataset(spec)
    classified, thresholds = classify_regimes(dataset, spec)
    feature = classified["primary_feature_name"].dropna().iloc[0]

    idx = pd.to_datetime(classified["date"])
    score = pd.Series(classified[feature].to_numpy(), index=idx)
    regime = pd.Series(classified["regime"].to_numpy(), index=idx)
    close_px = pd.Series(classified["close"].to_numpy(), index=idx)
    flow_signal = load_flow_series().reindex(idx)
    prev_regime = regime.shift(1)
    recovery_start_flag = ((prev_regime == "stress") & (regime == "uptrend")).fillna(False)
    recovery_hold_flag = (
        recovery_start_flag.astype(int).rolling(window=3, min_periods=1).max().fillna(0).astype(bool)
    )

    v1_target = build_model2_position(
        score,
        regime,
        uptrend_weight=1.0,
        chop_weight=0.0,
        downtrend_weight=0.0,
        stress_weight=0.0,
    )
    overlay_positions = build_overlay_positions(score, regime, flow_signal)
    v2_target = overlay_positions["RecoveryC_plus_FlowB"]

    v1_active = (v1_target > 0).astype(int)
    v2_active = (v2_target > 0).astype(int)
    target_delta = v2_target - v1_target
    flow_aligned = (flow_signal > 0).fillna(False)
    v2_reason = pd.Series("same_as_v1", index=idx, dtype="object")
    divergent_mask = target_delta.abs() > 1e-12
    v2_reason.loc[divergent_mask & (regime == "chop") & recovery_hold_flag] = "recovery_decay_hold_in_chop"
    v2_reason.loc[divergent_mask & (regime == "uptrend") & recovery_hold_flag & ~recovery_start_flag & flow_aligned] = "recovery_decay_after_stress"
    v2_reason.loc[divergent_mask & (regime == "uptrend") & recovery_hold_flag & ~recovery_start_flag & ~flow_aligned] = "recovery_decay_plus_flow_scale"
    v2_reason.loc[divergent_mask & recovery_start_flag & flow_aligned] = "stress_recovery_boost"
    v2_reason.loc[divergent_mask & recovery_start_flag & ~flow_aligned] = "stress_recovery_plus_flow_scale"
    v2_reason.loc[divergent_mask & (regime == "uptrend") & ~recovery_hold_flag & ~flow_aligned] = "flow_misaligned_scale_down"

    signal_log = pd.DataFrame(
        {
            "date": idx,
            "asset": "BTC",
            "frequency": "1d",
            "regime": regime.to_numpy(),
            "momentum_score": score.to_numpy(),
            "recovery_flag": recovery_start_flag.astype(int).to_numpy(),
            "recovery_hold_flag": recovery_hold_flag.astype(int).to_numpy(),
            "flow_alignment": (flow_signal > 0).fillna(False).astype(int).to_numpy(),
            "flow_signal": flow_signal.to_numpy(),
            "v1_target_exposure": v1_target.to_numpy(),
            "v1_signal_active": v1_active.to_numpy(),
            "v2_target_exposure": v2_target.to_numpy(),
            "v2_signal_active": v2_active.to_numpy(),
            "target_exposure_delta": target_delta.to_numpy(),
            "v2_reason": v2_reason.to_numpy(),
            "close": close_px.to_numpy(),
        }
    ).sort_values("date")

    latest = signal_log.iloc[-1].copy()
    latest["v1_next_session_target_exposure"] = latest["v1_target_exposure"]
    latest["v2_next_session_target_exposure"] = latest["v2_target_exposure"]

    latest_payload = {
        "date": pd.Timestamp(latest["date"]).strftime("%Y-%m-%d"),
        "asset": latest["asset"],
        "frequency": latest["frequency"],
        "regime": latest["regime"],
        "momentum_score": float(latest["momentum_score"]),
        "recovery_flag": int(latest["recovery_flag"]),
        "recovery_hold_flag": int(latest["recovery_hold_flag"]),
        "flow_alignment": int(latest["flow_alignment"]),
        "flow_signal": float(latest["flow_signal"]) if pd.notna(latest["flow_signal"]) else None,
        "v1_target_exposure": float(latest["v1_target_exposure"]),
        "v1_signal_active": int(latest["v1_signal_active"]),
        "v2_target_exposure": float(latest["v2_target_exposure"]),
        "v2_signal_active": int(latest["v2_signal_active"]),
        "target_exposure_delta": float(latest["target_exposure_delta"]),
        "v2_reason": latest["v2_reason"],
        "close": float(latest["close"]),
        "v1_next_session_target_exposure": float(latest["v1_next_session_target_exposure"]),
        "v2_next_session_target_exposure": float(latest["v2_next_session_target_exposure"]),
        "thresholds": thresholds,
    }
    return signal_log, latest_payload


def main() -> None:
    signal_log, latest_payload = build_signal_payloads()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_csv = OUTPUT_ROOT / "btc_1d_signal_log.csv"
    latest_json = OUTPUT_ROOT / "btc_1d_signal_latest.json"
    latest_md = OUTPUT_ROOT / "btc_1d_signal_latest.md"
    compare_md = OUTPUT_ROOT / "btc_1d_signal_compare_latest.md"
    signal_log.to_csv(log_csv, index=False)
    latest_json.write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")

    latest_md.write_text(
        "\n".join(
            [
                "# BTC 1d Production Signal",
                "",
                f"- date: {latest_payload['date']}",
                f"- regime: {latest_payload['regime']}",
                f"- momentum_score: {latest_payload['momentum_score']:.4f}",
                f"- recovery_flag: {latest_payload['recovery_flag']}",
                f"- recovery_hold_flag: {latest_payload['recovery_hold_flag']}",
                f"- flow_alignment: {latest_payload['flow_alignment']}",
                f"- flow_signal: {latest_payload['flow_signal'] if latest_payload['flow_signal'] is not None else 'n/a'}",
                f"- v1_target_exposure: {latest_payload['v1_target_exposure']:.4f}",
                f"- v1_signal_active: {latest_payload['v1_signal_active']}",
                f"- v2_target_exposure: {latest_payload['v2_target_exposure']:.4f}",
                f"- v2_signal_active: {latest_payload['v2_signal_active']}",
                f"- target_exposure_delta: {latest_payload['target_exposure_delta']:.4f}",
                f"- v2_reason: {latest_payload['v2_reason']}",
                f"- close: {latest_payload['close']:.2f}",
                f"- v1_next_session_target_exposure: {latest_payload['v1_next_session_target_exposure']:.4f}",
                f"- v2_next_session_target_exposure: {latest_payload['v2_next_session_target_exposure']:.4f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    compare_md.write_text(
        "\n".join(
            [
                "# BTC 1d Shadow Comparison",
                "",
                f"- date: {latest_payload['date']}",
                f"- regime: {latest_payload['regime']}",
                f"- v1 exposure: {latest_payload['v1_target_exposure']:.4f}",
                f"- v2 exposure: {latest_payload['v2_target_exposure']:.4f}",
                f"- exposure delta: {latest_payload['target_exposure_delta']:.4f}",
                f"- recovery flag: {latest_payload['recovery_flag']}",
                f"- recovery hold flag: {latest_payload['recovery_hold_flag']}",
                f"- flow aligned: {latest_payload['flow_alignment']}",
                f"- reason: {latest_payload['v2_reason']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {log_csv}")
    print(f"Wrote {latest_json}")
    print(f"Wrote {latest_md}")
    print(f"Wrote {compare_md}")


if __name__ == "__main__":
    main()
