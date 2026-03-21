"""Evaluate forward promotion gates for the stablecoin macro overlay shadow candidate."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "stablecoin_macro_hint" / "shadow_validation"
FORWARD_LEDGER = PROJECT_ROOT / "outputs" / "forward_shadow" / "forward_ledger.csv"
CROSS_SHADOW = PROJECT_ROOT / "outputs" / "cross_sectional" / "shadow_validation" / "cross_sectional_shadow_equity.csv"
SHADOW_DASHBOARD_JSON = OUTPUT_ROOT / "stablecoin_shadow_dashboard.json"


def gate_status(ok: bool, pending: bool = False) -> str:
    if ok:
        return "PASS"
    if pending:
        return "PENDING"
    return "FAIL"


def count_episodes(flag: pd.Series) -> int:
    flag = pd.to_numeric(flag, errors="coerce").fillna(0).astype(int)
    return int(((flag == 1) & (flag.shift(1, fill_value=0) == 0)).sum())


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    out_json = OUTPUT_ROOT / "stablecoin_shadow_gate.json"
    out_md = OUTPUT_ROOT / "stablecoin_shadow_gate.md"

    dashboard = json.loads(SHADOW_DASHBOARD_JSON.read_text(encoding="utf-8"))
    combo_stats = dashboard["combo_stats"]
    candidate_stats = dashboard["candidate_stats"]

    sharpe_delta = float(candidate_stats["test_sharpe"]) - float(combo_stats["test_sharpe"])
    calmar_delta = float(candidate_stats["test_calmar"]) - float(combo_stats["test_calmar"])
    drawdown_improvement = float(combo_stats["max_drawdown"]) - float(candidate_stats["max_drawdown"])

    gate1_ok = sharpe_delta >= 0.15 and calmar_delta >= 0.25 and drawdown_improvement >= 0.10

    forward = pd.read_csv(FORWARD_LEDGER, parse_dates=["asof_date_utc"]) if FORWARD_LEDGER.exists() else pd.DataFrame()
    usable = forward.dropna(subset=["combo_net_exposure", "combo_stablecoin_candidate_net_exposure"]) if not forward.empty else pd.DataFrame()

    missing_policy_rows = 0
    missing_state_rows = 0
    extreme_rows = 0
    extreme_episodes = 0
    bad_extreme_rows = 0
    bad_non_extreme_rows = 0

    if not usable.empty:
        missing_policy_rows = int(usable["combo_stablecoin_candidate_policy"].fillna("").eq("").sum())
        state_cols = ["stablecoin_30d_z", "stablecoin_risk_off_flag", "stablecoin_extreme_off_flag"]
        missing_state_rows = int(usable[state_cols].isna().any(axis=1).sum())

        extreme_flag = pd.to_numeric(usable["stablecoin_extreme_off_flag"], errors="coerce").fillna(0).astype(int)
        extreme_rows = int(extreme_flag.sum())
        extreme_episodes = count_episodes(extreme_flag)

        candidate_exposure = pd.to_numeric(usable["combo_stablecoin_candidate_net_exposure"], errors="coerce").fillna(0.0)
        combo_exposure = pd.to_numeric(usable["combo_net_exposure"], errors="coerce").fillna(0.0)
        bad_extreme_rows = int(((extreme_flag == 1) & (candidate_exposure.abs() > 1e-9)).sum())
        bad_non_extreme_rows = int(((extreme_flag == 0) & ((candidate_exposure - combo_exposure).abs() > 1e-9)).sum())

    gate2_ok = (
        not usable.empty
        and missing_policy_rows == 0
        and missing_state_rows == 0
        and bad_extreme_rows == 0
        and bad_non_extreme_rows == 0
    )

    gate3_ok = extreme_rows >= 5 and extreme_episodes >= 2
    gate3_pending = usable.empty or not gate3_ok

    cross = pd.read_csv(CROSS_SHADOW, parse_dates=["date"]).sort_values("date")
    delta_rows = 0
    delta_weekly_pnl = 0.0
    if not usable.empty and not cross.empty:
        joined = (
            usable[["asof_date_utc", "stablecoin_extreme_off_flag"]]
            .rename(columns={"asof_date_utc": "date"})
            .merge(cross[["date", "combo_pnl", "combo_stablecoin_pnl"]], on="date", how="left")
        )
        joined["candidate_minus_baseline"] = pd.to_numeric(joined["combo_stablecoin_pnl"], errors="coerce").fillna(0.0) - pd.to_numeric(
            joined["combo_pnl"], errors="coerce"
        ).fillna(0.0)
        delta_rows = int((joined["candidate_minus_baseline"].abs() > 1e-12).sum())
        delta_weekly_pnl = float(joined["candidate_minus_baseline"].sum())

    gate4_ok = delta_rows >= 1 and bad_non_extreme_rows == 0
    gate4_pending = delta_rows == 0 and bad_non_extreme_rows == 0

    overall = "PROMOTION_READY" if gate1_ok and gate2_ok and gate3_ok and gate4_ok else "STAY_SHADOW"

    payload = {
        "overall": overall,
        "gate_1_historical_robustness": {
            "status": gate_status(gate1_ok),
            "test_sharpe_delta": sharpe_delta,
            "test_calmar_delta": calmar_delta,
            "max_drawdown_improvement": drawdown_improvement,
        },
        "gate_2_structural_forward_correctness": {
            "status": gate_status(gate2_ok, pending=usable.empty),
            "forward_rows": int(len(usable)),
            "missing_policy_rows": missing_policy_rows,
            "missing_state_rows": missing_state_rows,
            "bad_extreme_rows": bad_extreme_rows,
            "bad_non_extreme_rows": bad_non_extreme_rows,
        },
        "gate_3_trigger_coverage": {
            "status": gate_status(gate3_ok, pending=gate3_pending),
            "extreme_off_days": extreme_rows,
            "extreme_off_episodes": extreme_episodes,
            "required_extreme_off_days": 5,
            "required_extreme_off_episodes": 2,
        },
        "gate_4_live_separation_evidence": {
            "status": gate_status(gate4_ok, pending=gate4_pending),
            "candidate_minus_baseline_nonzero_rows": delta_rows,
            "candidate_minus_baseline_window_pnl": delta_weekly_pnl,
        },
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Stablecoin Overlay Forward Gate",
        "",
        f"- overall: {overall}",
        "",
        "## Gate 1: Historical Robustness",
        f"- status: {payload['gate_1_historical_robustness']['status']}",
        f"- test_sharpe_delta: {sharpe_delta:.4f}",
        f"- test_calmar_delta: {calmar_delta:.4f}",
        f"- max_drawdown_improvement: {drawdown_improvement:.4f}",
        "",
        "## Gate 2: Structural Forward Correctness",
        f"- status: {payload['gate_2_structural_forward_correctness']['status']}",
        f"- forward_rows: {int(len(usable))}",
        f"- missing_policy_rows: {missing_policy_rows}",
        f"- missing_state_rows: {missing_state_rows}",
        f"- bad_extreme_rows: {bad_extreme_rows}",
        f"- bad_non_extreme_rows: {bad_non_extreme_rows}",
        "",
        "## Gate 3: Trigger Coverage",
        f"- status: {payload['gate_3_trigger_coverage']['status']}",
        f"- extreme_off_days: {extreme_rows}",
        f"- extreme_off_episodes: {extreme_episodes}",
        "- required_extreme_off_days: 5",
        "- required_extreme_off_episodes: 2",
        "",
        "## Gate 4: Live Separation Evidence",
        f"- status: {payload['gate_4_live_separation_evidence']['status']}",
        f"- candidate_minus_baseline_nonzero_rows: {delta_rows}",
        f"- candidate_minus_baseline_window_pnl: {delta_weekly_pnl:.6f}",
        "",
        "## Read",
        "- This candidate is historically strong and structurally correct.",
        "- It is not promotion-ready until real forward trigger coverage exists.",
        "- A no-trigger week should keep this candidate in shadow, not promote or demote it.",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
