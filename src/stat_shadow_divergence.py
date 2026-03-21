"""Analyze when BTC 1d v1 and v2 production signals diverge."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIGNAL_ROOT = PROJECT_ROOT / "outputs" / "production_signals"
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "system_backtest" / "shadow_validation"


def _find_blocks(mask: pd.Series) -> pd.Series:
    return mask.ne(mask.shift()).cumsum()


def main() -> None:
    log_path = SIGNAL_ROOT / "btc_1d_signal_log.csv"
    df = pd.read_csv(log_path, parse_dates=["date"]).sort_values("date")
    df["diverges"] = (df["v2_target_exposure"] - df["v1_target_exposure"]).abs() > 1e-12
    df["abs_delta"] = df["target_exposure_delta"].abs()

    divergent = df[df["diverges"]].copy()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    divergent.to_csv(OUTPUT_ROOT / "btc_1d_divergence_rows.csv", index=False)

    if divergent.empty:
        (OUTPUT_ROOT / "btc_1d_divergence_report.md").write_text(
            "# BTC 1d Shadow Divergence\n\n- no divergence rows found\n",
            encoding="utf-8",
        )
        return

    divergent["block"] = _find_blocks(divergent["date"].diff().dt.days.ne(1).fillna(True))
    blocks = []
    for _, frame in divergent.groupby("block"):
        frame = frame.sort_values("date")
        blocks.append(
            {
                "start_date": frame["date"].iloc[0].date().isoformat(),
                "end_date": frame["date"].iloc[-1].date().isoformat(),
                "n_days": int(len(frame)),
                "mean_abs_delta": float(frame["abs_delta"].mean()),
                "max_abs_delta": float(frame["abs_delta"].max()),
                "dominant_reason": frame["v2_reason"].mode().iloc[0],
                "regime_set": ",".join(sorted(frame["regime"].astype(str).unique())),
            }
        )
    blocks_df = pd.DataFrame(blocks).sort_values(
        ["max_abs_delta", "n_days"],
        ascending=[False, False],
    )
    blocks_df.to_csv(OUTPUT_ROOT / "btc_1d_divergence_blocks.csv", index=False)

    reason_summary = (
        divergent.groupby("v2_reason", as_index=False)
        .agg(
            n_rows=("v2_reason", "size"),
            mean_abs_delta=("abs_delta", "mean"),
            max_abs_delta=("abs_delta", "max"),
        )
        .sort_values(["n_rows", "mean_abs_delta"], ascending=[False, False])
    )
    reason_summary.to_csv(OUTPUT_ROOT / "btc_1d_divergence_reason_summary.csv", index=False)

    latest = divergent.iloc[-1]
    top_block = blocks_df.iloc[0]
    lines = [
        "# BTC 1d Shadow Divergence",
        "",
        "## Summary",
        f"- divergent rows: {len(divergent)}",
        f"- divergence share: {len(divergent) / len(df):.4f}",
        f"- latest divergence date: {latest['date'].date().isoformat()}",
        f"- latest divergence reason: {latest['v2_reason']}",
        "",
        "## Largest Divergence Block",
        f"- start: {top_block['start_date']}",
        f"- end: {top_block['end_date']}",
        f"- n_days: {int(top_block['n_days'])}",
        f"- max_abs_delta: {top_block['max_abs_delta']:.4f}",
        f"- dominant_reason: {top_block['dominant_reason']}",
        f"- regimes: {top_block['regime_set']}",
        "",
        "## Reason Summary",
    ]
    for _, row in reason_summary.iterrows():
        lines.append(
            f"- {row['v2_reason']}: n_rows={int(row['n_rows'])}, "
            f"mean_abs_delta={row['mean_abs_delta']:.4f}, max_abs_delta={row['max_abs_delta']:.4f}"
        )

    (OUTPUT_ROOT / "btc_1d_divergence_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_ROOT / 'btc_1d_divergence_report.md'}")


if __name__ == "__main__":
    main()
