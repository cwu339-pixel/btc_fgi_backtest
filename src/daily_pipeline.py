"""Daily forward-shadow pipeline for BTC 1d v1/v2 signals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from src import (
    production_cross_sectional,
    production_shock_reversal,
    production_signal,
    stat_capacity_sheet,
    stat_cross_sectional_shadow,
    stat_risk_budget_sheet,
    stat_shock_shadow,
    stat_shadow_dashboard,
    stat_shadow_divergence,
    stat_stablecoin_macro_hint,
    stat_stablecoin_shadow,
    stat_stablecoin_shadow_gate,
    stat_vix_hint,
    stat_weekly_attribution_sheet,
)
from src.stat_stablecoin_macro_hint import load_stablecoin_macro_filter
from src.stat_vix_hint import load_vix_filter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MARKET_DIR = DATA_DIR / "market"
CONFIG_DIR = PROJECT_ROOT / "config"
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_SIGNALS = PROJECT_ROOT / "outputs" / "production_signals"
FORWARD_ROOT = PROJECT_ROOT / "outputs" / "forward_shadow"
SNAPSHOT_ROOT = FORWARD_ROOT / "data_snapshots"
RERUN_ROOT = FORWARD_ROOT / "reruns"
WEEKLY_ROOT = FORWARD_ROOT / "weekly_reports"
INCIDENT_LOG = FORWARD_ROOT / "incident_log.csv"
SHOCK_FORWARD_LEDGER = FORWARD_ROOT / "shock_forward_ledger.csv"
LAST_RUN_PATH = FORWARD_ROOT / "last_run.json"
OPS_DASHBOARD_MD = FORWARD_ROOT / "forward_ops_dashboard.md"
OPS_DASHBOARD_JSON = FORWARD_ROOT / "forward_ops_dashboard.json"
API_URL = "https://api.binance.com/api/v3/klines"
BTC_CANONICAL_PATH = DATA_DIR / "bitcoin.csv"
COMBO_POLICY_PATH = CONFIG_DIR / "forward_shadow_combo_policy_v1.json"
SHOCK_COMBO_POLICY_PATH = CONFIG_DIR / "shock_shadow_combo_policy_v1.json"
RISK_BUDGET_POLICY_PATH = CONFIG_DIR / "risk_budget_policy_v1.json"
CONFIG_PATHS = [
    PROJECT_ROOT / "BTC_1D_PRODUCTION_POLICY_V2.md",
    PROJECT_ROOT / "PARAMETER_FAMILY_V1.md",
    PROJECT_ROOT / "EXECUTION_ROBUSTNESS_THRESHOLDS.md",
    COMBO_POLICY_PATH,
    SHOCK_COMBO_POLICY_PATH,
    RISK_BUDGET_POLICY_PATH,
]


@dataclass(frozen=True)
class RunContext:
    run_id: str
    asof_date_utc: str
    candle_timezone: str
    symbol: str
    interval: str
    git_sha: str
    python_version: str
    host: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BTC 1d forward-shadow daily pipeline.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--time-zone", default="0")
    parser.add_argument("--skip-refresh", action="store_true")
    parser.add_argument("--lookback-bars", type=int, default=5)
    parser.add_argument("--fee-bps", type=int, default=10)
    parser.add_argument("--slip-bps", type=int, default=10)
    parser.add_argument("--latency-mode", default="next_open")
    parser.add_argument("--rerun-as-of")
    parser.add_argument("--allow-existing-asof", action="store_true")
    return parser.parse_args()


def get_git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def sha256_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _fetch_klines_batch(symbol: str, interval: str, start_ms: int, end_ms: int, time_zone: str) -> list[list[Any]]:
    query = urlencode(
        {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
            "timeZone": time_zone,
        }
    )
    request = Request(f"{API_URL}?{query}", headers={"User-Agent": "btc-fgi-backtest/1.0"})
    attempts = 0
    while attempts < 5:
        attempts += 1
        try:
            with urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            retry_after = exc.headers.get("Retry-After")
            if exc.code in {418, 429} and retry_after:
                time.sleep(float(retry_after))
                continue
            if exc.code in {418, 429}:
                time.sleep(min(2**attempts, 30))
                continue
            raise
        except URLError:
            time.sleep(min(2**attempts, 30))
    raise RuntimeError("Binance REST refresh failed after retries")


def fetch_klines_rest(symbol: str, interval: str, start_ms: int, end_ms: int, time_zone: str) -> list[list[Any]]:
    rows: list[list[Any]] = []
    current = start_ms
    while current <= end_ms:
        batch = _fetch_klines_batch(symbol, interval, current, end_ms, time_zone)
        if not batch:
            break
        rows.extend(batch)
        last_close_ms = int(batch[-1][6])
        if len(batch) < 1000 or last_close_ms >= end_ms:
            break
        current = last_close_ms + 1
    return rows


def read_existing_market_csv(path: Path) -> pd.DataFrame:
    expected_columns = [
        "date",
        "symbol",
        "open_time_ms",
        "close_time_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
        "taker_buy_base_ratio",
        "taker_buy_quote_ratio",
        "time_zone",
        "source",
    ]
    if not path.exists():
        return pd.DataFrame(columns=expected_columns)
    df = pd.read_csv(path)
    required = {"open_time_ms", "close_time_ms"}
    if required.issubset(df.columns):
        return df
    legacy_required = {"date", "open", "high", "low", "close", "volume", "quote_volume"}
    if legacy_required.issubset(df.columns):
        if BTC_CANONICAL_PATH.exists():
            btc_df = pd.read_csv(BTC_CANONICAL_PATH)
            if {"date", "timeOpen", "timeClose"}.issubset(btc_df.columns):
                merged = df.copy()
                merged["date"] = pd.to_datetime(merged["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                btc_df["date"] = pd.to_datetime(btc_df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
                merged = merged.merge(
                    btc_df[
                        [
                            "date",
                            "timeOpen",
                            "timeClose",
                            "tradeCount",
                            "takerBuyBaseVolume",
                            "takerBuyQuoteVolume",
                        ]
                    ],
                    on="date",
                    how="left",
                )
                merged["symbol"] = merged.get("symbol", "BTCUSDT").fillna("BTCUSDT")
                merged["open_time_ms"] = pd.to_numeric(merged["timeOpen"], errors="coerce")
                merged["close_time_ms"] = pd.to_numeric(merged["timeClose"], errors="coerce")
                merged["trade_count"] = pd.to_numeric(merged.get("tradeCount"), errors="coerce")
                merged["taker_buy_base_volume"] = pd.to_numeric(merged.get("takerBuyBaseVolume"), errors="coerce")
                merged["taker_buy_quote_volume"] = pd.to_numeric(merged.get("takerBuyQuoteVolume"), errors="coerce")
                merged["taker_buy_base_ratio"] = merged["taker_buy_base_volume"] / pd.to_numeric(
                    merged["volume"], errors="coerce"
                )
                merged["taker_buy_quote_ratio"] = merged["taker_buy_quote_volume"] / pd.to_numeric(
                    merged["quote_volume"], errors="coerce"
                )
                merged["time_zone"] = "UTC"
                merged["source"] = "legacy_market_csv"
                return merged[expected_columns]
        return df.assign(
            symbol=df.get("symbol", "BTCUSDT"),
            open_time_ms=pd.NA,
            close_time_ms=pd.NA,
            trade_count=pd.NA,
            taker_buy_base_volume=pd.NA,
            taker_buy_quote_volume=pd.NA,
            taker_buy_base_ratio=pd.NA,
            taker_buy_quote_ratio=pd.NA,
            time_zone="UTC",
            source="legacy_market_csv",
        )[expected_columns]
    return pd.DataFrame(columns=expected_columns)


def load_best_canonical_btc_df() -> pd.DataFrame:
    candidates: list[tuple[pd.Timestamp, Path]] = []
    if BTC_CANONICAL_PATH.exists():
        try:
            df = pd.read_csv(BTC_CANONICAL_PATH)
            if not df.empty and "date" in df.columns:
                last_date = pd.to_datetime(df["date"], errors="coerce").max()
                if pd.notna(last_date):
                    candidates.append((last_date, BTC_CANONICAL_PATH))
        except Exception:
            pass
    for snapshot in SNAPSHOT_ROOT.glob("btc_1d_*.csv"):
        try:
            df = pd.read_csv(snapshot)
            if not df.empty and "date" in df.columns:
                last_date = pd.to_datetime(df["date"], errors="coerce").max()
                if pd.notna(last_date):
                    candidates.append((last_date, snapshot))
        except Exception:
            continue
    if not candidates:
        return pd.DataFrame()
    best_path = max(candidates, key=lambda x: x[0])[1]
    return pd.read_csv(best_path)


def sync_market_csv_from_canonical(symbol: str) -> tuple[Path, pd.DataFrame]:
    market_path = MARKET_DIR / f"binance_spot_daily_{symbol}.csv"
    market_df = read_existing_market_csv(market_path)
    if symbol != "BTCUSDT":
        return market_path, market_df
    canonical_df = load_best_canonical_btc_df()
    if canonical_df.empty:
        return market_path, market_df
    if not {"date", "timeOpen", "timeClose", "priceOpen", "priceHigh", "priceLow", "priceClose", "volume", "baseVolume"}.issubset(canonical_df.columns):
        return market_path, market_df
    converted = pd.DataFrame(
        {
            "date": pd.to_datetime(canonical_df["date"], errors="coerce").dt.strftime("%Y-%m-%d"),
            "symbol": symbol,
            "open_time_ms": pd.to_numeric(canonical_df["timeOpen"], errors="coerce"),
            "close_time_ms": pd.to_numeric(canonical_df["timeClose"], errors="coerce"),
            "open": pd.to_numeric(canonical_df["priceOpen"], errors="coerce"),
            "high": pd.to_numeric(canonical_df["priceHigh"], errors="coerce"),
            "low": pd.to_numeric(canonical_df["priceLow"], errors="coerce"),
            "close": pd.to_numeric(canonical_df["priceClose"], errors="coerce"),
            "volume": pd.to_numeric(canonical_df["baseVolume"], errors="coerce"),
            "quote_volume": pd.to_numeric(canonical_df["volume"], errors="coerce"),
            "trade_count": pd.to_numeric(canonical_df.get("tradeCount"), errors="coerce"),
            "taker_buy_base_volume": pd.to_numeric(canonical_df.get("takerBuyBaseVolume"), errors="coerce"),
            "taker_buy_quote_volume": pd.to_numeric(canonical_df.get("takerBuyQuoteVolume"), errors="coerce"),
            "taker_buy_base_ratio": pd.to_numeric(canonical_df.get("takerBuyBaseVolume"), errors="coerce")
            / pd.to_numeric(canonical_df["baseVolume"], errors="coerce"),
            "taker_buy_quote_ratio": pd.to_numeric(canonical_df.get("takerBuyQuoteVolume"), errors="coerce")
            / pd.to_numeric(canonical_df["volume"], errors="coerce"),
            "time_zone": "UTC",
            "source": "canonical_backfill",
        }
    )
    merged = pd.concat([market_df, converted], ignore_index=True)
    merged = merged.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)
    merged.to_csv(market_path, index=False)
    return market_path, merged


def refresh_market_csv(symbol: str, interval: str, time_zone: str, lookback_bars: int) -> tuple[Path, pd.DataFrame]:
    MARKET_DIR.mkdir(parents=True, exist_ok=True)
    market_path = MARKET_DIR / f"binance_spot_daily_{symbol}.csv"
    existing = read_existing_market_csv(market_path)

    if existing.empty:
        start_ms = int(pd.Timestamp("2022-01-01", tz=UTC).timestamp() * 1000)
    else:
        latest_open = int(existing["open_time_ms"].max())
        start_ms = max(0, latest_open - (lookback_bars * 86_400_000))
    end_ms = int(datetime.now(UTC).timestamp() * 1000)

    rows = fetch_klines_rest(symbol, interval, start_ms, end_ms, time_zone)
    fetched = []
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    for row in rows:
        open_time_ms = int(row[0])
        close_time_ms = int(row[6])
        if close_time_ms > now_ms:
            continue
        volume = float(row[5])
        quote_volume = float(row[7])
        taker_buy_base_volume = float(row[9])
        taker_buy_quote_volume = float(row[10])
        fetched.append(
            {
                "date": pd.Timestamp(open_time_ms, unit="ms", tz=UTC).strftime("%Y-%m-%d"),
                "symbol": symbol,
                "open_time_ms": open_time_ms,
                "close_time_ms": close_time_ms,
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": volume,
                "quote_volume": quote_volume,
                "trade_count": int(row[8]),
                "taker_buy_base_volume": taker_buy_base_volume,
                "taker_buy_quote_volume": taker_buy_quote_volume,
                "taker_buy_base_ratio": taker_buy_base_volume / volume if volume else None,
                "taker_buy_quote_ratio": taker_buy_quote_volume / quote_volume if quote_volume else None,
                "time_zone": "UTC",
                "source": "binance_rest",
            }
        )

    merged = pd.concat([existing, pd.DataFrame(fetched)], ignore_index=True)
    merged["open_time_ms"] = pd.to_numeric(merged["open_time_ms"], errors="coerce")
    merged["close_time_ms"] = pd.to_numeric(merged["close_time_ms"], errors="coerce")
    merged = merged.loc[merged["close_time_ms"] <= now_ms].copy()
    merged["date"] = pd.to_datetime(merged["date"]).dt.strftime("%Y-%m-%d")
    merged = merged.drop_duplicates(subset="open_time_ms", keep="last").sort_values("open_time_ms")
    merged.to_csv(market_path, index=False)
    return market_path, merged.reset_index(drop=True)


def validate_market_data(df: pd.DataFrame) -> tuple[bool, str]:
    if df.empty:
        return False, "data_integrity_fail:empty_market_data"
    if df["open_time_ms"].duplicated().any():
        return False, "data_integrity_fail:duplicate_open_time"
    diffs = pd.Series(df["open_time_ms"]).diff().dropna()
    if not diffs.empty and not (diffs == 86_400_000).all():
        return False, "data_integrity_fail:gap_or_non_daily_spacing"
    if not pd.Series(df["open_time_ms"]).is_monotonic_increasing:
        return False, "data_integrity_fail:non_monotonic_open_time"
    latest_close_ms = int(df["close_time_ms"].iloc[-1])
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    if latest_close_ms > now_ms:
        return False, "data_integrity_fail:future_close_time"
    return True, ""


def sync_canonical_btc_csv(market_df: pd.DataFrame) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = load_best_canonical_btc_df()
    if not existing.empty and "date" in existing.columns:
        existing_last = pd.to_datetime(existing["date"], errors="coerce").max()
        market_last = pd.to_datetime(market_df["date"], errors="coerce").max()
        if pd.notna(existing_last) and pd.notna(market_last) and existing_last > market_last:
            existing.to_csv(BTC_CANONICAL_PATH, index=False)
            return BTC_CANONICAL_PATH
    canonical = pd.DataFrame(
        {
            "date": pd.to_datetime(market_df["date"]),
            "timeOpen": pd.to_numeric(market_df["open_time_ms"]),
            "timeClose": pd.to_numeric(market_df["close_time_ms"]),
            "priceOpen": pd.to_numeric(market_df["open"]),
            "priceHigh": pd.to_numeric(market_df["high"]),
            "priceLow": pd.to_numeric(market_df["low"]),
            "priceClose": pd.to_numeric(market_df["close"]),
            "volume": pd.to_numeric(market_df["quote_volume"]),
            "baseVolume": pd.to_numeric(market_df["volume"]),
            "tradeCount": pd.to_numeric(market_df["trade_count"]),
            "takerBuyBaseVolume": pd.to_numeric(market_df["taker_buy_base_volume"]),
            "takerBuyQuoteVolume": pd.to_numeric(market_df["taker_buy_quote_volume"]),
        }
    ).sort_values("date")
    canonical.to_csv(BTC_CANONICAL_PATH, index=False)
    return BTC_CANONICAL_PATH


def load_canonical_close_time_ms(asof_date_utc: str) -> int | None:
    if not BTC_CANONICAL_PATH.exists():
        return None
    df = pd.read_csv(BTC_CANONICAL_PATH)
    if "date" not in df.columns:
        return None
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    match = df.loc[df["date"] == asof_date_utc]
    if match.empty or "timeClose" not in match.columns:
        return None
    return int(pd.to_numeric(match.iloc[-1]["timeClose"], errors="coerce"))


def infer_asof_date(signal_payload: dict[str, Any]) -> str:
    return str(signal_payload["date"])


def load_signal_payload() -> dict[str, Any]:
    return json.loads((OUTPUT_SIGNALS / "btc_1d_signal_latest.json").read_text(encoding="utf-8"))


def load_combo_policy() -> dict[str, Any]:
    return json.loads(COMBO_POLICY_PATH.read_text(encoding="utf-8"))


def load_risk_budget_policy() -> dict[str, Any]:
    return json.loads(RISK_BUDGET_POLICY_PATH.read_text(encoding="utf-8"))


def build_combo_payload(signal_payload: dict[str, Any], cross_payload: dict[str, Any]) -> dict[str, Any]:
    policy = load_combo_policy()
    btc_weight = float(policy["btc_weight"])
    cross_weight = float(policy["cross_weight"])
    combo_weights: dict[str, float] = {}
    btc_exposure = float(signal_payload["v2_target_exposure"])
    if btc_exposure > 0:
        combo_weights["BTC"] = combo_weights.get("BTC", 0.0) + btc_weight * btc_exposure
    for asset, weight in cross_payload["selected_weights"].items():
        combo_weights[asset] = combo_weights.get(asset, 0.0) + cross_weight * float(weight)
    combo_weights = {asset: round(weight, 6) for asset, weight in combo_weights.items() if abs(weight) > 1e-12}
    combo_net_exposure = round(sum(combo_weights.values()), 6)
    combo_gross_exposure = round(sum(abs(weight) for weight in combo_weights.values()), 6)
    return {
        "policy_name": policy["policy_name"],
        "btc_weight": btc_weight,
        "cross_weight": cross_weight,
        "target_weights": combo_weights,
        "net_exposure": combo_net_exposure,
        "gross_exposure": combo_gross_exposure,
        "risk_action": "none",
        "risk_reason": "",
        "risk_halt": False,
    }


def build_combo_stablecoin_candidate(combo_payload: dict[str, Any], asof_date_utc: str) -> dict[str, Any]:
    out = dict(combo_payload)
    out["candidate_name"] = "combo_stablecoin_extreme_off"
    out["stablecoin_30d_z"] = None
    out["stablecoin_risk_off_flag"] = False
    out["stablecoin_extreme_off_flag"] = False
    out["candidate_action"] = "same_as_combo"
    out["candidate_reason"] = ""

    filt = load_stablecoin_macro_filter().copy()
    filt["date"] = pd.to_datetime(filt["date"])
    row = filt.loc[filt["date"] <= pd.Timestamp(asof_date_utc)].sort_values("date").tail(1)
    if row.empty:
        return out

    stablecoin_30d_z = row.iloc[0].get("stablecoin_30d_z")
    risk_off_flag = bool(row.iloc[0].get("stablecoin_risk_off_flag", 0))
    extreme_off_flag = bool(row.iloc[0].get("stablecoin_extreme_off_flag", 0))
    out["stablecoin_30d_z"] = None if pd.isna(stablecoin_30d_z) else float(stablecoin_30d_z)
    out["stablecoin_risk_off_flag"] = risk_off_flag
    out["stablecoin_extreme_off_flag"] = extreme_off_flag

    if extreme_off_flag:
        out["target_weights"] = {}
        out["net_exposure"] = 0.0
        out["gross_exposure"] = 0.0
        out["candidate_action"] = "flat_on_stablecoin_extreme_off"
        out["candidate_reason"] = "stablecoin_extreme_off"
    return out


def build_combo_vix_candidate(combo_payload: dict[str, Any], asof_date_utc: str) -> dict[str, Any]:
    out = dict(combo_payload)
    out["candidate_name"] = "combo_vix_elevated_half"
    out["vix_close"] = None
    out["vix_z_90d"] = None
    out["elevated_vix_flag"] = False
    out["extreme_vix_flag"] = False
    out["candidate_action"] = "same_as_combo"
    out["candidate_reason"] = ""

    filt = load_vix_filter().copy()
    filt["date"] = pd.to_datetime(filt["date"])
    row = filt.loc[filt["date"] <= pd.Timestamp(asof_date_utc)].sort_values("date").tail(1)
    if row.empty:
        return out

    vix_close = row.iloc[0].get("vix_close")
    vix_z_90d = row.iloc[0].get("vix_z_90d")
    elevated_flag = bool(row.iloc[0].get("elevated_vix_flag", 0))
    extreme_flag = bool(row.iloc[0].get("extreme_vix_flag", 0))
    out["vix_close"] = None if pd.isna(vix_close) else float(vix_close)
    out["vix_z_90d"] = None if pd.isna(vix_z_90d) else float(vix_z_90d)
    out["elevated_vix_flag"] = elevated_flag
    out["extreme_vix_flag"] = extreme_flag

    if elevated_flag:
        out["target_weights"] = {asset: round(weight * 0.5, 6) for asset, weight in out["target_weights"].items()}
        out["net_exposure"] = round(sum(out["target_weights"].values()), 6)
        out["gross_exposure"] = round(sum(abs(v) for v in out["target_weights"].values()), 6)
        out["candidate_action"] = "half_on_elevated_vix"
        out["candidate_reason"] = "elevated_vix"
    return out


def _trailing_drawdown(path: Path, pnl_col: str, days: int = 7) -> float | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    if df.empty or pnl_col not in df.columns:
        return None
    tail = df.loc[df["date"] >= df["date"].max() - pd.Timedelta(days=days - 1)].copy()
    if tail.empty:
        return None
    eq = (1.0 + pd.to_numeric(tail[pnl_col], errors="coerce").fillna(0.0)).cumprod()
    return float((eq / eq.cummax() - 1.0).min())


def _recent_halt_count(days: int = 30) -> int:
    ledger_path = FORWARD_ROOT / "forward_ledger.csv"
    if not ledger_path.exists():
        return 0
    df = pd.read_csv(ledger_path)
    if df.empty or "halt" not in df.columns or "asof_date_utc" not in df.columns:
        return 0
    df["asof_date_utc"] = pd.to_datetime(df["asof_date_utc"], errors="coerce")
    cutoff = df["asof_date_utc"].max() - pd.Timedelta(days=days - 1)
    recent = df.loc[df["asof_date_utc"] >= cutoff].copy()
    return int(pd.to_numeric(recent["halt"], errors="coerce").fillna(0).sum())


def apply_combo_risk_controls(combo_payload: dict[str, Any], halt: bool, halt_reason: str) -> dict[str, Any]:
    out = dict(combo_payload)
    policy = load_risk_budget_policy()
    portfolio = policy["portfolio"]

    if halt and portfolio.get("kill_switch_on_data_halt", True):
        out["target_weights"] = {}
        out["net_exposure"] = 0.0
        out["gross_exposure"] = 0.0
        out["risk_action"] = "flatten"
        out["risk_reason"] = halt_reason or "data_halt"
        out["risk_halt"] = True
        return out

    combo_dd = _trailing_drawdown(
        PROJECT_ROOT / "outputs" / "cross_sectional" / "shadow_validation" / "cross_sectional_shadow_equity.csv",
        "combo_pnl",
        days=7,
    )
    if combo_dd is not None and abs(combo_dd) > float(portfolio["max_weekly_drawdown"]):
        out["target_weights"] = {}
        out["net_exposure"] = 0.0
        out["gross_exposure"] = 0.0
        out["risk_action"] = "flatten"
        out["risk_reason"] = "weekly_drawdown_breach"
        out["risk_halt"] = True
        return out

    if portfolio.get("kill_switch_on_three_halts_30d", True) and _recent_halt_count(30) >= 3:
        out["target_weights"] = {}
        out["net_exposure"] = 0.0
        out["gross_exposure"] = 0.0
        out["risk_action"] = "flatten"
        out["risk_reason"] = "recent_halt_count_breach"
        out["risk_halt"] = True
        return out

    max_gross = float(portfolio["max_gross_exposure"])
    if out["gross_exposure"] > max_gross and out["gross_exposure"] > 0:
        scale = max_gross / out["gross_exposure"]
        out["target_weights"] = {asset: round(weight * scale, 6) for asset, weight in out["target_weights"].items()}
        out["net_exposure"] = round(sum(out["target_weights"].values()), 6)
        out["gross_exposure"] = round(sum(abs(v) for v in out["target_weights"].values()), 6)
        out["risk_action"] = "scale_down"
        out["risk_reason"] = "gross_exposure_cap"
    return out


def append_forward_ledger(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    existing_keys: set[tuple[str, str, str, str]] = set()
    if path.exists():
        existing_rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = reader.fieldnames or []
            for existing in reader:
                existing_rows.append(existing)
                existing_keys.add(
                    (
                        existing["asof_date_utc"],
                        existing["symbol"],
                        existing["interval"],
                        existing["git_sha"],
                    )
                )
        if existing_fieldnames != fieldnames:
            merged_fieldnames = list(existing_fieldnames)
            for field in fieldnames:
                if field not in merged_fieldnames:
                    merged_fieldnames.append(field)
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=merged_fieldnames)
                writer.writeheader()
                for existing in existing_rows:
                    writer.writerow({field: existing.get(field, "") for field in merged_fieldnames})
            fieldnames = merged_fieldnames
        if (
            row["asof_date_utc"],
            row["symbol"],
            row["interval"],
            row["git_sha"],
        ) in existing_keys:
            raise RuntimeError("forward ledger already contains this as-of date for the same code version")
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writerow(row)
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def write_latest_and_archive(
    context: RunContext,
    signal_payload: dict[str, Any],
    cross_payload: dict[str, Any],
    cross_2d_payload: dict[str, Any],
    combo_payload: dict[str, Any],
    combo_stablecoin_payload: dict[str, Any],
    combo_vix_payload: dict[str, Any],
    shock_payload: dict[str, Any],
    close_time_ms: int,
    raw_data_fingerprint: str,
    config_fingerprint: str,
    halt: bool,
    halt_reason: str,
    fee_bps: int,
    slip_bps: int,
    latency_mode: str,
) -> Path:
    OUTPUT_SIGNALS.mkdir(parents=True, exist_ok=True)
    archive_dir = OUTPUT_SIGNALS / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "context": asdict(context),
        "data": {
            "close": float(signal_payload["close"]),
            "close_time_ms": int(close_time_ms),
            "raw_data_fingerprint": raw_data_fingerprint,
            "config_fingerprint": config_fingerprint,
        },
        "signal": {
            "regime": signal_payload["regime"],
            "v1_target_exposure": float(signal_payload["v1_target_exposure"]),
            "v2_target_exposure": float(signal_payload["v2_target_exposure"]),
            "v2_reason": signal_payload["v2_reason"],
            "flow_alignment": int(signal_payload["flow_alignment"]),
            "flow_signal": signal_payload.get("flow_signal"),
            "recovery_flag": int(signal_payload["recovery_flag"]),
            "recovery_hold_flag": int(signal_payload["recovery_hold_flag"]),
        },
        "cross": {
            "universe_assets": cross_payload["universe_assets"],
            "selected_assets": cross_payload["selected_assets"],
            "selected_weights": cross_payload["selected_weights"],
            "portfolio_turnover": float(cross_payload["portfolio_turnover"]),
            "signal_version": "cross_v0",
        },
        "cross_2d_candidate": {
            "universe_assets": cross_2d_payload["universe_assets"],
            "selected_assets": cross_2d_payload["selected_assets"],
            "selected_weights": cross_2d_payload["selected_weights"],
            "portfolio_turnover": float(cross_2d_payload["portfolio_turnover"]),
            "signal_version": cross_2d_payload.get("signal_version", "cross_2d_candidate"),
        },
        "combo": combo_payload,
        "combo_stablecoin_candidate": combo_stablecoin_payload,
        "combo_vix_candidate": combo_vix_payload,
        "shock": {
            "threshold": shock_payload["threshold"],
            "trigger": shock_payload["trigger"],
            "shock_score": shock_payload["shock_score"],
            "shock_flag": shock_payload["shock_flag"],
            "reclaim_flag": shock_payload["reclaim_flag"],
            "target_exposure": shock_payload["target_exposure"],
            "hold_days_remaining": shock_payload["hold_days_remaining"],
            "reason": shock_payload["reason"],
            "selected_assets": shock_payload["selected_assets"],
            "selected_weights": shock_payload["selected_weights"],
        },
        "halt": halt,
        "halt_reason": halt_reason,
        "assumptions": {
            "expected_exec": "signal_close_next_open",
            "fee_bps": fee_bps,
            "slip_bps": slip_bps,
            "latency_mode": latency_mode,
        },
    }
    latest_json = OUTPUT_SIGNALS / "btc_1d_latest.json"
    latest_multi_json = OUTPUT_SIGNALS / "latest_multi.json"
    archive_json = archive_dir / f"{context.asof_date_utc}.json"
    latest_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    latest_multi_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not archive_json.exists():
        archive_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return archive_json


def run_shadow_reports() -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any]]:
    signal_log, latest_payload = production_signal.build_signal_payloads()
    cross_signal_log, cross_latest_payload = production_cross_sectional.build_cross_sectional_payloads()
    _, cross_2d_latest_payload = production_cross_sectional.build_cross_sectional_payloads(
        config=production_cross_sectional.RESEARCH_CANDIDATE_V1,
        label="cross_2d_candidate",
    )
    shock_signal_log, shock_latest_payload = production_shock_reversal.build_shock_payloads()
    production_signal.main()
    production_cross_sectional.main()
    production_shock_reversal.main()
    stat_stablecoin_macro_hint.main()
    stat_vix_hint.main()
    stat_shadow_divergence.main()
    stat_shadow_dashboard.main()
    stat_cross_sectional_shadow.main()
    stat_stablecoin_shadow.main()
    stat_stablecoin_shadow_gate.main()
    stat_shock_shadow.main()
    stat_risk_budget_sheet.main()
    stat_capacity_sheet.main()
    stat_weekly_attribution_sheet.main()
    return (
        signal_log,
        latest_payload,
        cross_signal_log,
        cross_latest_payload,
        cross_2d_latest_payload,
        shock_signal_log,
        shock_latest_payload,
    )


def write_log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    with (LOG_DIR / "daily_pipeline.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {message}\n")


def append_incident(level: str, run_id: str, asof_date_utc: str, message: str) -> None:
    INCIDENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "level": level,
        "run_id": run_id,
        "asof_date_utc": asof_date_utc,
        "message": message,
    }
    if INCIDENT_LOG.exists():
        with INCIDENT_LOG.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writerow(row)
        return
    with INCIDENT_LOG.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def archive_snapshot(asof_date_utc: str) -> Path:
    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    snapshot_path = SNAPSHOT_ROOT / f"btc_1d_{asof_date_utc}.csv"
    pd.read_csv(BTC_CANONICAL_PATH).to_csv(snapshot_path, index=False)
    return snapshot_path


def write_weekly_report() -> Path | None:
    ledger_path = FORWARD_ROOT / "forward_ledger.csv"
    if not ledger_path.exists():
        return None
    ledger = pd.read_csv(ledger_path, parse_dates=["asof_date_utc"])
    if ledger.empty:
        return None
    end = ledger["asof_date_utc"].max()
    start = end - pd.Timedelta(days=6)
    weekly = ledger.loc[ledger["asof_date_utc"] >= start].copy()
    WEEKLY_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = WEEKLY_ROOT / f"{end.strftime('%Y-%m-%d')}.md"
    incidents = pd.read_csv(INCIDENT_LOG) if INCIDENT_LOG.exists() else pd.DataFrame()
    incident_count = 0
    if not incidents.empty:
        incidents["timestamp_utc"] = pd.to_datetime(incidents["timestamp_utc"])
        incident_count = int((incidents["timestamp_utc"] >= pd.Timestamp(start).tz_localize("UTC")).sum())
    lines = [
        "# Weekly Forward Shadow Review",
        "",
        f"- window_start: {start.strftime('%Y-%m-%d')}",
        f"- window_end: {end.strftime('%Y-%m-%d')}",
        f"- runs: {len(weekly)}",
        f"- halts: {int(weekly['halt'].fillna(0).sum())}",
        f"- v1_v2_divergence_rows: {int((weekly['v1_target_exposure'] != weekly['v2_target_exposure']).sum())}",
        f"- incidents: {incident_count}",
    ]
    if "cross_selected_assets" in weekly.columns:
        selected = (
            weekly["cross_selected_assets"]
            .fillna("")
            .loc[weekly["cross_selected_assets"].fillna("") != ""]
            .value_counts()
        )
        if not selected.empty:
            lines.extend(
                [
                    "",
                    "## Cross Selection Frequency",
                    *[f"- {assets}: {count}" for assets, count in selected.head(5).items()],
                ]
            )
    if {"combo_net_exposure", "combo_gross_exposure"}.issubset(weekly.columns):
        lines.extend(
            [
                "",
                "## Combo Exposure",
                f"- avg_combo_net_exposure: {pd.to_numeric(weekly['combo_net_exposure'], errors='coerce').mean():.4f}",
                f"- avg_combo_gross_exposure: {pd.to_numeric(weekly['combo_gross_exposure'], errors='coerce').mean():.4f}",
            ]
        )
    if {"combo_stablecoin_candidate_net_exposure", "stablecoin_extreme_off_flag"}.issubset(weekly.columns):
        def count_episodes(flag: pd.Series) -> int:
            series = pd.to_numeric(flag, errors="coerce").fillna(0).astype(int)
            return int(((series == 1) & (series.shift(1, fill_value=0) == 0)).sum())

        extreme_flag = pd.to_numeric(weekly["stablecoin_extreme_off_flag"], errors="coerce").fillna(0).astype(int)
        extreme_days = int(extreme_flag.sum())
        extreme_episodes = count_episodes(extreme_flag)
        lines.extend(
            [
                "",
                "## Combo Stablecoin Candidate",
                f"- avg_candidate_net_exposure: {pd.to_numeric(weekly['combo_stablecoin_candidate_net_exposure'], errors='coerce').mean():.4f}",
                f"- stablecoin_extreme_off_days: {extreme_days}",
                f"- stablecoin_extreme_off_episodes: {extreme_episodes}",
            ]
        )
        gate_path = PROJECT_ROOT / "outputs" / "stablecoin_macro_hint" / "shadow_validation" / "stablecoin_shadow_gate.json"
        if gate_path.exists():
            try:
                gate = json.loads(gate_path.read_text(encoding="utf-8"))
                lines.extend(
                    [
                        "",
                        "## Stablecoin Overlay Gate Snapshot",
                        f"- overall: {gate.get('overall', 'n/a')}",
                        f"- gate_1: {gate.get('gate_1_historical_robustness', {}).get('status', 'n/a')}",
                        f"- gate_2: {gate.get('gate_2_structural_forward_correctness', {}).get('status', 'n/a')}",
                        f"- gate_3: {gate.get('gate_3_trigger_coverage', {}).get('status', 'n/a')}",
                        f"- gate_4: {gate.get('gate_4_live_separation_evidence', {}).get('status', 'n/a')}",
                    ]
                )
            except json.JSONDecodeError:
                lines.extend(
                    [
                        "",
                        "## Stablecoin Overlay Gate Snapshot",
                        "- overall: n/a (invalid gate JSON)",
                    ]
                )
    if {"combo_vix_candidate_net_exposure", "elevated_vix_flag", "extreme_vix_flag"}.issubset(weekly.columns):
        lines.extend(
            [
                "",
                "## Combo VIX Candidate",
                f"- avg_candidate_net_exposure: {pd.to_numeric(weekly['combo_vix_candidate_net_exposure'], errors='coerce').mean():.4f}",
                f"- elevated_vix_days: {int(pd.to_numeric(weekly['elevated_vix_flag'], errors='coerce').fillna(0).sum())}",
                f"- extreme_vix_days: {int(pd.to_numeric(weekly['extreme_vix_flag'], errors='coerce').fillna(0).sum())}",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_last_run(
    context: RunContext,
    signal_payload: dict[str, Any],
    cross_payload: dict[str, Any],
    combo_payload: dict[str, Any],
    combo_stablecoin_payload: dict[str, Any],
    combo_vix_payload: dict[str, Any],
    shock_payload: dict[str, Any],
    halt: bool,
    halt_reason: str,
) -> Path:
    FORWARD_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": context.run_id,
        "asof_date_utc": context.asof_date_utc,
        "git_sha": context.git_sha,
        "halt": bool(halt),
        "halt_reason": halt_reason,
        "btc": {
            "regime": signal_payload["regime"],
            "v1_target_exposure": float(signal_payload["v1_target_exposure"]),
            "v2_target_exposure": float(signal_payload["v2_target_exposure"]),
            "v2_reason": signal_payload["v2_reason"],
        },
        "cross": {
            "selected_assets": cross_payload["selected_assets"],
            "selected_weights": cross_payload["selected_weights"],
            "portfolio_turnover": float(cross_payload["portfolio_turnover"]),
        },
        "combo": combo_payload,
        "combo_stablecoin_candidate": combo_stablecoin_payload,
        "combo_vix_candidate": combo_vix_payload,
        "shock": {
            "target_exposure": float(shock_payload["target_exposure"]),
            "hold_days_remaining": int(shock_payload["hold_days_remaining"]),
            "reason": shock_payload["reason"],
            "selected_assets": shock_payload["selected_assets"],
        },
    }
    LAST_RUN_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return LAST_RUN_PATH


def write_ops_dashboard() -> tuple[Path | None, Path | None]:
    ledger_path = FORWARD_ROOT / "forward_ledger.csv"
    latest_multi = OUTPUT_SIGNALS / "latest_multi.json"
    if not ledger_path.exists() or not latest_multi.exists():
        return None, None
    ledger = pd.read_csv(ledger_path)
    if ledger.empty:
        return None, None
    latest = json.loads(latest_multi.read_text(encoding="utf-8"))
    incidents = pd.read_csv(INCIDENT_LOG) if INCIDENT_LOG.exists() else pd.DataFrame()
    recent_incidents = incidents.tail(5) if not incidents.empty else pd.DataFrame()
    halts = int(pd.to_numeric(ledger["halt"], errors="coerce").fillna(0).sum()) if "halt" in ledger.columns else 0
    divergence = int(
        (
            pd.to_numeric(ledger.get("v1_target_exposure", 0), errors="coerce").fillna(0.0)
            != pd.to_numeric(ledger.get("v2_target_exposure", 0), errors="coerce").fillna(0.0)
        ).sum()
    )
    payload = {
        "latest_context": latest["context"],
        "latest_signal": latest["signal"],
        "latest_cross": latest.get("cross", {}),
        "latest_combo": latest.get("combo", {}),
        "latest_combo_stablecoin_candidate": latest.get("combo_stablecoin_candidate", {}),
        "latest_combo_vix_candidate": latest.get("combo_vix_candidate", {}),
        "latest_shock": latest.get("shock", {}),
        "ops": {
            "ledger_rows": int(len(ledger)),
            "halt_count": halts,
            "divergence_rows": divergence,
            "recent_incident_count": int(len(recent_incidents)),
        },
    }
    OPS_DASHBOARD_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Forward Ops Dashboard",
        "",
        "## Latest Run",
        f"- asof_date_utc: {latest['context']['asof_date_utc']}",
        f"- git_sha: {latest['context']['git_sha']}",
        f"- halt: {latest['halt']}",
        f"- halt_reason: {latest['halt_reason'] or 'ok'}",
        "",
        "## BTC",
        f"- regime: {latest['signal']['regime']}",
        f"- v1_target_exposure: {latest['signal']['v1_target_exposure']:.4f}",
        f"- v2_target_exposure: {latest['signal']['v2_target_exposure']:.4f}",
        f"- v2_reason: {latest['signal']['v2_reason']}",
        "",
        "## Cross",
        f"- selected_assets: {', '.join(latest.get('cross', {}).get('selected_assets', [])) or 'none'}",
        f"- portfolio_turnover: {float(latest.get('cross', {}).get('portfolio_turnover', 0.0)):.4f}",
        "",
        "## Combo",
        f"- policy_name: {latest.get('combo', {}).get('policy_name', 'n/a')}",
        f"- net_exposure: {float(latest.get('combo', {}).get('net_exposure', 0.0)):.4f}",
        f"- gross_exposure: {float(latest.get('combo', {}).get('gross_exposure', 0.0)):.4f}",
        f"- risk_action: {latest.get('combo', {}).get('risk_action', 'none')}",
        f"- risk_reason: {latest.get('combo', {}).get('risk_reason', '') or 'ok'}",
        f"- risk_halt: {bool(latest.get('combo', {}).get('risk_halt', False))}",
        "",
        "## Combo Stablecoin Candidate",
        f"- candidate_action: {latest.get('combo_stablecoin_candidate', {}).get('candidate_action', 'same_as_combo')}",
        f"- candidate_reason: {latest.get('combo_stablecoin_candidate', {}).get('candidate_reason', '') or 'ok'}",
        f"- stablecoin_30d_z: {latest.get('combo_stablecoin_candidate', {}).get('stablecoin_30d_z', 'n/a')}",
        f"- stablecoin_risk_off_flag: {latest.get('combo_stablecoin_candidate', {}).get('stablecoin_risk_off_flag', False)}",
        f"- stablecoin_extreme_off_flag: {latest.get('combo_stablecoin_candidate', {}).get('stablecoin_extreme_off_flag', False)}",
        f"- candidate_net_exposure: {float(latest.get('combo_stablecoin_candidate', {}).get('net_exposure', 0.0)):.4f}",
        "",
        "## Combo VIX Candidate",
        f"- candidate_action: {latest.get('combo_vix_candidate', {}).get('candidate_action', 'same_as_combo')}",
        f"- candidate_reason: {latest.get('combo_vix_candidate', {}).get('candidate_reason', '') or 'ok'}",
        f"- vix_close: {latest.get('combo_vix_candidate', {}).get('vix_close', 'n/a')}",
        f"- vix_z_90d: {latest.get('combo_vix_candidate', {}).get('vix_z_90d', 'n/a')}",
        f"- elevated_vix_flag: {latest.get('combo_vix_candidate', {}).get('elevated_vix_flag', False)}",
        f"- extreme_vix_flag: {latest.get('combo_vix_candidate', {}).get('extreme_vix_flag', False)}",
        f"- candidate_net_exposure: {float(latest.get('combo_vix_candidate', {}).get('net_exposure', 0.0)):.4f}",
        "",
        "## Shock",
        f"- target_exposure: {float(latest.get('shock', {}).get('target_exposure', 0.0)):.4f}",
        f"- hold_days_remaining: {int(latest.get('shock', {}).get('hold_days_remaining', 0) or 0)}",
        f"- reason: {latest.get('shock', {}).get('reason', 'n/a')}",
        f"- selected_assets: {', '.join(latest.get('shock', {}).get('selected_assets', [])) or 'none'}",
        "",
        "## Ops",
        f"- ledger_rows: {len(ledger)}",
        f"- halt_count: {halts}",
        f"- divergence_rows: {divergence}",
        f"- recent_incident_count: {len(recent_incidents)}",
    ]
    if not recent_incidents.empty:
        lines.extend(["", "## Recent Incidents"])
        for _, row in recent_incidents.iterrows():
            lines.append(f"- {row['timestamp_utc']} [{row['level']}] {row['message']}")
    OPS_DASHBOARD_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return OPS_DASHBOARD_MD, OPS_DASHBOARD_JSON


def run_rerun_validation(asof_date_utc: str, fee_bps: int, slip_bps: int, latency_mode: str) -> tuple[bool, str]:
    snapshot_path = SNAPSHOT_ROOT / f"btc_1d_{asof_date_utc}.csv"
    archive_path = OUTPUT_SIGNALS / "archive" / f"{asof_date_utc}.json"
    if not snapshot_path.exists():
        latest_signal_path = OUTPUT_SIGNALS / "btc_1d_signal_latest.json"
        if latest_signal_path.exists():
            latest_signal = json.loads(latest_signal_path.read_text(encoding="utf-8"))
            if latest_signal.get("date") == asof_date_utc and BTC_CANONICAL_PATH.exists():
                SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
                pd.read_csv(BTC_CANONICAL_PATH).to_csv(snapshot_path, index=False)
    if not snapshot_path.exists():
        return False, f"missing_snapshot:{snapshot_path.name}"
    if not archive_path.exists():
        return False, f"missing_archive:{archive_path.name}"

    archived = json.loads(archive_path.read_text(encoding="utf-8"))
    env = dict(os.environ)
    env["BTC_FGI_PRICE_PATH"] = str(snapshot_path)
    subprocess.run(
        ["python3", "-m", "src.production_signal"],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    latest = load_signal_payload()
    rerun_payload = {
        "data": {
            "close": float(latest["close"]),
            "raw_data_fingerprint": sha256_file(snapshot_path),
            "config_fingerprint": sha256_paths([path for path in CONFIG_PATHS if path.exists()]),
        },
        "signal": {
            "regime": latest["regime"],
            "v1_target_exposure": float(latest["v1_target_exposure"]),
            "v2_target_exposure": float(latest["v2_target_exposure"]),
            "v2_reason": latest["v2_reason"],
            "flow_alignment": int(latest["flow_alignment"]),
            "flow_signal": latest.get("flow_signal"),
            "recovery_flag": int(latest["recovery_flag"]),
            "recovery_hold_flag": int(latest["recovery_hold_flag"]),
        },
        "halt": False,
        "halt_reason": "",
        "assumptions": {
            "expected_exec": "signal_close_next_open",
            "fee_bps": fee_bps,
            "slip_bps": slip_bps,
            "latency_mode": latency_mode,
        },
    }
    comparable_archived = {
        "data": {
            "close": archived["data"]["close"],
            "raw_data_fingerprint": archived["data"]["raw_data_fingerprint"],
            "config_fingerprint": archived["data"]["config_fingerprint"],
        },
        "signal": archived["signal"],
        "halt": archived["halt"],
        "halt_reason": archived["halt_reason"],
        "assumptions": archived["assumptions"],
    }
    RERUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RERUN_ROOT / f"{asof_date_utc}.json").write_text(json.dumps(rerun_payload, indent=2), encoding="utf-8")
    if comparable_archived != rerun_payload:
        return False, "rerun_output_mismatch"
    return True, "ok"


def main() -> None:
    args = parse_args()
    git_sha = get_git_sha()
    run_id = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_log(f"start run_id={run_id} symbol={args.symbol} interval={args.interval}")

    halt = False
    halt_reason = ""
    market_path = MARKET_DIR / f"binance_spot_daily_{args.symbol}.csv"
    try:
        if args.rerun_as_of:
            ok, reason = run_rerun_validation(
                asof_date_utc=args.rerun_as_of,
                fee_bps=args.fee_bps,
                slip_bps=args.slip_bps,
                latency_mode=args.latency_mode,
            )
            write_log(f"rerun asof_date={args.rerun_as_of} status={reason}")
            if not ok:
                append_incident("ERROR", run_id, args.rerun_as_of, reason)
                raise RuntimeError(reason)
            print(f"Rerun validation passed for {args.rerun_as_of}")
            return

        if args.skip_refresh:
            _, market_df = sync_market_csv_from_canonical(args.symbol)
        else:
            try:
                market_path, market_df = refresh_market_csv(
                    args.symbol,
                    args.interval,
                    args.time_zone,
                    args.lookback_bars,
                )
            except Exception as refresh_exc:
                _, market_df = sync_market_csv_from_canonical(args.symbol)
                append_incident("WARN", run_id, "", f"refresh_fallback:{refresh_exc}")
                write_log(f"refresh fallback run_id={run_id} reason={refresh_exc}")
                if market_df.empty:
                    raise
        data_ok, data_reason = validate_market_data(market_df)
        if not data_ok:
            halt = True
            halt_reason = data_reason
            append_incident("ERROR", run_id, "", halt_reason)
        else:
            sync_canonical_btc_csv(market_df)

        _, signal_payload, _, cross_payload, cross_2d_payload, _, shock_payload = run_shadow_reports()
        combo_payload = build_combo_payload(signal_payload, cross_payload)
        combo_payload = apply_combo_risk_controls(combo_payload, halt=halt, halt_reason=halt_reason)
        if combo_payload.get("risk_halt") and not halt:
            append_incident("WARN", run_id, "", f"combo_risk_halt:{combo_payload.get('risk_reason', '')}")
        asof_date_utc = infer_asof_date(signal_payload)
        combo_stablecoin_payload = build_combo_stablecoin_candidate(combo_payload, asof_date_utc)
        combo_vix_payload = build_combo_vix_candidate(combo_payload, asof_date_utc)
        context = RunContext(
            run_id=run_id,
            asof_date_utc=asof_date_utc,
            candle_timezone="UTC",
            symbol=args.symbol,
            interval=args.interval,
            git_sha=git_sha,
            python_version=platform.python_version(),
            host=platform.node(),
        )
        raw_data_fingerprint = sha256_file(BTC_CANONICAL_PATH)
        config_fingerprint = sha256_paths([path for path in CONFIG_PATHS if path.exists()])
        asof_match = market_df.loc[market_df["date"] == asof_date_utc]
        if asof_match.empty:
            close_time_ms = load_canonical_close_time_ms(asof_date_utc)
            if close_time_ms is None:
                raise RuntimeError(f"no market row found for asof_date_utc={asof_date_utc}")
        else:
            close_time_ms = int(asof_match.iloc[-1]["close_time_ms"])

        if halt:
            signal_payload["v1_target_exposure"] = 0.0
            signal_payload["v2_target_exposure"] = 0.0
            shock_payload["target_exposure"] = 0.0
            shock_payload["selected_assets"] = []
            shock_payload["selected_weights"] = {}
            shock_payload["hold_days_remaining"] = 0
            shock_payload["reason"] = halt_reason

        archive_snapshot(context.asof_date_utc)
        write_latest_and_archive(
            context=context,
            signal_payload=signal_payload,
            cross_payload=cross_payload,
            cross_2d_payload=cross_2d_payload,
            combo_payload=combo_payload,
            combo_stablecoin_payload=combo_stablecoin_payload,
            combo_vix_payload=combo_vix_payload,
            shock_payload=shock_payload,
            close_time_ms=close_time_ms,
            raw_data_fingerprint=raw_data_fingerprint,
            config_fingerprint=config_fingerprint,
            halt=halt,
            halt_reason=halt_reason,
            fee_bps=args.fee_bps,
            slip_bps=args.slip_bps,
            latency_mode=args.latency_mode,
        )
        ledger_row = {
            "run_id": context.run_id,
            "asof_date_utc": context.asof_date_utc,
            "symbol": context.symbol,
            "interval": context.interval,
            "candle_timezone": context.candle_timezone,
            "git_sha": context.git_sha,
            "close": float(signal_payload["close"]),
            "close_time_ms": close_time_ms,
            "raw_data_fingerprint": raw_data_fingerprint,
            "config_fingerprint": config_fingerprint,
            "regime": signal_payload["regime"],
            "v1_target_exposure": float(signal_payload["v1_target_exposure"]),
            "v2_target_exposure": float(signal_payload["v2_target_exposure"]),
            "cross_selected_assets": ",".join(cross_payload["selected_assets"]),
            "cross_selected_weights": json.dumps(cross_payload["selected_weights"], sort_keys=True),
            "cross_portfolio_turnover": float(cross_payload["portfolio_turnover"]),
            "cross_signal_version": "cross_v0",
            "cross_2d_selected_assets": ",".join(cross_2d_payload["selected_assets"]),
            "cross_2d_selected_weights": json.dumps(cross_2d_payload["selected_weights"], sort_keys=True),
            "cross_2d_portfolio_turnover": float(cross_2d_payload["portfolio_turnover"]),
            "cross_2d_signal_version": cross_2d_payload.get("signal_version", "cross_2d_candidate"),
            "combo_policy": combo_payload["policy_name"],
            "combo_target_weights": json.dumps(combo_payload["target_weights"], sort_keys=True),
            "combo_net_exposure": combo_payload["net_exposure"],
            "combo_gross_exposure": combo_payload["gross_exposure"],
            "combo_stablecoin_candidate_policy": combo_stablecoin_payload.get("candidate_name", ""),
            "combo_stablecoin_candidate_weights": json.dumps(combo_stablecoin_payload["target_weights"], sort_keys=True),
            "combo_stablecoin_candidate_net_exposure": combo_stablecoin_payload["net_exposure"],
            "combo_stablecoin_candidate_gross_exposure": combo_stablecoin_payload["gross_exposure"],
            "stablecoin_30d_z": combo_stablecoin_payload.get("stablecoin_30d_z", ""),
            "stablecoin_risk_off_flag": int(bool(combo_stablecoin_payload.get("stablecoin_risk_off_flag", False))),
            "stablecoin_extreme_off_flag": int(bool(combo_stablecoin_payload.get("stablecoin_extreme_off_flag", False))),
            "combo_vix_candidate_policy": combo_vix_payload.get("candidate_name", ""),
            "combo_vix_candidate_weights": json.dumps(combo_vix_payload["target_weights"], sort_keys=True),
            "combo_vix_candidate_net_exposure": combo_vix_payload["net_exposure"],
            "combo_vix_candidate_gross_exposure": combo_vix_payload["gross_exposure"],
            "vix_close": combo_vix_payload.get("vix_close", ""),
            "vix_z_90d": combo_vix_payload.get("vix_z_90d", ""),
            "elevated_vix_flag": int(bool(combo_vix_payload.get("elevated_vix_flag", False))),
            "extreme_vix_flag": int(bool(combo_vix_payload.get("extreme_vix_flag", False))),
            "combo_btc_weight": combo_payload["btc_weight"],
            "combo_cross_weight": combo_payload["cross_weight"],
            "combo_v2_effective_exposure": float(signal_payload["v2_target_exposure"]) * combo_payload["btc_weight"],
            "combo_risk_action": combo_payload.get("risk_action", "none"),
            "combo_risk_reason": combo_payload.get("risk_reason", ""),
            "combo_risk_halt": int(bool(combo_payload.get("risk_halt", False))),
            "v2_reason": signal_payload["v2_reason"],
            "flow_alignment": int(signal_payload["flow_alignment"]),
            "recovery_flag": int(signal_payload["recovery_flag"]),
            "halt": int(halt),
            "halt_reason": halt_reason,
            "incident_id": context.run_id if halt else "",
            "expected_exec": "signal_close_next_open",
            "fee_bps": args.fee_bps,
            "slip_bps": args.slip_bps,
            "latency_mode": args.latency_mode,
        }
        try:
            append_forward_ledger(FORWARD_ROOT / "forward_ledger.csv", ledger_row)
        except RuntimeError:
            if not args.allow_existing_asof:
                raise
            write_log(
                f"skip duplicate ledger asof_date={context.asof_date_utc} git_sha={context.git_sha}"
            )
        shock_ledger_row = {
            "run_id": context.run_id,
            "asof_date_utc": context.asof_date_utc,
            "symbol": "SHOCK_REVERSAL",
            "interval": context.interval,
            "candle_timezone": context.candle_timezone,
            "git_sha": context.git_sha,
            "raw_data_fingerprint": raw_data_fingerprint,
            "config_fingerprint": config_fingerprint,
            "threshold": shock_payload["threshold"],
            "trigger": shock_payload["trigger"],
            "shock_score": shock_payload["shock_score"],
            "shock_flag": int(shock_payload["shock_flag"]),
            "reclaim_flag": int(shock_payload["reclaim_flag"]),
            "target_exposure": float(shock_payload["target_exposure"]),
            "hold_days_remaining": int(shock_payload["hold_days_remaining"]),
            "reason": shock_payload["reason"],
            "selected_assets": ",".join(shock_payload["selected_assets"]),
            "selected_weights": json.dumps(shock_payload["selected_weights"], sort_keys=True),
            "halt": int(halt),
            "halt_reason": halt_reason,
            "incident_id": context.run_id if halt else "",
        }
        try:
            append_forward_ledger(SHOCK_FORWARD_LEDGER, shock_ledger_row)
        except RuntimeError:
            if not args.allow_existing_asof:
                raise
            write_log(f"skip duplicate shock ledger asof_date={context.asof_date_utc} git_sha={context.git_sha}")
        write_last_run(
            context=context,
            signal_payload=signal_payload,
            cross_payload=cross_payload,
            combo_payload=combo_payload,
            combo_stablecoin_payload=combo_stablecoin_payload,
            combo_vix_payload=combo_vix_payload,
            shock_payload=shock_payload,
            halt=halt,
            halt_reason=halt_reason,
        )
        write_weekly_report()
        write_ops_dashboard()
        write_log(
            f"finish run_id={run_id} asof_date={context.asof_date_utc} halt={int(halt)} reason={halt_reason or 'ok'}"
        )
        print(f"Wrote {(FORWARD_ROOT / 'forward_ledger.csv')}")
    except Exception as exc:
        append_incident("ERROR", run_id, "", str(exc))
        write_log(f"error run_id={run_id} error={exc}")
        raise


if __name__ == "__main__":
    main()
