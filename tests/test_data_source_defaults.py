from __future__ import annotations

from pathlib import Path

from src import data_source


def test_default_bitcoin_path_falls_back_to_market_csv(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    market_path = data_dir / "market" / "binance_spot_daily_BTCUSDT.csv"
    market_path.parent.mkdir(parents=True, exist_ok=True)
    market_path.write_text("date,close\n2026-03-20,85000\n", encoding="utf-8")

    monkeypatch.delenv("BTC_FGI_PRICE_PATH", raising=False)
    monkeypatch.setattr(data_source, "DATA_DIR", data_dir)
    monkeypatch.setattr(data_source, "BITCOIN_PATH", data_dir / "bitcoin.xlsx")

    assert data_source.default_bitcoin_path() == market_path
