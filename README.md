# BTC FGI Factor Research

Factor research and simple backtests for BTC, blending Fear & Greed Index (FGI), price momentum, and derivative metrics such as funding and open interest. Includes a few Dune SQL subprojects that are unrelated to the BTC backtests but live in the same repo for convenience.

## Project Structure
- `src/data_source.py`: load daily spot data (`data/bitcoin.xlsx`, `data/fgi.json`), merge with derivatives, compute daily returns and buy-and-hold equity.
- `src/data_source_deriv.py`: load derivatives from `data/derivatives/*.csv` (funding, open interest), aggregate to daily series.
- `src/research.py`: main research script; builds factors (momentum, trend, low-vol, volume, funding, OI crowding), runs long-only strategies, and prints performance/segment summaries and simple walk-forward diagnostics.
- `notebooks/bt_v0_fgi_trend.ipynb`: interactive notebook for ad-hoc exploration and plots.
- `data/`: raw data folder (not tracked in Git; see below).
- `dune_project_metrics/`: Uniswap v3 Dune SQL project (dashboards/layout).
- `whale_analysis/`: Dune SQL/dashboard skeleton for whale behavior analysis.

## Data (not in repo)
Place local data under `data/`:
- Spot: `data/bitcoin.xlsx` with daily OHLCV (expects `date`/`timeClose` and `priceClose` columns).
- FGI: `data/fgi.json` with `timestamp`/`value` fields or a `{ "data": [...] }` wrapper.
- Derivatives: `data/derivatives/binance_funding_BTCUSDT.csv` and (optionally) `binance_oi_BTCUSDT.csv` with timestamps and numeric fields.

## Installation
1) Use Python 3.10+ and create a virtualenv:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
```
2) Install deps (minimal):
```bash
pip install pandas numpy openpyxl
```
Add `jupyter` if you want to run the notebook.

## Usage
- Script: from repo root, run the research entrypoint
```bash
python -m src.research
```
It loads local data, builds factors, runs long-only variants (momentum, filters, combos), and prints stats by regime and train/test splits.

- Notebook: open `notebooks/bt_v0_fgi_trend.ipynb` in VS Code or Jupyter to inspect data and plot curves.

## Dune Subprojects
- `dune_project_metrics/`: Uniswap v3 dashboards/SQL/layout docs.
- `whale_analysis/`: Whale behavior dashboards/SQL skeleton.
These are independent from the BTC factor code.

## Notes / Disclaimer
- Data files are excluded from Git; keep your local `data/` up to date before running.
- Strategies here are for research only, not investment advice. Use at your own risk.
