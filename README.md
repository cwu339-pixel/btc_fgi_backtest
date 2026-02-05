# 📊 BTC Multi-Factor Quantitative Research Platform

A systematic quantitative research platform for Bitcoin trading strategies, combining behavioral indicators (Fear & Greed Index), technical factors (momentum, trend), and derivatives metrics (funding rate, open interest) with rigorous backtesting methodology.

## 🎯 Key Features

- **Multi-Factor Framework**: Integrates 5+ factors including FGI, momentum, trend-following, volatility, and derivatives crowding
- **Walk-Forward Validation**: Implements proper out-of-sample testing to prevent overfitting
- **Regime Analysis**: Segments performance by market conditions (bull/bear/crisis)
- **Derivatives Integration**: Incorporates futures funding rates and open interest for crowding signals
- **On-Chain Analytics**: Includes Dune SQL projects for Uniswap v3 metrics and whale behavior analysis

## 📁 Project Structure

```
btc_fgi_backtest/
├── src/
│   ├── data_source.py         # Spot data loader (OHLCV + FGI)
│   ├── data_source_deriv.py   # Derivatives data loader (funding, OI)
│   └── research.py            # Main research script with factor construction
├── notebooks/
│   └── bt_v0_fgi_trend.ipynb  # Interactive exploration and visualization
├── dune_project_metrics/       # Uniswap v3 SQL analytics
├── whale_analysis/             # Whale behavior SQL dashboards
└── data/                       # Local data (not tracked in Git)
```

## 🧪 Factor Library

### Behavioral Factors
- **Fear & Greed Index (FGI)**: Contrarian signals based on market sentiment extremes
- **Momentum**: Price-based momentum with multiple lookback periods

### Technical Factors
- **Trend-Following**: Moving average crossovers and directional filters
- **Low Volatility**: Risk-adjusted positioning during calm markets
- **Volume**: Breakout confirmation signals

### Derivatives Factors
- **Funding Rate**: Perpetual futures funding as crowding indicator
- **Open Interest**: Position size expansion/contraction signals

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/cwu339-pixel/btc_fgi_backtest.git
cd btc_fgi_backtest

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install pandas numpy openpyxl jupyter
```

### Data Setup

Place your local data files under `data/`:

```
data/
├── bitcoin.xlsx              # Daily OHLCV (columns: date, timeClose, priceClose)
├── fgi.json                  # Fear & Greed Index (fields: timestamp, value)
└── derivatives/
    ├── binance_funding_BTCUSDT.csv
    └── binance_oi_BTCUSDT.csv
```

**Data Sources**:
- BTC spot data: [Yahoo Finance](https://finance.yahoo.com/quote/BTC-USD/history), [CoinGecko API](https://www.coingecko.com/en/api)
- Fear & Greed Index: [Alternative.me API](https://alternative.me/crypto/fear-and-greed-index/)
- Derivatives: [Binance API](https://binance-docs.github.io/apidocs/futures/en/)

### Run Backtests

```bash
# Run full research pipeline
python -m src.research

# Or explore interactively
jupyter notebook notebooks/bt_v0_fgi_trend.ipynb
```

## 📊 Sample Output

```
=== Performance Summary ===
Strategy: Momentum + FGI Filter
Total Return: 156.3%
Sharpe Ratio: 1.82
Max Drawdown: -23.4%
Win Rate: 58.2%

=== Regime Analysis ===
Bull Market (FGI > 70): +45.2% return
Bear Market (FGI < 30): +12.1% return
Neutral: +8.5% return

=== Walk-Forward Results ===
Train Period (2020-2022): Sharpe 1.95
Test Period (2023-2024): Sharpe 1.68
```

## 🛠️ Advanced Usage

### Custom Factor Development

```python
from src.data_source import load_data
from src.research import build_factors

# Load data
df = load_data()

# Build custom factor
df['my_factor'] = (df['fgi'] < 25) & (df['momentum_20d'] > 0)

# Backtest
strategy_returns = df['daily_return'] * df['my_factor'].shift(1)
```

### Walk-Forward Optimization

The research script automatically runs walk-forward validation with:
- Training window: 2 years
- Test window: 6 months
- Rolling basis: 3 months

## 📈 Additional Projects

### Dune Analytics Dashboards

- **Uniswap v3 Project Metrics** (`dune_project_metrics/`): SQL queries for protocol analytics
- **Whale Behavior Analysis** (`whale_analysis/`): Large holder tracking and flow analysis

These are independent subprojects using [Dune Analytics](https://dune.com/) for on-chain data analysis.

## 🔬 Research Methodology

This project follows quantitative research best practices:

1. **Hypothesis-Driven**: Each factor has a clear economic rationale
2. **Out-of-Sample Testing**: Walk-forward validation prevents look-ahead bias
3. **Regime Analysis**: Performance evaluated across different market conditions
4. **Transaction Costs**: Future work will incorporate realistic trading costs
5. **Risk Management**: Position sizing based on volatility

## 📚 References & Inspiration

- Fama-French factor models
- Momentum strategies (Jegadeesh & Titman, 1993)
- Sentiment indicators in crypto markets
- Derivatives-based crowding signals

## ⚠️ Disclaimer

This project is for **educational and research purposes only**.

- **Not investment advice**: Past performance does not guarantee future results
- **No warranty**: Use at your own risk
- **Market risk**: Cryptocurrency trading involves substantial risk of loss

## 📝 License

MIT License - see [LICENSE](LICENSE) for details

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional factors (on-chain metrics, social sentiment)
- Machine learning integration
- Multi-asset support (ETH, altcoins)
- Real-time signal generation

## 📧 Contact

GitHub: [@cwu339-pixel](https://github.com/cwu339-pixel)

---

**⭐ If you find this project useful, please consider giving it a star!**
