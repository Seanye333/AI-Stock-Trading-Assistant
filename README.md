# AI Stock Trading Assistant

An AI-powered stock analysis and trading simulation system combining technical analysis, fundamental analysis, LLM reasoning (Claude), and ML models (XGBoost/LightGBM).

## Features

| Category | Details |
|---|---|
| **Data** | Yahoo Finance via `yfinance` (free, no API key required) |
| **Technical** | RSI, MACD, VWAP, Bollinger Bands, ATR, ADX, Stochastic, EMA/SMA |
| **Fundamental** | EPS, P/E, Revenue Growth, Margins, CAPEX, FCF, D/E, ROE/ROA |
| **AI Signals** | Claude Opus 4.6 (adaptive thinking) + XGBoost + LightGBM ensemble |
| **Backtesting** | Rule-based strategy engine with trade log and equity curve |
| **Monte Carlo** | GBM + bootstrap simulations, VaR, CVaR, fan charts |
| **Risk** | Kelly Criterion sizing, portfolio VaR, correlation matrix, HHI |
| **Dashboard** | Streamlit + Plotly interactive web UI |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### 3. Run the dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

## Project Structure

```
AI Stock Trading Assistant/
├── app.py                      # Streamlit entry point
├── requirements.txt
├── .env.example
├── config/
│   └── settings.py             # Global config & defaults
└── src/
    ├── data/
    │   └── fetcher.py          # yfinance OHLCV + fundamentals
    ├── indicators/
    │   ├── technical.py        # RSI, MACD, BB, VWAP, ATR, ADX...
    │   └── fundamental.py      # Scoring + summary formatting
    ├── ai/
    │   ├── llm_analyst.py      # Claude API integration
    │   ├── ml_models.py        # XGBoost + LightGBM ensemble
    │   └── signal_generator.py # Combined signal pipeline
    ├── backtesting/
    │   ├── engine.py           # Trade simulation
    │   └── monte_carlo.py      # MC simulations + VaR
    ├── risk/
    │   └── portfolio_manager.py # Position sizing + portfolio analytics
    └── dashboard/
        ├── app.py              # Main Streamlit app
        └── components/
            └── charts.py       # Plotly chart builders
```

## Usage

### Dashboard Tabs

- **Overview** — price chart, key metrics, fundamental breakdown
- **Technical** — indicator signals, ML feature importance
- **AI Analysis** — Claude LLM thesis + ML signal + position sizing
- **Backtest** — equity curve, trade log, performance stats
- **Monte Carlo** — price path simulations, VaR/CVaR
- **Portfolio** — correlation matrix, portfolio risk, watchlist

### Without Claude API

The app works fully without an `ANTHROPIC_API_KEY` — the LLM analysis tab will be disabled, but all other features (technical indicators, ML models, backtesting, Monte Carlo, portfolio analytics) work out of the box.

## Notes

- `pandas-ta` is used instead of TA-Lib (no C compilation required on Windows)
- Data is cached for 5 minutes (price) and 1 hour (fundamentals)
- ML models retrain automatically when a new ticker is selected
- Monte Carlo uses Geometric Brownian Motion (GBM) by default; switch to historical bootstrap in the sidebar
# AI-Stock-Trading-Assistant
