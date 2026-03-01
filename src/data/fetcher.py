"""
Data fetcher: pulls OHLCV price data and fundamental data via yfinance.
"""
from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def fetch_ohlcv(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV data for a single ticker.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume.
    Index is a DatetimeIndex (UTC-aware dropped to tz-naive).
    """
    raw = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if raw.empty:
        raise ValueError(f"No data returned for {ticker}")

    # Flatten MultiIndex columns if present (happens with single ticker too in newer yfinance)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.dropna(inplace=True)
    return df


def fetch_multiple(
    tickers: list[str],
    period: str = "1y",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple tickers. Returns {ticker: df}."""
    result = {}
    for t in tickers:
        try:
            result[t] = fetch_ohlcv(t, period=period, interval=interval)
        except Exception as e:
            print(f"[fetcher] Warning: could not fetch {t}: {e}")
    return result


# ---------------------------------------------------------------------------
# Fundamental data
# ---------------------------------------------------------------------------

def fetch_fundamentals(ticker: str) -> dict:
    """Pull key fundamental metrics from yfinance Info.

    Returns a dict with:
      eps, pe_ratio, revenue_growth, gross_margin, operating_margin,
      net_margin, capex, free_cash_flow, debt_to_equity, roe, roa,
      market_cap, beta, dividend_yield, sector, industry.
    """
    info = yf.Ticker(ticker).info

    def _safe(key: str, default=None):
        v = info.get(key, default)
        return v if v is not None else default

    return {
        "ticker": ticker,
        "company": _safe("longName", ticker),
        "sector": _safe("sector", "N/A"),
        "industry": _safe("industry", "N/A"),
        "market_cap": _safe("marketCap"),
        "beta": _safe("beta"),
        "dividend_yield": _safe("dividendYield"),
        # Valuation
        "pe_ratio": _safe("trailingPE"),
        "forward_pe": _safe("forwardPE"),
        "ps_ratio": _safe("priceToSalesTrailing12Months"),
        "pb_ratio": _safe("priceToBook"),
        # Earnings
        "eps": _safe("trailingEps"),
        "eps_forward": _safe("forwardEps"),
        # Growth
        "revenue_growth": _safe("revenueGrowth"),
        "earnings_growth": _safe("earningsGrowth"),
        # Margins
        "gross_margin": _safe("grossMargins"),
        "operating_margin": _safe("operatingMargins"),
        "net_margin": _safe("profitMargins"),
        # Cash flow
        "free_cash_flow": _safe("freeCashflow"),
        "capex": _safe("capitalExpenditures"),
        # Balance sheet
        "debt_to_equity": _safe("debtToEquity"),
        "current_ratio": _safe("currentRatio"),
        # Returns
        "roe": _safe("returnOnEquity"),
        "roa": _safe("returnOnAssets"),
        # Price targets
        "target_mean_price": _safe("targetMeanPrice"),
        "analyst_recommendation": _safe("recommendationKey", "N/A"),
    }


def fetch_fundamentals_multiple(tickers: list[str]) -> dict[str, dict]:
    """Fetch fundamentals for multiple tickers."""
    return {t: fetch_fundamentals(t) for t in tickers}
