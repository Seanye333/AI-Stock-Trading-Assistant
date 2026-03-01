"""
Technical indicator calculations using the `ta` library (Python 3.13 compatible).

Adds all indicators in-place to the OHLCV DataFrame and returns it.
"""
from __future__ import annotations

import math
import pandas as pd
import ta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append all technical indicators to the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: Open, High, Low, Close, Volume.

    Returns
    -------
    pd.DataFrame with indicator columns appended.
    """
    df = df.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # --- RSI ---
    df["RSI"] = RSIIndicator(close=close, window=14).rsi()

    # --- MACD ---
    macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]        = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_hist"]   = macd.macd_diff()

    # --- Bollinger Bands ---
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_mid"]   = bb.bollinger_mavg()
    df["BB_lower"] = bb.bollinger_lband()
    df["BB_width"] = bb.bollinger_wband()
    df["BB_pct"]   = bb.bollinger_pband()

    # --- VWAP (rolling 20-period approximation) ---
    df["VWAP"] = _rolling_vwap(high, low, close, volume, window=20)

    # --- ATR ---
    df["ATR"] = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()

    # --- Moving Averages ---
    df["EMA_9"]   = EMAIndicator(close=close, window=9).ema_indicator()
    df["EMA_21"]  = EMAIndicator(close=close, window=21).ema_indicator()
    df["SMA_50"]  = SMAIndicator(close=close, window=50).sma_indicator()
    df["SMA_200"] = SMAIndicator(close=close, window=200).sma_indicator()

    # --- ADX ---
    adx_ind = ADXIndicator(high=high, low=low, close=close, window=14)
    df["ADX"]      = adx_ind.adx()
    df["DI_plus"]  = adx_ind.adx_pos()
    df["DI_minus"] = adx_ind.adx_neg()

    # --- Stochastic ---
    stoch = StochasticOscillator(high=high, low=low, close=close, window=14, smooth_window=3)
    df["STOCH_K"] = stoch.stoch()
    df["STOCH_D"] = stoch.stoch_signal()

    # --- OBV ---
    df["OBV"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

    # --- Price-derived features ---
    df["Returns"]       = close.pct_change()
    df["Log_Returns"]   = df["Returns"].apply(
        lambda x: math.log(1 + x) if pd.notna(x) and x > -1 else float("nan")
    )
    df["Volatility_20"] = df["Returns"].rolling(20).std() * (252 ** 0.5)

    # --- Trend signals ---
    df["Golden_Cross"]      = (df["SMA_50"] > df["SMA_200"]).astype(int)
    df["Price_above_EMA21"] = (close > df["EMA_21"]).astype(int)

    return df


def _rolling_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Rolling VWAP using typical price."""
    typical = (high + low + close) / 3
    return (typical * volume).rolling(window).sum() / volume.rolling(window).sum()


def get_signal_summary(df: pd.DataFrame) -> dict:
    """Return a human-readable summary of the latest indicator values."""
    latest = df.iloc[-1]
    signals = {}

    rsi = latest.get("RSI")
    if pd.notna(rsi):
        signals["RSI"] = {
            "value": round(float(rsi), 2),
            "signal": "Oversold" if rsi < 30 else ("Overbought" if rsi > 70 else "Neutral"),
        }

    macd_val = latest.get("MACD")
    macd_sig = latest.get("MACD_signal")
    if pd.notna(macd_val) and pd.notna(macd_sig):
        signals["MACD"] = {
            "value": round(float(macd_val), 4),
            "signal_line": round(float(macd_sig), 4),
            "signal": "Bullish" if macd_val > macd_sig else "Bearish",
        }

    bb_pct = latest.get("BB_pct")
    if pd.notna(bb_pct):
        signals["Bollinger"] = {
            "bb_pct": round(float(bb_pct), 3),
            "signal": "Near Upper Band" if bb_pct > 0.8 else ("Near Lower Band" if bb_pct < 0.2 else "Mid Range"),
        }

    vwap = latest.get("VWAP")
    close_val = latest.get("Close")
    if pd.notna(vwap) and pd.notna(close_val):
        signals["VWAP"] = {
            "value": round(float(vwap), 2),
            "signal": "Above VWAP (Bullish)" if close_val > vwap else "Below VWAP (Bearish)",
        }

    sma50  = latest.get("SMA_50")
    sma200 = latest.get("SMA_200")
    if pd.notna(sma50) and pd.notna(sma200):
        signals["MA_Trend"] = {
            "golden_cross": bool(sma50 > sma200),
            "signal": "Golden Cross (Bullish)" if sma50 > sma200 else "Death Cross (Bearish)",
        }

    adx = latest.get("ADX")
    if pd.notna(adx):
        signals["ADX"] = {
            "value": round(float(adx), 2),
            "signal": "Strong Trend" if adx > 25 else "Weak/No Trend",
        }

    return signals
