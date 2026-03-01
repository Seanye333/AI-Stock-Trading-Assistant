"""
Technical indicator calculations using pandas-ta (pure Python, no C dependencies).

Adds all indicators in-place to the OHLCV DataFrame and returns it.
"""
from __future__ import annotations

import pandas as pd
import pandas_ta as ta


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append RSI, MACD, VWAP, Bollinger Bands, ATR, EMA,
    SMA, Stochastic, and ADX to the dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: Open, High, Low, Close, Volume.

    Returns
    -------
    pd.DataFrame
        Same dataframe with indicator columns appended.
    """
    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # --- RSI ---
    df["RSI"] = ta.rsi(close, length=14)

    # --- MACD ---
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    if macd is not None:
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACD_signal"] = macd["MACDs_12_26_9"]
        df["MACD_hist"] = macd["MACDh_12_26_9"]

    # --- Bollinger Bands ---
    bb = ta.bbands(close, length=20, std=2)
    if bb is not None:
        df["BB_upper"] = bb["BBU_20_2.0"]
        df["BB_mid"] = bb["BBM_20_2.0"]
        df["BB_lower"] = bb["BBL_20_2.0"]
        df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]
        df["BB_pct"] = (close - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])

    # --- VWAP (daily rolling) ---
    df["VWAP"] = _rolling_vwap(high, low, close, volume, window=20)

    # --- ATR ---
    df["ATR"] = ta.atr(high, low, close, length=14)

    # --- Moving Averages ---
    df["EMA_9"] = ta.ema(close, length=9)
    df["EMA_21"] = ta.ema(close, length=21)
    df["SMA_50"] = ta.sma(close, length=50)
    df["SMA_200"] = ta.sma(close, length=200)

    # --- ADX ---
    adx = ta.adx(high, low, close, length=14)
    if adx is not None:
        df["ADX"] = adx["ADX_14"]
        df["DI_plus"] = adx["DMP_14"]
        df["DI_minus"] = adx["DMN_14"]

    # --- Stochastic ---
    stoch = ta.stoch(high, low, close, k=14, d=3)
    if stoch is not None:
        df["STOCH_K"] = stoch["STOCHk_14_3_3"]
        df["STOCH_D"] = stoch["STOCHd_14_3_3"]

    # --- OBV (On-Balance Volume) ---
    df["OBV"] = ta.obv(close, volume)

    # --- Price-derived features ---
    df["Returns"] = close.pct_change()
    df["Log_Returns"] = close.pct_change().apply(lambda x: pd.NA if pd.isna(x) else __import__("math").log(1 + x))
    df["Volatility_20"] = df["Returns"].rolling(20).std() * (252 ** 0.5)

    # Trend signals
    df["Golden_Cross"] = (df["SMA_50"] > df["SMA_200"]).astype(int)
    df["Price_above_EMA21"] = (close > df["EMA_21"]).astype(int)

    return df


def _rolling_vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Rolling VWAP approximation using typical price."""
    typical = (high + low + close) / 3
    tp_vol = typical * volume
    return tp_vol.rolling(window).sum() / volume.rolling(window).sum()


def get_signal_summary(df: pd.DataFrame) -> dict:
    """Return a human-readable summary of the latest indicator values."""
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    signals = {}

    # RSI
    rsi = latest.get("RSI")
    if pd.notna(rsi):
        signals["RSI"] = {
            "value": round(float(rsi), 2),
            "signal": "Oversold" if rsi < 30 else ("Overbought" if rsi > 70 else "Neutral"),
        }

    # MACD
    macd = latest.get("MACD")
    macd_sig = latest.get("MACD_signal")
    if pd.notna(macd) and pd.notna(macd_sig):
        signals["MACD"] = {
            "value": round(float(macd), 4),
            "signal_line": round(float(macd_sig), 4),
            "signal": "Bullish" if macd > macd_sig else "Bearish",
        }

    # Bollinger Bands
    bb_pct = latest.get("BB_pct")
    if pd.notna(bb_pct):
        signals["Bollinger"] = {
            "bb_pct": round(float(bb_pct), 3),
            "signal": "Near Upper Band" if bb_pct > 0.8 else ("Near Lower Band" if bb_pct < 0.2 else "Mid Range"),
        }

    # VWAP
    vwap = latest.get("VWAP")
    close = latest.get("Close")
    if pd.notna(vwap) and pd.notna(close):
        signals["VWAP"] = {
            "value": round(float(vwap), 2),
            "signal": "Above VWAP (Bullish)" if close > vwap else "Below VWAP (Bearish)",
        }

    # Moving Averages
    sma50 = latest.get("SMA_50")
    sma200 = latest.get("SMA_200")
    if pd.notna(sma50) and pd.notna(sma200):
        signals["MA_Trend"] = {
            "golden_cross": bool(sma50 > sma200),
            "signal": "Golden Cross (Bullish)" if sma50 > sma200 else "Death Cross (Bearish)",
        }

    # ADX
    adx = latest.get("ADX")
    if pd.notna(adx):
        signals["ADX"] = {
            "value": round(float(adx), 2),
            "signal": "Strong Trend" if adx > 25 else "Weak/No Trend",
        }

    return signals
