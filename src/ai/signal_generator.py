"""
Signal Generator: combines LLM analysis + ML model predictions
into a unified trading signal with confidence scoring.
"""
from __future__ import annotations

from typing import Optional
import pandas as pd

from src.ai.llm_analyst import analyse_stock
from src.ai.ml_models import get_or_train_model
from src.data.fetcher import fetch_fundamentals
from src.indicators.technical import add_all_indicators


SIGNAL_WEIGHTS = {
    "llm": 0.50,      # Claude LLM analysis
    "ml": 0.35,       # XGBoost/LightGBM ensemble
    "technical": 0.15, # Raw technical rule-based
}


def generate_signal(
    ticker: str,
    price_df: pd.DataFrame,
    extra_context: str = "",
    use_llm: bool = True,
) -> dict:
    """Generate a comprehensive trading signal for a ticker.

    Parameters
    ----------
    ticker : str
    price_df : pd.DataFrame  raw OHLCV (will add indicators internally)
    extra_context : str  optional macro/news context for LLM
    use_llm : bool  set False to skip the Claude API call (faster/cheaper)

    Returns
    -------
    dict with keys:
        ticker, signal, confidence, composite_score,
        llm_analysis, ml_prediction, technical_score,
        price_target_pct, stop_loss_pct, take_profit_pct
    """
    # 1. Add technical indicators
    df = add_all_indicators(price_df)

    # 2. ML prediction
    model, train_metrics = get_or_train_model(ticker, df)
    ml_pred = model.predict_latest(df)
    ml_signal = ml_pred.get("signal", "HOLD")
    ml_conf = ml_pred.get("confidence", 50)

    # 3. Rule-based technical score (0-100)
    tech_score, tech_signals = _rule_based_score(df)

    # 4. LLM analysis
    llm_result = {}
    if use_llm:
        try:
            fundamentals = fetch_fundamentals(ticker)
            llm_result = analyse_stock(ticker, df, fundamentals, extra_context)
        except Exception as e:
            llm_result = {
                "signal": "HOLD",
                "confidence": 0,
                "price_target_pct": 0.0,
                "thesis": f"LLM analysis failed: {e}",
                "suggested_stop_loss_pct": 7.0,
                "suggested_take_profit_pct": 15.0,
                "error": str(e),
            }

    # 5. Composite signal
    signal_scores = {
        "BUY": 0.0,
        "HOLD": 0.0,
        "SELL": 0.0,
    }

    # LLM vote
    if use_llm and llm_result:
        llm_sig = llm_result.get("signal", "HOLD").upper()
        llm_conf = llm_result.get("confidence", 50) / 100.0
        if llm_sig in signal_scores:
            signal_scores[llm_sig] += SIGNAL_WEIGHTS["llm"] * llm_conf

    # ML vote
    ml_conf_norm = ml_conf / 100.0
    if ml_signal in signal_scores:
        signal_scores[ml_signal] += SIGNAL_WEIGHTS["ml"] * ml_conf_norm

    # Technical vote
    if tech_score > 65:
        signal_scores["BUY"] += SIGNAL_WEIGHTS["technical"]
    elif tech_score < 35:
        signal_scores["SELL"] += SIGNAL_WEIGHTS["technical"]
    else:
        signal_scores["HOLD"] += SIGNAL_WEIGHTS["technical"]

    final_signal = max(signal_scores, key=lambda k: signal_scores[k])
    composite_score = signal_scores[final_signal] * 100
    confidence = min(100, int(composite_score * 2))

    return {
        "ticker": ticker,
        "signal": final_signal,
        "confidence": confidence,
        "composite_score": round(composite_score, 1),
        "current_price": float(df.iloc[-1]["Close"]),
        # LLM
        "llm_signal": llm_result.get("signal", "N/A") if use_llm else "Skipped",
        "llm_confidence": llm_result.get("confidence", 0) if use_llm else 0,
        "llm_thesis": llm_result.get("thesis", ""),
        "llm_strengths": llm_result.get("key_strengths", []),
        "llm_risks": llm_result.get("key_risks", []),
        "llm_technical_outlook": llm_result.get("technical_outlook", ""),
        "llm_fundamental_outlook": llm_result.get("fundamental_outlook", ""),
        "fundamental_score": llm_result.get("fundamental_score", 0),
        # ML
        "ml_signal": ml_signal,
        "ml_confidence": ml_conf,
        "ml_prob_up": ml_pred.get("probability_up", 0.5),
        "ml_accuracy": ml_pred.get("model_accuracy", 0),
        "ml_train_metrics": train_metrics,
        # Technical
        "technical_score": tech_score,
        "technical_signals": tech_signals,
        # Trade parameters
        "price_target_pct": llm_result.get("price_target_pct", _default_target(final_signal)),
        "stop_loss_pct": llm_result.get("suggested_stop_loss_pct", 7.0),
        "take_profit_pct": llm_result.get("suggested_take_profit_pct", 15.0),
        # Raw
        "llm_raw": llm_result,
    }


def _rule_based_score(df: pd.DataFrame) -> tuple[float, dict]:
    """Simple rule-based technical scoring (0-100)."""
    latest = df.iloc[-1]
    score = 50.0
    signals = {}

    # RSI
    rsi = latest.get("RSI")
    if pd.notna(rsi):
        if rsi < 30:
            score += 15
            signals["RSI"] = "Oversold (+15)"
        elif rsi > 70:
            score -= 15
            signals["RSI"] = "Overbought (-15)"
        elif 40 <= rsi <= 60:
            signals["RSI"] = "Neutral"

    # MACD
    macd = latest.get("MACD")
    macd_sig = latest.get("MACD_signal")
    if pd.notna(macd) and pd.notna(macd_sig):
        if macd > macd_sig:
            score += 10
            signals["MACD"] = "Bullish crossover (+10)"
        else:
            score -= 10
            signals["MACD"] = "Bearish crossover (-10)"

    # Bollinger Bands
    bb_pct = latest.get("BB_pct")
    if pd.notna(bb_pct):
        if bb_pct < 0.1:
            score += 10
            signals["BB"] = "Near lower band (+10)"
        elif bb_pct > 0.9:
            score -= 10
            signals["BB"] = "Near upper band (-10)"

    # Moving average trend
    sma50 = latest.get("SMA_50")
    sma200 = latest.get("SMA_200")
    if pd.notna(sma50) and pd.notna(sma200):
        if sma50 > sma200:
            score += 10
            signals["MA"] = "Golden cross (+10)"
        else:
            score -= 10
            signals["MA"] = "Death cross (-10)"

    # VWAP
    vwap = latest.get("VWAP")
    close = latest.get("Close")
    if pd.notna(vwap) and pd.notna(close):
        if close > vwap:
            score += 5
            signals["VWAP"] = "Above VWAP (+5)"
        else:
            score -= 5
            signals["VWAP"] = "Below VWAP (-5)"

    return round(max(0, min(100, score)), 1), signals


def _default_target(signal: str) -> float:
    return 10.0 if signal == "BUY" else (-10.0 if signal == "SELL" else 0.0)
