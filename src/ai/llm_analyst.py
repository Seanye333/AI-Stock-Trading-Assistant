"""
LLM Analyst: uses Claude Opus 4.6 with adaptive thinking to generate
a structured investment thesis and trading signal.
"""
from __future__ import annotations

import json
from typing import Optional

import anthropic

from config.settings import ANTHROPIC_API_KEY, CLAUDE_MODEL
from src.indicators.fundamental import format_fundamental_summary, score_fundamentals
from src.indicators.technical import get_signal_summary


_client: Optional[anthropic.Anthropic] = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
            )
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


SYSTEM_PROMPT = """You are a senior quantitative analyst and portfolio manager with 20+ years of experience.
You analyse stocks using a blend of technical analysis, fundamental analysis, and macro context.

Your role:
1. Evaluate the provided technical indicators and fundamental data.
2. Generate a structured investment thesis with a clear BUY / HOLD / SELL signal.
3. Identify key risks and catalysts.
4. Provide a confidence level (0-100) and a short-term price target (1-4 weeks).

Always be objective, data-driven, and concise. Format your output as valid JSON.
"""

ANALYSIS_SCHEMA = {
    "signal": "BUY | HOLD | SELL",
    "confidence": "integer 0-100",
    "price_target_pct": "float: expected % move in 1-4 weeks (e.g. 5.5 for +5.5%)",
    "thesis": "2-3 sentence summary of the investment thesis",
    "key_strengths": ["list of top 3 bullish factors"],
    "key_risks": ["list of top 3 risk factors"],
    "technical_outlook": "brief technical summary",
    "fundamental_outlook": "brief fundamental summary",
    "suggested_stop_loss_pct": "float: suggested stop loss % below current price (e.g. 7.0)",
    "suggested_take_profit_pct": "float: suggested take profit % above current price (e.g. 15.0)",
}


def analyse_stock(
    ticker: str,
    df_with_indicators,
    fundamentals: dict,
    extra_context: str = "",
) -> dict:
    """Run Claude analysis on a stock.

    Parameters
    ----------
    ticker : str
    df_with_indicators : pd.DataFrame  with technical indicator columns
    fundamentals : dict  from fetch_fundamentals()
    extra_context : str  optional additional market/macro context

    Returns
    -------
    dict  with keys matching ANALYSIS_SCHEMA, plus 'raw_response' and 'error'.
    """
    client = _get_client()

    # Build context
    tech_signals = get_signal_summary(df_with_indicators)
    fund_summary = format_fundamental_summary(fundamentals)
    fund_scores = score_fundamentals(fundamentals)

    latest = df_with_indicators.iloc[-1]
    prev_close = df_with_indicators.iloc[-2]["Close"] if len(df_with_indicators) > 1 else latest["Close"]
    day_change_pct = (latest["Close"] - prev_close) / prev_close * 100

    user_prompt = f"""
=== STOCK ANALYSIS REQUEST ===
Ticker: {ticker}
Current Price: ${latest['Close']:.2f}  (Today: {day_change_pct:+.2f}%)
52W High: ${df_with_indicators['High'].max():.2f} | 52W Low: ${df_with_indicators['Low'].min():.2f}
Volume (today): {int(latest['Volume']):,}
Avg Volume (20d): {int(df_with_indicators['Volume'].tail(20).mean()):,}

=== TECHNICAL SIGNALS ===
{json.dumps(tech_signals, indent=2)}

Additional Indicators (latest):
- ATR (14): {_safe_round(latest.get('ATR'))}
- Volatility (20d annualised): {_safe_round(latest.get('Volatility_20'), pct=True)}
- OBV trend (5d change): {_obv_trend(df_with_indicators)}
- Stoch K/D: {_safe_round(latest.get('STOCH_K'))}/{_safe_round(latest.get('STOCH_D'))}

=== FUNDAMENTAL ANALYSIS ===
{fund_summary}

Fundamental Score: {fund_scores['composite']}/100 ({fund_scores['rating']})
Individual scores: {json.dumps(fund_scores['scores'], indent=2)}

=== EXTRA CONTEXT ===
{extra_context or 'None provided.'}

=== INSTRUCTIONS ===
Based on all the above data, produce a thorough analysis.
Return ONLY valid JSON matching exactly this schema:
{json.dumps(ANALYSIS_SCHEMA, indent=2)}
"""

    try:
        # Use streaming with adaptive thinking for robust LLM reasoning
        with client.messages.stream(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            response = stream.get_final_message()

        raw_text = ""
        for block in response.content:
            if block.type == "text":
                raw_text = block.text
                break

        # Parse JSON from response
        result = _extract_json(raw_text)
        result["ticker"] = ticker
        result["current_price"] = float(latest["Close"])
        result["fundamental_score"] = fund_scores["composite"]
        result["raw_response"] = raw_text
        result["error"] = None
        return result

    except Exception as e:
        return {
            "ticker": ticker,
            "signal": "HOLD",
            "confidence": 0,
            "price_target_pct": 0.0,
            "thesis": f"Analysis failed: {e}",
            "key_strengths": [],
            "key_risks": [],
            "technical_outlook": "N/A",
            "fundamental_outlook": "N/A",
            "suggested_stop_loss_pct": 7.0,
            "suggested_take_profit_pct": 15.0,
            "error": str(e),
            "raw_response": "",
        }


def _extract_json(text: str) -> dict:
    """Extract JSON object from LLM response text."""
    text = text.strip()
    # Find first { ... last }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    try:
        return json.loads(text[start:end])
    except json.JSONDecodeError:
        return {"raw_text": text}


def _safe_round(val, decimals: int = 2, pct: bool = False) -> str:
    if val is None:
        return "N/A"
    try:
        v = float(val)
        if pct:
            return f"{v*100:.1f}%"
        return str(round(v, decimals))
    except Exception:
        return "N/A"


def _obv_trend(df) -> str:
    if "OBV" not in df.columns or len(df) < 6:
        return "N/A"
    obv_5d = df["OBV"].iloc[-5:]
    if obv_5d.iloc[-1] > obv_5d.iloc[0]:
        return "Rising (accumulation)"
    return "Falling (distribution)"
