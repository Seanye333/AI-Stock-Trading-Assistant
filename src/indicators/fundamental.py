"""
Fundamental analysis scoring and summary generation.
"""
from __future__ import annotations


def score_fundamentals(info: dict) -> dict:
    """Score fundamental metrics on a 0-100 scale.

    Returns a dict with individual scores and an overall composite score.
    """
    scores = {}
    weights = {}

    # --- Valuation score (lower PE/PB is better, but not negative) ---
    pe = info.get("pe_ratio")
    if pe and pe > 0:
        # Score: PE < 15 = 100, PE = 25 = 50, PE > 40 = 0
        scores["valuation"] = max(0, min(100, 100 - (pe - 15) * 3.3))
        weights["valuation"] = 0.15

    # --- Growth score ---
    rev_growth = info.get("revenue_growth")
    if rev_growth is not None:
        # 30%+ growth = 100, 0% = 50, negative = 0
        scores["growth"] = max(0, min(100, 50 + rev_growth * 167))
        weights["growth"] = 0.20

    # --- Profitability score (margins) ---
    net_margin = info.get("net_margin")
    if net_margin is not None:
        # 20%+ = 100, 10% = 60, 0% = 20, negative = 0
        scores["profitability"] = max(0, min(100, net_margin * 500))
        weights["profitability"] = 0.20

    # --- Efficiency (ROE) ---
    roe = info.get("roe")
    if roe is not None:
        # ROE 20%+ = 100, 10% = 60, 0% = 20
        scores["efficiency"] = max(0, min(100, roe * 500))
        weights["efficiency"] = 0.15

    # --- Financial health (debt-to-equity) ---
    de = info.get("debt_to_equity")
    if de is not None and de >= 0:
        # D/E < 0.5 = 100, D/E = 1 = 70, D/E = 3 = 30, D/E > 5 = 0
        scores["financial_health"] = max(0, min(100, 100 - de * 20))
        weights["financial_health"] = 0.15

    # --- Free cash flow score ---
    fcf = info.get("free_cash_flow")
    mkt_cap = info.get("market_cap")
    if fcf is not None and mkt_cap and mkt_cap > 0:
        fcf_yield = fcf / mkt_cap
        # FCF yield 5%+ = 100, 2% = 60, 0 = 20, negative = 0
        scores["cash_flow"] = max(0, min(100, fcf_yield * 2000))
        weights["cash_flow"] = 0.15

    # --- Composite ---
    if scores:
        total_weight = sum(weights.get(k, 0.1) for k in scores)
        composite = sum(
            scores[k] * weights.get(k, 0.1) for k in scores
        ) / total_weight
    else:
        composite = 50.0

    return {
        "scores": scores,
        "composite": round(composite, 1),
        "rating": _rating(composite),
    }


def _rating(score: float) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 65:
        return "Good"
    if score >= 50:
        return "Average"
    if score >= 35:
        return "Below Average"
    return "Poor"


def format_fundamental_summary(info: dict) -> str:
    """Return a compact text summary of fundamentals for LLM context."""
    lines = [
        f"Company: {info.get('company', info.get('ticker'))}",
        f"Sector: {info.get('sector', 'N/A')} | Industry: {info.get('industry', 'N/A')}",
        f"Market Cap: ${_fmt_num(info.get('market_cap'))} | Beta: {info.get('beta', 'N/A')}",
        "",
        "--- Valuation ---",
        f"P/E (TTM): {info.get('pe_ratio', 'N/A')} | Fwd P/E: {info.get('forward_pe', 'N/A')}",
        f"P/S: {info.get('ps_ratio', 'N/A')} | P/B: {info.get('pb_ratio', 'N/A')}",
        "",
        "--- Earnings & Growth ---",
        f"EPS (TTM): {info.get('eps', 'N/A')} | Fwd EPS: {info.get('eps_forward', 'N/A')}",
        f"Revenue Growth: {_pct(info.get('revenue_growth'))} | Earnings Growth: {_pct(info.get('earnings_growth'))}",
        "",
        "--- Margins ---",
        f"Gross: {_pct(info.get('gross_margin'))} | Operating: {_pct(info.get('operating_margin'))} | Net: {_pct(info.get('net_margin'))}",
        "",
        "--- Cash Flow & Balance Sheet ---",
        f"Free Cash Flow: ${_fmt_num(info.get('free_cash_flow'))} | CAPEX: ${_fmt_num(info.get('capex'))}",
        f"Debt/Equity: {info.get('debt_to_equity', 'N/A')} | Current Ratio: {info.get('current_ratio', 'N/A')}",
        "",
        "--- Returns ---",
        f"ROE: {_pct(info.get('roe'))} | ROA: {_pct(info.get('roa'))}",
        "",
        "--- Analyst ---",
        f"Target Price: {info.get('target_mean_price', 'N/A')} | Recommendation: {info.get('analyst_recommendation', 'N/A')}",
    ]
    return "\n".join(lines)


def _pct(val) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def _fmt_num(val) -> str:
    if val is None:
        return "N/A"
    if abs(val) >= 1e12:
        return f"{val/1e12:.2f}T"
    if abs(val) >= 1e9:
        return f"{val/1e9:.2f}B"
    if abs(val) >= 1e6:
        return f"{val/1e6:.2f}M"
    return str(val)
