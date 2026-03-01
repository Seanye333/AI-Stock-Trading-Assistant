"""
Portfolio Risk Manager

Handles:
- Position sizing (Kelly Criterion / fixed fractional)
- Portfolio-level risk metrics (Sharpe, Sortino, Calmar, Beta, Alpha)
- Correlation analysis
- Concentration risk
- Drawdown analysis
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import MAX_POSITION_SIZE, STOP_LOSS_PCT


# ---------------------------------------------------------------------------
# Position Sizing
# ---------------------------------------------------------------------------

def kelly_position_size(
    win_rate: float,
    avg_win_pct: float,
    avg_loss_pct: float,
    max_fraction: float = MAX_POSITION_SIZE,
    kelly_fraction: float = 0.5,   # half-Kelly for conservatism
) -> float:
    """Calculate optimal position size using the (fractional) Kelly Criterion.

    Parameters
    ----------
    win_rate : float  e.g. 0.55 for 55%
    avg_win_pct : float  e.g. 0.08 for 8%
    avg_loss_pct : float  e.g. 0.04 for 4% (positive value)
    max_fraction : float  maximum cap
    kelly_fraction : float  fraction of full Kelly to use

    Returns
    -------
    float  recommended fraction of portfolio to deploy (0-1)
    """
    if avg_loss_pct <= 0:
        return max_fraction
    win_loss_ratio = avg_win_pct / avg_loss_pct
    kelly = win_rate - (1 - win_rate) / win_loss_ratio
    # Apply half-Kelly and cap at max_fraction
    size = max(0.0, min(max_fraction, kelly * kelly_fraction))
    return round(size, 4)


def fixed_fraction_position_size(
    capital: float,
    price: float,
    risk_pct: float = 0.02,            # risk 2% of capital per trade
    stop_loss_pct: float = STOP_LOSS_PCT,
) -> dict:
    """Calculate share count using fixed fractional position sizing.

    Risk per trade = capital * risk_pct
    Position size = Risk / Stop-loss per share

    Returns
    -------
    dict with shares, position_value, position_pct, risk_amount
    """
    risk_amount = capital * risk_pct
    stop_loss_per_share = price * stop_loss_pct
    shares = math.floor(risk_amount / stop_loss_per_share) if stop_loss_per_share > 0 else 0
    position_value = shares * price
    position_pct = position_value / capital if capital > 0 else 0

    # Cap at MAX_POSITION_SIZE
    if position_pct > MAX_POSITION_SIZE:
        position_pct = MAX_POSITION_SIZE
        position_value = capital * MAX_POSITION_SIZE
        shares = math.floor(position_value / price)
        position_value = shares * price

    return {
        "shares": shares,
        "position_value": round(position_value, 2),
        "position_pct": round(position_pct * 100, 2),
        "risk_amount": round(risk_amount, 2),
        "stop_loss_price": round(price * (1 - stop_loss_pct), 2),
    }


# ---------------------------------------------------------------------------
# Portfolio Analytics
# ---------------------------------------------------------------------------

def compute_portfolio_metrics(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.05,
) -> dict:
    """Compute comprehensive portfolio performance metrics.

    Parameters
    ----------
    returns : pd.Series  daily portfolio returns
    benchmark_returns : pd.Series  daily benchmark returns (e.g. SPY)
    risk_free_rate : float  annual risk-free rate (e.g. 0.05 = 5%)

    Returns
    -------
    dict of performance metrics
    """
    returns = returns.dropna()
    if len(returns) < 2:
        return {"error": "Insufficient return data"}

    rf_daily = risk_free_rate / 252

    # --- Return metrics ---
    total_return = float((1 + returns).prod() - 1)
    n_years = len(returns) / 252
    ann_return = float((1 + total_return) ** (1 / max(n_years, 0.01)) - 1)

    # --- Volatility ---
    ann_vol = float(returns.std() * math.sqrt(252))

    # --- Sharpe Ratio ---
    excess_returns = returns - rf_daily
    sharpe = float(excess_returns.mean() / excess_returns.std() * math.sqrt(252)) if excess_returns.std() > 0 else 0.0

    # --- Sortino Ratio (downside deviation) ---
    downside = returns[returns < rf_daily] - rf_daily
    downside_std = float(downside.std() * math.sqrt(252)) if len(downside) > 1 else 1e-9
    sortino = float(excess_returns.mean() * 252 / downside_std)

    # --- Calmar Ratio ---
    equity = (1 + returns).cumprod()
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = float(drawdown.min())
    calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else 0.0

    # --- Max drawdown duration ---
    in_dd = drawdown < 0
    dd_start = None
    max_dd_duration = 0
    current_duration = 0
    for is_dd in in_dd:
        if is_dd:
            current_duration += 1
            max_dd_duration = max(max_dd_duration, current_duration)
        else:
            current_duration = 0

    # --- Hit rate ---
    hit_rate = float((returns > 0).mean() * 100)

    # --- Skewness & Kurtosis ---
    skew = float(returns.skew())
    kurt = float(returns.kurtosis())

    metrics = {
        "total_return_pct": round(total_return * 100, 2),
        "annualised_return_pct": round(ann_return * 100, 2),
        "annualised_volatility_pct": round(ann_vol * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "max_drawdown_duration_days": max_dd_duration,
        "hit_rate_pct": round(hit_rate, 1),
        "return_skewness": round(skew, 3),
        "return_kurtosis": round(kurt, 3),
    }

    # --- Beta & Alpha vs benchmark ---
    if benchmark_returns is not None:
        bench = benchmark_returns.reindex(returns.index).dropna()
        common = returns.reindex(bench.index).dropna()
        bench = bench.reindex(common.index)
        if len(common) > 10:
            cov_matrix = np.cov(common.values, bench.values)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 1.0
            bench_ann = float((1 + bench).prod() ** (252 / len(bench)) - 1)
            alpha = ann_return - (risk_free_rate + beta * (bench_ann - risk_free_rate))
            metrics["beta"] = round(float(beta), 3)
            metrics["alpha_pct"] = round(float(alpha) * 100, 2)
            metrics["benchmark_return_pct"] = round(float(bench_ann) * 100, 2)
            metrics["correlation"] = round(float(np.corrcoef(common.values, bench.values)[0, 1]), 3)

    return metrics


def correlation_matrix(returns_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Compute return correlation matrix for a set of assets.

    Parameters
    ----------
    returns_dict : {ticker: daily_returns_series}

    Returns
    -------
    pd.DataFrame  correlation matrix
    """
    df = pd.DataFrame(returns_dict).dropna()
    return df.corr().round(3)


def concentration_risk(weights: dict[str, float]) -> dict:
    """Measure portfolio concentration using Herfindahl-Hirschman Index.

    HHI = sum(w_i^2). Higher = more concentrated.
    HHI > 0.25 = high concentration.

    Parameters
    ----------
    weights : {ticker: weight}  (weights should sum to 1)

    Returns
    -------
    dict with hhi, effective_n (1/HHI), top_holding, concentration_rating
    """
    w = np.array(list(weights.values()))
    w = w / w.sum()
    hhi = float((w ** 2).sum())
    effective_n = 1.0 / hhi if hhi > 0 else len(w)
    top_ticker = max(weights, key=lambda k: weights[k])

    if hhi > 0.25:
        rating = "High Concentration (Risky)"
    elif hhi > 0.15:
        rating = "Moderate Concentration"
    else:
        rating = "Well Diversified"

    return {
        "hhi": round(hhi, 4),
        "effective_n": round(effective_n, 1),
        "top_holding": top_ticker,
        "top_holding_weight_pct": round(float(weights[top_ticker]) * 100, 1),
        "concentration_rating": rating,
    }


def drawdown_analysis(equity_curve: pd.Series) -> pd.DataFrame:
    """Return a DataFrame of all drawdown periods with duration and depth."""
    rolling_max = equity_curve.cummax()
    dd = (equity_curve - rolling_max) / rolling_max

    records = []
    in_dd = False
    start = None
    peak = None

    for date, val in dd.items():
        if val < 0 and not in_dd:
            in_dd = True
            start = date
            peak = float(rolling_max[date])
        elif val == 0 and in_dd:
            in_dd = False
            depth = float(dd[start:date].min())
            records.append({
                "start": start,
                "end": date,
                "peak": round(peak, 2),
                "depth_pct": round(depth * 100, 2),
            })

    if records:
        df = pd.DataFrame(records)
        df["duration_days"] = (pd.to_datetime(df["end"]) - pd.to_datetime(df["start"])).dt.days
        return df.sort_values("depth_pct").head(10)
    return pd.DataFrame(columns=["start", "end", "peak", "depth_pct", "duration_days"])
