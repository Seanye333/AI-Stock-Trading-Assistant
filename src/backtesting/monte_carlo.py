"""
Monte Carlo simulation for portfolio/single-stock price path forecasting.

Uses bootstrapped daily returns to simulate N future price paths over T days.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import MC_SIMULATIONS, MC_DAYS


def run_monte_carlo(
    price_series: pd.Series,
    n_simulations: int = MC_SIMULATIONS,
    n_days: int = MC_DAYS,
    method: str = "gbm",    # 'gbm' (geometric Brownian motion) or 'bootstrap'
    confidence_intervals: list[float] = (0.05, 0.25, 0.50, 0.75, 0.95),
    random_seed: int = 42,
) -> dict:
    """Run Monte Carlo price path simulations.

    Parameters
    ----------
    price_series : pd.Series  historical close prices
    n_simulations : int  number of paths to simulate
    n_days : int  number of forward-looking trading days
    method : str  'gbm' for Geometric Brownian Motion or 'bootstrap' for historical bootstrap
    confidence_intervals : list[float]  percentiles to compute for fan chart
    random_seed : int

    Returns
    -------
    dict with:
        simulations : np.ndarray  shape (n_simulations, n_days)
        percentiles : dict  {pct_label: np.ndarray of shape (n_days,)}
        metrics : dict  summary statistics
        initial_price : float
    """
    rng = np.random.default_rng(random_seed)
    daily_returns = price_series.pct_change().dropna()
    initial_price = float(price_series.iloc[-1])

    mu = float(daily_returns.mean())
    sigma = float(daily_returns.std())

    simulations = np.zeros((n_simulations, n_days + 1))
    simulations[:, 0] = initial_price

    if method == "gbm":
        # Geometric Brownian Motion: dS = mu*S*dt + sigma*S*dW
        dt = 1  # daily
        drift = (mu - 0.5 * sigma ** 2) * dt
        for t in range(1, n_days + 1):
            Z = rng.standard_normal(n_simulations)
            simulations[:, t] = simulations[:, t - 1] * np.exp(drift + sigma * math.sqrt(dt) * Z)
    else:
        # Historical bootstrap: resample daily returns
        hist_returns = daily_returns.values
        for t in range(1, n_days + 1):
            sampled = rng.choice(hist_returns, size=n_simulations, replace=True)
            simulations[:, t] = simulations[:, t - 1] * (1 + sampled)

    # Final price distribution
    final_prices = simulations[:, -1]
    pct_changes = (final_prices - initial_price) / initial_price

    # Percentile paths
    pct_paths = {}
    for ci in confidence_intervals:
        label = f"p{int(ci*100)}"
        pct_paths[label] = np.percentile(simulations, ci * 100, axis=0)

    # Value at Risk
    var_95 = float(np.percentile(pct_changes, 5))
    cvar_95 = float(pct_changes[pct_changes <= var_95].mean())

    metrics = {
        "initial_price": round(initial_price, 2),
        "mean_final_price": round(float(np.mean(final_prices)), 2),
        "median_final_price": round(float(np.median(final_prices)), 2),
        "std_final_price": round(float(np.std(final_prices)), 2),
        "mean_return_pct": round(float(np.mean(pct_changes)) * 100, 2),
        "median_return_pct": round(float(np.median(pct_changes)) * 100, 2),
        "prob_positive": round(float(np.mean(pct_changes > 0)) * 100, 1),
        "prob_loss_10pct": round(float(np.mean(pct_changes < -0.10)) * 100, 1),
        "prob_gain_20pct": round(float(np.mean(pct_changes > 0.20)) * 100, 1),
        "var_95_pct": round(var_95 * 100, 2),      # 5th percentile loss
        "cvar_95_pct": round(cvar_95 * 100, 2),    # Expected shortfall
        "best_case_pct": round(float(np.percentile(pct_changes, 95)) * 100, 2),
        "worst_case_pct": round(float(np.percentile(pct_changes, 5)) * 100, 2),
        "n_simulations": n_simulations,
        "n_days": n_days,
        "method": method,
    }

    return {
        "simulations": simulations,
        "percentiles": pct_paths,
        "metrics": metrics,
        "initial_price": initial_price,
        "final_prices": final_prices,
    }


def portfolio_var(
    returns_df: pd.DataFrame,
    weights: Optional[np.ndarray] = None,
    confidence: float = 0.95,
    horizon_days: int = 1,
) -> dict:
    """Compute portfolio Value at Risk and CVaR (parametric method).

    Parameters
    ----------
    returns_df : pd.DataFrame  daily returns for each asset (columns = tickers)
    weights : np.ndarray  portfolio weights (equal if None)
    confidence : float  VaR confidence level
    horizon_days : int  VaR horizon in trading days

    Returns
    -------
    dict with var_pct, cvar_pct, portfolio_vol_annual
    """
    if weights is None:
        weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)

    weights = np.array(weights) / np.sum(np.abs(weights))
    cov_matrix = returns_df.cov().values

    port_variance = weights @ cov_matrix @ weights
    port_vol_daily = math.sqrt(port_variance)
    port_vol_annual = port_vol_daily * math.sqrt(252)

    port_mean_daily = float((returns_df.mean() * weights).sum())
    port_mean_horizon = port_mean_daily * horizon_days
    port_vol_horizon = port_vol_daily * math.sqrt(horizon_days)

    # Parametric VaR (assumes normality)
    from scipy import stats
    z = stats.norm.ppf(1 - confidence)
    var_pct = -(port_mean_horizon + z * port_vol_horizon) * 100

    # CVaR (expected shortfall)
    z_ci = stats.norm.ppf(1 - confidence)
    cvar_pct = -(port_mean_horizon - port_vol_horizon * stats.norm.pdf(z_ci) / (1 - confidence)) * 100

    return {
        "var_pct": round(float(var_pct), 3),
        "cvar_pct": round(float(cvar_pct), 3),
        "portfolio_vol_annual_pct": round(float(port_vol_annual) * 100, 2),
        "confidence": confidence,
        "horizon_days": horizon_days,
    }
