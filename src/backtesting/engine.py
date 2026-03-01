"""
Backtesting Engine: simulates a simple strategy on historical data.

Strategy uses rule-based technical signals (RSI + MACD + MA crossover)
to enter/exit positions, with configurable stop-loss and take-profit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import INITIAL_CAPITAL, COMMISSION_RATE, STOP_LOSS_PCT, TAKE_PROFIT_PCT
from src.indicators.technical import add_all_indicators


@dataclass
class Trade:
    entry_date: str
    exit_date: Optional[str]
    direction: str          # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: Optional[float]
    shares: float
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = ""   # 'signal', 'stop_loss', 'take_profit', 'end'


@dataclass
class BacktestResult:
    ticker: str
    initial_capital: float
    final_value: float
    total_return_pct: float
    annualised_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win_pct: float
    avg_loss_pct: float
    profit_factor: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    benchmark_return_pct: float = 0.0   # Buy & Hold return


def run_backtest(
    ticker: str,
    df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    commission: float = COMMISSION_RATE,
    stop_loss_pct: float = STOP_LOSS_PCT,
    take_profit_pct: float = TAKE_PROFIT_PCT,
    allow_short: bool = False,
) -> BacktestResult:
    """Run a rule-based backtest on the provided OHLCV DataFrame.

    Entry logic:
      - BUY when RSI < 35 AND MACD > MACD_signal AND price > EMA_21
      - SELL (exit long / enter short) when RSI > 65 AND MACD < MACD_signal

    Parameters
    ----------
    ticker : str
    df : pd.DataFrame  raw OHLCV — indicators added internally
    initial_capital : float
    commission : float  per-side commission rate (e.g. 0.001 = 0.1%)
    stop_loss_pct : float  e.g. 0.07 = 7%
    take_profit_pct : float  e.g. 0.20 = 20%
    allow_short : bool  enable shorting on sell signals

    Returns
    -------
    BacktestResult
    """
    data = add_all_indicators(df).dropna(subset=["RSI", "MACD", "MACD_signal", "EMA_21"])
    data = data.reset_index()

    capital = float(initial_capital)
    position: Optional[str] = None   # 'LONG', 'SHORT', or None
    entry_price: float = 0.0
    shares: float = 0.0
    entry_date: str = ""

    equity_values = []
    equity_dates = []
    trades: list[Trade] = []

    for i, row in data.iterrows():
        date = str(row["Date"] if "Date" in row.index else row.index if hasattr(row, "name") else i)
        close = float(row["Close"])
        rsi = float(row["RSI"]) if not math.isnan(row["RSI"]) else 50
        macd = float(row["MACD"]) if not math.isnan(row["MACD"]) else 0
        macd_sig = float(row["MACD_signal"]) if not math.isnan(row["MACD_signal"]) else 0
        ema21 = float(row["EMA_21"]) if not math.isnan(row["EMA_21"]) else close

        # --- Manage open position ---
        if position == "LONG":
            current_value = capital + shares * close
            pnl_pct = (close - entry_price) / entry_price

            # Stop-loss
            if pnl_pct <= -stop_loss_pct:
                proceeds = shares * close * (1 - commission)
                capital += proceeds
                trade = Trade(
                    entry_date=entry_date, exit_date=date,
                    direction="LONG", entry_price=entry_price,
                    exit_price=close, shares=shares,
                    pnl=proceeds - shares * entry_price * (1 + commission),
                    pnl_pct=pnl_pct * 100, exit_reason="stop_loss",
                )
                trades.append(trade)
                shares = 0.0
                position = None
                continue

            # Take-profit
            if pnl_pct >= take_profit_pct:
                proceeds = shares * close * (1 - commission)
                capital += proceeds
                trade = Trade(
                    entry_date=entry_date, exit_date=date,
                    direction="LONG", entry_price=entry_price,
                    exit_price=close, shares=shares,
                    pnl=proceeds - shares * entry_price * (1 + commission),
                    pnl_pct=pnl_pct * 100, exit_reason="take_profit",
                )
                trades.append(trade)
                shares = 0.0
                position = None
                continue

        # --- Entry/exit signals ---
        buy_signal = rsi < 35 and macd > macd_sig and close > ema21
        sell_signal = rsi > 65 and macd < macd_sig

        if position is None:
            if buy_signal:
                cost = capital * 0.95  # Use 95% of capital
                shares = cost / (close * (1 + commission))
                capital -= shares * close * (1 + commission)
                entry_price = close
                entry_date = date
                position = "LONG"

        elif position == "LONG" and sell_signal:
            proceeds = shares * close * (1 - commission)
            pnl_pct = (close - entry_price) / entry_price
            trade = Trade(
                entry_date=entry_date, exit_date=date,
                direction="LONG", entry_price=entry_price,
                exit_price=close, shares=shares,
                pnl=proceeds - shares * entry_price * (1 + commission),
                pnl_pct=pnl_pct * 100, exit_reason="signal",
            )
            trades.append(trade)
            capital += proceeds
            shares = 0.0
            position = None

        # Track equity
        portfolio_value = capital + (shares * close if position else 0)
        equity_values.append(portfolio_value)
        equity_dates.append(row.get("Date", i))

    # Close any open position at end
    if position == "LONG" and shares > 0:
        close_last = float(data.iloc[-1]["Close"])
        proceeds = shares * close_last * (1 - commission)
        pnl_pct = (close_last - entry_price) / entry_price
        trades.append(Trade(
            entry_date=entry_date, exit_date=str(data.iloc[-1].get("Date", "")),
            direction="LONG", entry_price=entry_price, exit_price=close_last,
            shares=shares,
            pnl=proceeds - shares * entry_price * (1 + commission),
            pnl_pct=pnl_pct * 100, exit_reason="end",
        ))
        capital += proceeds

    equity_curve = pd.Series(equity_values, index=equity_dates)
    final_value = equity_curve.iloc[-1] if not equity_curve.empty else initial_capital

    # Metrics
    total_return = (final_value - initial_capital) / initial_capital * 100
    n_days = len(data)
    n_years = n_days / 252
    ann_return = ((final_value / initial_capital) ** (1 / max(n_years, 0.01)) - 1) * 100

    daily_returns = equity_curve.pct_change().dropna()
    sharpe = (
        (daily_returns.mean() / daily_returns.std() * math.sqrt(252))
        if daily_returns.std() > 0 else 0.0
    )

    # Max drawdown
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_dd = float(drawdown.min() * 100)

    # Trade stats
    completed = [t for t in trades if t.exit_price is not None]
    wins = [t for t in completed if t.pnl > 0]
    losses = [t for t in completed if t.pnl <= 0]
    win_rate = len(wins) / max(len(completed), 1) * 100
    avg_win = float(np.mean([t.pnl_pct for t in wins])) if wins else 0.0
    avg_loss = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0
    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / max(gross_loss, 1e-9)

    # Benchmark: buy & hold
    bh_return = (float(data.iloc[-1]["Close"]) - float(data.iloc[0]["Close"])) / float(data.iloc[0]["Close"]) * 100

    return BacktestResult(
        ticker=ticker,
        initial_capital=initial_capital,
        final_value=round(final_value, 2),
        total_return_pct=round(total_return, 2),
        annualised_return_pct=round(ann_return, 2),
        sharpe_ratio=round(sharpe, 3),
        max_drawdown_pct=round(max_dd, 2),
        win_rate=round(win_rate, 1),
        total_trades=len(completed),
        winning_trades=len(wins),
        losing_trades=len(losses),
        avg_win_pct=round(avg_win, 2),
        avg_loss_pct=round(avg_loss, 2),
        profit_factor=round(profit_factor, 3),
        trades=trades,
        equity_curve=equity_curve,
        benchmark_return_pct=round(bh_return, 2),
    )
