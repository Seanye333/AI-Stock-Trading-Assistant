"""
AI Stock Trading Assistant — Streamlit Dashboard

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import sys
import os

# Ensure project root is in sys.path (handles both local and Streamlit Cloud)
_here = os.path.abspath(__file__)
for _candidate in [
    os.path.dirname(os.path.dirname(os.path.dirname(_here))),  # src/dashboard/app.py → root
    os.path.dirname(os.path.dirname(_here)),                   # fallback
]:
    if os.path.isfile(os.path.join(_candidate, "requirements.txt")):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        break

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from config.settings import (
    DEFAULT_TICKERS, DEFAULT_PERIOD, INITIAL_CAPITAL,
    ANTHROPIC_API_KEY, CLAUDE_MODEL,
)
from src.data.fetcher import fetch_ohlcv, fetch_fundamentals, fetch_multiple
from src.indicators.technical import add_all_indicators, get_signal_summary
from src.indicators.fundamental import score_fundamentals, format_fundamental_summary
from src.ai.ml_models import get_or_train_model, clear_model_cache
from src.ai.signal_generator import generate_signal
from src.backtesting.engine import run_backtest
from src.backtesting.monte_carlo import run_monte_carlo, portfolio_var
from src.risk.portfolio_manager import (
    fixed_fraction_position_size, kelly_position_size,
    compute_portfolio_metrics, correlation_matrix, concentration_risk,
)
from src.dashboard.components.charts import (
    candlestick_chart, rsi_chart, equity_curve_chart,
    monte_carlo_chart, correlation_heatmap, feature_importance_chart,
)


# ============================================================
# Page Config
# ============================================================

st.set_page_config(
    page_title="AI Stock Trading Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e2e;
        border-radius: 8px;
        padding: 16px;
        margin: 4px;
        border: 1px solid #333;
    }
    .signal-buy { color: #26a69a; font-size: 2em; font-weight: bold; }
    .signal-sell { color: #ef5350; font-size: 2em; font-weight: bold; }
    .signal-hold { color: #ffa726; font-size: 2em; font-weight: bold; }
    .stProgress > div > div > div { background-color: #2196F3; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.title("📈 AI Stock Trading Assistant")
    st.markdown("---")

    ticker = st.text_input("Ticker Symbol", value="AAPL", placeholder="AAPL, MSFT, NVDA...").upper().strip()
    period = st.selectbox("Data Period", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
    interval = st.selectbox("Interval", ["1d", "1wk"], index=0)

    st.markdown("---")
    st.subheader("Analysis Options")
    use_llm = st.toggle(
        "Claude AI Analysis",
        value=bool(ANTHROPIC_API_KEY),
        help="Requires ANTHROPIC_API_KEY in .env",
        disabled=not bool(ANTHROPIC_API_KEY),
    )
    extra_context = st.text_area(
        "Extra Context (optional)",
        placeholder="e.g. 'Fed meeting next week, company reports earnings on Friday'",
        height=80,
    )

    st.markdown("---")
    st.subheader("Backtest Settings")
    bt_capital = st.number_input("Initial Capital ($)", value=100_000, step=10_000)
    bt_stop_loss = st.slider("Stop Loss %", 2, 20, 7) / 100
    bt_take_profit = st.slider("Take Profit %", 5, 50, 20) / 100
    bt_commission = st.slider("Commission (bps)", 0, 20, 10) / 10_000

    st.markdown("---")
    st.subheader("Monte Carlo")
    mc_sims = st.slider("Simulations", 100, 2000, 500, step=100)
    mc_days = st.slider("Forecast Days", 30, 504, 126)
    mc_method = st.radio("Method", ["gbm", "bootstrap"], horizontal=True)

    st.markdown("---")
    st.subheader("Portfolio Watch")
    watch_input = st.text_input("Watchlist (comma-separated)", value="AAPL,MSFT,GOOGL,NVDA,AMZN")
    watch_tickers = [t.strip().upper() for t in watch_input.split(",") if t.strip()]

    run_btn = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)

if not ticker:
    st.info("Enter a ticker symbol in the sidebar and click **Run Full Analysis**.")
    st.stop()


# ============================================================
# Session State
# ============================================================

if "results" not in st.session_state:
    st.session_state.results = {}


# ============================================================
# Data Loading + Caching
# ============================================================

def _fmt_num(val) -> str:
    if val is None:
        return "N/A"
    if abs(val) >= 1e12:
        return f"${val/1e12:.2f}T"
    if abs(val) >= 1e9:
        return f"${val/1e9:.2f}B"
    if abs(val) >= 1e6:
        return f"${val/1e6:.2f}M"
    return f"${val:,.0f}"


def _pct(val) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


@st.cache_data(ttl=300, show_spinner=False)
def load_data(ticker: str, period: str, interval: str):
    return fetch_ohlcv(ticker, period=period, interval=interval)


@st.cache_data(ttl=3600, show_spinner=False)
def load_fundamentals(ticker: str):
    return fetch_fundamentals(ticker)


# ============================================================
# Main Panel — Tabs
# ============================================================

tab_overview, tab_technical, tab_ai, tab_backtest, tab_monte_carlo, tab_portfolio = st.tabs([
    "📊 Overview",
    "📉 Technical",
    "🤖 AI Analysis",
    "⚙️ Backtest",
    "🎲 Monte Carlo",
    "💼 Portfolio",
])


# ============================================================
# Load Data
# ============================================================

with st.spinner(f"Fetching data for {ticker}..."):
    try:
        df_raw = load_data(ticker, period, interval)
        df = add_all_indicators(df_raw)
        fundamentals = load_fundamentals(ticker)
        fund_scores = score_fundamentals(fundamentals)
    except Exception as e:
        st.error(f"Failed to load data for **{ticker}**: {e}")
        st.stop()

latest = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else latest
day_change = float(latest["Close"] - prev["Close"])
day_change_pct = float(day_change / prev["Close"] * 100)


# ============================================================
# TAB 1: Overview
# ============================================================

with tab_overview:
    st.header(f"{fundamentals.get('company', ticker)} ({ticker})")
    st.caption(f"{fundamentals.get('sector', '')} | {fundamentals.get('industry', '')}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Price", f"${latest['Close']:.2f}", f"{day_change:+.2f} ({day_change_pct:+.2f}%)")
    col2.metric("52W High", f"${df['High'].max():.2f}")
    col3.metric("52W Low", f"${df['Low'].min():.2f}")
    col4.metric("Market Cap", _fmt_num(fundamentals.get("market_cap")))
    col5.metric("Fundamental Score", f"{fund_scores['composite']}/100", fund_scores['rating'])

    st.markdown("---")

    # Price chart
    st.subheader("Price Chart")
    show_bb = st.checkbox("Bollinger Bands", value=True)
    show_ema = st.checkbox("Moving Averages", value=True)
    st.plotly_chart(
        candlestick_chart(df, ticker, show_bb=show_bb, show_ema=show_ema),
        use_container_width=True,
    )

    # Key stats
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Key Metrics")
        km = {
            "P/E Ratio": fundamentals.get("pe_ratio", "N/A"),
            "Forward P/E": fundamentals.get("forward_pe", "N/A"),
            "EPS (TTM)": fundamentals.get("eps", "N/A"),
            "Revenue Growth": _pct(fundamentals.get("revenue_growth")),
            "Net Margin": _pct(fundamentals.get("net_margin")),
            "ROE": _pct(fundamentals.get("roe")),
            "Debt/Equity": fundamentals.get("debt_to_equity", "N/A"),
            "Beta": fundamentals.get("beta", "N/A"),
            "Analyst Rating": fundamentals.get("analyst_recommendation", "N/A"),
        }
        st.table(pd.DataFrame(km.items(), columns=["Metric", "Value"]).set_index("Metric"))

    with col_b:
        st.subheader("Fundamental Breakdown")
        scores = fund_scores["scores"]
        for metric, score in scores.items():
            label = metric.replace("_", " ").title()
            color = "green" if score >= 65 else ("orange" if score >= 40 else "red")
            st.write(f"**{label}**")
            st.progress(int(score), text=f"{score:.0f}/100")


# ============================================================
# TAB 2: Technical Analysis
# ============================================================

with tab_technical:
    st.header("Technical Indicators")

    col1, col2, col3, col4 = st.columns(4)
    rsi_val = float(latest.get("RSI", 50))
    col1.metric("RSI (14)", f"{rsi_val:.1f}",
                "Oversold" if rsi_val < 30 else ("Overbought" if rsi_val > 70 else "Neutral"))

    macd_val = float(latest.get("MACD", 0))
    macd_sig_val = float(latest.get("MACD_signal", 0))
    col2.metric("MACD", f"{macd_val:.3f}",
                "Bullish" if macd_val > macd_sig_val else "Bearish")

    bb_pct = float(latest.get("BB_pct", 0.5))
    col3.metric("BB %", f"{bb_pct:.2f}",
                "Upper Band" if bb_pct > 0.8 else ("Lower Band" if bb_pct < 0.2 else "Mid Range"))

    adx_val = float(latest.get("ADX", 0))
    col4.metric("ADX (14)", f"{adx_val:.1f}",
                "Strong Trend" if adx_val > 25 else "Weak Trend")

    st.plotly_chart(rsi_chart(df), use_container_width=True)

    # Signal summary table
    st.subheader("Signal Summary")
    signals = get_signal_summary(df)
    rows = []
    for ind, data in signals.items():
        sig = data.get("signal", "N/A")
        val = data.get("value", data.get("bb_pct", data.get("golden_cross", "N/A")))
        emoji = "🟢" if "Bull" in sig or "Oversold" in sig or "Above" in sig or "Golden" in sig else (
                 "🔴" if "Bear" in sig or "Overbought" in sig or "Below" in sig or "Death" in sig else "🟡")
        rows.append({"Indicator": ind, "Value": val, "Signal": f"{emoji} {sig}"})
    st.dataframe(pd.DataFrame(rows).set_index("Indicator"), use_container_width=True)

    # ML Feature Importance
    st.subheader("ML Feature Importance")
    try:
        model, metrics = get_or_train_model(ticker, df)
        if model.is_trained:
            fi = model.get_feature_importance()
            if not fi.empty:
                st.plotly_chart(feature_importance_chart(fi), use_container_width=True)
                ml_pred = model.predict_latest(df)
                c1, c2, c3 = st.columns(3)
                c1.metric("ML Signal", ml_pred.get("signal", "N/A"))
                c2.metric("Prob Up", f"{ml_pred.get('probability_up', 0.5)*100:.1f}%")
                c3.metric("Model Accuracy", f"{ml_pred.get('model_accuracy', 0)*100:.1f}%")
    except Exception as e:
        st.warning(f"ML model: {e}")


# ============================================================
# TAB 3: AI Analysis
# ============================================================

with tab_ai:
    st.header("Claude AI Investment Analysis")

    if not ANTHROPIC_API_KEY:
        st.warning(
            "No `ANTHROPIC_API_KEY` set. Copy `.env.example` to `.env` and add your key, "
            "then restart the app."
        )

    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_ai = st.button("🤖 Run Claude Analysis", type="primary", disabled=not ANTHROPIC_API_KEY)
    with col_info:
        st.caption(f"Uses `{CLAUDE_MODEL}` with adaptive thinking for deep reasoning.")

    if run_ai or (run_btn and use_llm):
        with st.spinner("Claude is analysing the stock... (this may take 15–30 seconds)"):
            try:
                result = generate_signal(ticker, df_raw, extra_context=extra_context, use_llm=True)
                st.session_state.results[ticker] = result
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                result = None
    else:
        result = st.session_state.results.get(ticker)

    if result:
        # Signal header
        sig = result.get("signal", "HOLD")
        conf = result.get("confidence", 0)
        sig_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}.get(sig, "⚪")

        st.markdown(f"## {sig_color} Signal: **{sig}** (Confidence: {conf}%)")
        st.progress(conf)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Price Target", f"{result.get('price_target_pct', 0):+.1f}%")
        col2.metric("Stop Loss", f"-{result.get('stop_loss_pct', 7):.1f}%")
        col3.metric("Take Profit", f"+{result.get('take_profit_pct', 15):.1f}%")
        col4.metric("Fundamental Score", f"{result.get('fundamental_score', 0):.0f}/100")

        st.markdown("---")

        col_llm, col_ml = st.columns(2)
        with col_llm:
            st.subheader("Claude LLM")
            st.info(f"**Signal:** {result.get('llm_signal', 'N/A')}  |  **Confidence:** {result.get('llm_confidence', 0)}%")
            st.write("**Thesis:**", result.get("llm_thesis", ""))

            if result.get("llm_strengths"):
                st.write("**Key Strengths:**")
                for s in result["llm_strengths"]:
                    st.write(f"✅ {s}")

            if result.get("llm_risks"):
                st.write("**Key Risks:**")
                for r in result["llm_risks"]:
                    st.write(f"⚠️ {r}")

        with col_ml:
            st.subheader("ML Ensemble (XGBoost + LightGBM)")
            ml_sig = result.get("ml_signal", "HOLD")
            ml_emoji = "🟢" if ml_sig == "BUY" else ("🔴" if ml_sig == "SELL" else "🟡")
            st.info(f"**Signal:** {ml_emoji} {ml_sig}  |  **Prob Up:** {result.get('ml_prob_up', 0.5)*100:.1f}%")

            # Technical
            st.subheader("Technical Score")
            tech_score = result.get("technical_score", 50)
            st.progress(int(tech_score), text=f"{tech_score}/100")

            tech_sigs = result.get("technical_signals", {})
            for ind, sig_txt in tech_sigs.items():
                emoji = "🟢" if "+" in sig_txt else "🔴"
                st.write(f"{emoji} **{ind}:** {sig_txt}")

        # Position Sizing
        st.markdown("---")
        st.subheader("Position Sizing")
        price = result.get("current_price", float(latest["Close"]))
        pos = fixed_fraction_position_size(
            capital=INITIAL_CAPITAL,
            price=price,
            risk_pct=0.02,
            stop_loss_pct=result.get("stop_loss_pct", 7) / 100,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Suggested Shares", f"{pos['shares']:,}")
        c2.metric("Position Value", f"${pos['position_value']:,.0f}")
        c3.metric("% of Portfolio", f"{pos['position_pct']:.1f}%")
        c4.metric("Stop Loss Price", f"${pos['stop_loss_price']:.2f}")
    else:
        st.info("Click **Run Claude Analysis** to generate an AI-powered investment thesis.")


# ============================================================
# TAB 4: Backtest
# ============================================================

with tab_backtest:
    st.header("Strategy Backtesting")
    st.caption("Rule-based strategy: RSI + MACD + Moving Average crossover with configurable stop-loss/take-profit.")

    run_backtest_btn = st.button("▶️ Run Backtest", type="primary")

    if run_backtest_btn or run_btn:
        with st.spinner("Running backtest..."):
            bt = run_backtest(
                ticker, df_raw,
                initial_capital=bt_capital,
                commission=bt_commission,
                stop_loss_pct=bt_stop_loss,
                take_profit_pct=bt_take_profit,
            )

        # Metrics row
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Return", f"{bt.total_return_pct:+.1f}%",
                  f"vs B&H: {bt.benchmark_return_pct:+.1f}%")
        c2.metric("Ann. Return", f"{bt.annualised_return_pct:+.1f}%")
        c3.metric("Sharpe", f"{bt.sharpe_ratio:.2f}")
        c4.metric("Max DD", f"{bt.max_drawdown_pct:.1f}%")
        c5.metric("Win Rate", f"{bt.win_rate:.1f}%")
        c6.metric("# Trades", f"{bt.total_trades}")

        # Equity curve
        equity = bt.equity_curve
        bh_equity = (df_raw["Close"] / df_raw["Close"].iloc[0]) * bt_capital
        bh_equity = bh_equity.reindex(equity.index, method="ffill")

        st.plotly_chart(
            equity_curve_chart(equity, ticker, bh_equity, bt_capital),
            use_container_width=True,
        )

        # Trade log
        if bt.trades:
            st.subheader("Trade Log")
            trade_rows = [
                {
                    "Entry": t.entry_date[:10],
                    "Exit": t.exit_date[:10] if t.exit_date else "Open",
                    "Direction": t.direction,
                    "Entry Price": f"${t.entry_price:.2f}",
                    "Exit Price": f"${t.exit_price:.2f}" if t.exit_price else "—",
                    "P&L": f"${t.pnl:+.2f}",
                    "P&L %": f"{t.pnl_pct:+.2f}%",
                    "Exit Reason": t.exit_reason,
                }
                for t in bt.trades
            ]
            st.dataframe(pd.DataFrame(trade_rows), use_container_width=True, height=300)

        # Additional stats
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Performance Stats")
            stats = {
                "Final Portfolio Value": f"${bt.final_value:,.2f}",
                "Profit Factor": f"{bt.profit_factor:.2f}",
                "Avg Win": f"{bt.avg_win_pct:+.2f}%",
                "Avg Loss": f"{bt.avg_loss_pct:+.2f}%",
                "Winning Trades": bt.winning_trades,
                "Losing Trades": bt.losing_trades,
            }
            st.table(pd.DataFrame(stats.items(), columns=["Metric", "Value"]).set_index("Metric"))


# ============================================================
# TAB 5: Monte Carlo
# ============================================================

with tab_monte_carlo:
    st.header("Monte Carlo Simulation")

    run_mc_btn = st.button("🎲 Run Monte Carlo", type="primary")

    if run_mc_btn or run_btn:
        with st.spinner(f"Running {mc_sims:,} simulations over {mc_days} days..."):
            mc_result = run_monte_carlo(
                df["Close"],
                n_simulations=mc_sims,
                n_days=mc_days,
                method=mc_method,
            )

        st.plotly_chart(monte_carlo_chart(mc_result), use_container_width=True)

        m = mc_result["metrics"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Median Return", f"{m['median_return_pct']:+.1f}%")
        col2.metric("Prob. Positive", f"{m['prob_positive']:.1f}%")
        col3.metric("VaR (95%)", f"{m['var_95_pct']:.1f}%")
        col4.metric("CVaR (95%)", f"{m['cvar_95_pct']:.1f}%")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Median Final Price", f"${m['median_final_price']:.2f}")
        col6.metric("Best Case (95th)", f"{m['best_case_pct']:+.1f}%")
        col7.metric("Worst Case (5th)", f"{m['worst_case_pct']:+.1f}%")
        col8.metric("Prob Loss >10%", f"{m['prob_loss_10pct']:.1f}%")

        with st.expander("Full Metrics"):
            st.json(m)


# ============================================================
# TAB 6: Portfolio
# ============================================================

with tab_portfolio:
    st.header("Portfolio Analysis")

    with st.spinner(f"Loading portfolio data for {', '.join(watch_tickers)}..."):
        portfolio_data = {}
        for t in watch_tickers:
            try:
                portfolio_data[t] = fetch_ohlcv(t, period="1y", interval="1d")
            except Exception:
                pass

    if len(portfolio_data) < 2:
        st.warning("Need at least 2 valid tickers in the watchlist.")
    else:
        # Returns
        returns_dict = {t: df["Close"].pct_change().dropna() for t, df in portfolio_data.items()}
        returns_df = pd.DataFrame(returns_dict).dropna()

        # Correlation matrix
        corr_df = correlation_matrix(returns_dict)
        st.plotly_chart(correlation_heatmap(corr_df), use_container_width=True)

        # Equal-weight portfolio metrics
        st.subheader("Equal-Weight Portfolio Metrics")
        try:
            port_returns = returns_df.mean(axis=1)
            metrics = compute_portfolio_metrics(port_returns)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Ann. Return", f"{metrics.get('annualised_return_pct', 0):.1f}%")
            col2.metric("Volatility", f"{metrics.get('annualised_volatility_pct', 0):.1f}%")
            col3.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.2f}")
            col4.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.1f}%")
        except Exception as e:
            st.warning(f"Portfolio metrics: {e}")

        # Portfolio VaR
        st.subheader("Portfolio Risk (Equal Weight)")
        try:
            var_result = portfolio_var(returns_df)
            c1, c2, c3 = st.columns(3)
            c1.metric("1-Day VaR (95%)", f"{var_result['var_pct']:.2f}%")
            c2.metric("1-Day CVaR (95%)", f"{var_result['cvar_pct']:.2f}%")
            c3.metric("Annual Volatility", f"{var_result['portfolio_vol_annual_pct']:.1f}%")
        except Exception as e:
            st.warning(f"VaR: {e}")

        # Concentration
        weights = {t: 1.0 / len(portfolio_data) for t in portfolio_data}
        conc = concentration_risk(weights)
        st.subheader("Concentration Risk")
        c1, c2, c3 = st.columns(3)
        c1.metric("HHI Score", f"{conc['hhi']:.4f}")
        c2.metric("Effective N", f"{conc['effective_n']:.1f}")
        c3.metric("Concentration", conc["concentration_rating"])

        # Watchlist performance table
        st.subheader("Watchlist Performance")
        rows = []
        for t, df_t in portfolio_data.items():
            ret_1m = (df_t["Close"].iloc[-1] / df_t["Close"].iloc[-21] - 1) * 100 if len(df_t) > 21 else 0
            ret_3m = (df_t["Close"].iloc[-1] / df_t["Close"].iloc[-63] - 1) * 100 if len(df_t) > 63 else 0
            ret_ytd = (df_t["Close"].iloc[-1] / df_t["Close"].iloc[0] - 1) * 100
            vol = df_t["Close"].pct_change().std() * (252 ** 0.5) * 100
            rows.append({
                "Ticker": t,
                "Price": f"${df_t['Close'].iloc[-1]:.2f}",
                "1M Return": f"{ret_1m:+.1f}%",
                "3M Return": f"{ret_3m:+.1f}%",
                "YTD Return": f"{ret_ytd:+.1f}%",
                "Ann. Vol": f"{vol:.1f}%",
            })
        st.dataframe(pd.DataFrame(rows).set_index("Ticker"), use_container_width=True)


