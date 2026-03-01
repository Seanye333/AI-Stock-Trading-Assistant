"""
Plotly chart components for the Streamlit dashboard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def candlestick_chart(
    df: pd.DataFrame,
    ticker: str,
    show_bb: bool = True,
    show_ema: bool = True,
    show_volume: bool = True,
) -> go.Figure:
    """Full candlestick chart with optional BB, EMA, and volume."""
    rows = 3 if show_volume else 2
    row_heights = [0.55, 0.25, 0.20] if show_volume else [0.65, 0.35]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=[ticker, "MACD", "Volume" if show_volume else None],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Price", showlegend=False,
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Bollinger Bands
    if show_bb and "BB_upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_upper"], name="BB Upper",
            line=dict(color="rgba(100,100,255,0.4)", dash="dot"), showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_lower"], name="BB Lower",
            line=dict(color="rgba(100,100,255,0.4)", dash="dot"),
            fill="tonexty", fillcolor="rgba(100,100,255,0.05)", showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_mid"], name="BB Mid",
            line=dict(color="rgba(100,100,255,0.6)", dash="dash"), showlegend=False,
        ), row=1, col=1)

    # EMAs
    if show_ema:
        if "EMA_21" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["EMA_21"], name="EMA 21",
                line=dict(color="#ff9800", width=1.5),
            ), row=1, col=1)
        if "SMA_50" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["SMA_50"], name="SMA 50",
                line=dict(color="#2196F3", width=1.5),
            ), row=1, col=1)
        if "SMA_200" in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["SMA_200"], name="SMA 200",
                line=dict(color="#9C27B0", width=1.5),
            ), row=1, col=1)

    # VWAP
    if "VWAP" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VWAP"], name="VWAP",
            line=dict(color="#4CAF50", width=1, dash="dashdot"),
        ), row=1, col=1)

    # MACD
    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD",
            line=dict(color="#2196F3"),
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_signal"], name="Signal",
            line=dict(color="#FF5722"),
        ), row=2, col=1)
        colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_hist"], name="Histogram",
            marker_color=colors, showlegend=False,
        ), row=2, col=1)

    # Volume
    if show_volume and rows == 3:
        vol_colors = [
            "#26a69a" if c >= o else "#ef5350"
            for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=vol_colors, showlegend=False,
        ), row=3, col=1)

    fig.update_layout(
        height=700, template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def rsi_chart(df: pd.DataFrame) -> go.Figure:
    """RSI chart with overbought/oversold lines."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"], name="RSI",
        line=dict(color="#7C4DFF", width=2),
    ))
    fig.add_hline(y=70, line=dict(color="#ef5350", dash="dash"), annotation_text="Overbought (70)")
    fig.add_hline(y=30, line=dict(color="#26a69a", dash="dash"), annotation_text="Oversold (30)")
    fig.add_hline(y=50, line=dict(color="gray", dash="dot"))
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.1)", line_width=0)
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(38,166,154,0.1)", line_width=0)
    fig.update_layout(
        height=250, template="plotly_dark",
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis=dict(range=[0, 100]),
        title="RSI (14)",
    )
    return fig


def equity_curve_chart(
    equity_series: pd.Series,
    ticker: str,
    benchmark_series: pd.Series = None,
    initial_capital: float = 100_000,
) -> go.Figure:
    """Equity curve vs buy-and-hold benchmark."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_series.index, y=equity_series,
        name="Strategy", line=dict(color="#2196F3", width=2),
        fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
    ))
    if benchmark_series is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_series.index, y=benchmark_series,
            name="Buy & Hold", line=dict(color="#FF9800", width=2, dash="dash"),
        ))
    fig.add_hline(y=initial_capital, line=dict(color="gray", dash="dot"))
    fig.update_layout(
        height=350, template="plotly_dark",
        title=f"Equity Curve — {ticker}",
        yaxis_title="Portfolio Value ($)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def monte_carlo_chart(mc_result: dict) -> go.Figure:
    """Fan chart of Monte Carlo price path percentiles."""
    p = mc_result["percentiles"]
    x = list(range(len(p["p50"])))
    init = mc_result["initial_price"]

    fig = go.Figure()

    # 5–95% range
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(p["p95"]) + list(p["p5"][::-1]),
        fill="toself", fillcolor="rgba(33,150,243,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="5–95% Range",
    ))
    # 25–75% range
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=list(p["p75"]) + list(p["p25"][::-1]),
        fill="toself", fillcolor="rgba(33,150,243,0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        name="25–75% Range",
    ))
    # Median
    fig.add_trace(go.Scatter(
        x=x, y=p["p50"], name="Median",
        line=dict(color="#2196F3", width=2),
    ))
    # Initial price reference
    fig.add_hline(y=init, line=dict(color="gray", dash="dot"),
                  annotation_text=f"Current: ${init:.2f}")

    fig.update_layout(
        height=400, template="plotly_dark",
        title=f"Monte Carlo Simulation ({mc_result['metrics']['n_simulations']:,} paths, {mc_result['metrics']['n_days']} days)",
        yaxis_title="Price ($)",
        xaxis_title="Trading Days",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def correlation_heatmap(corr_df: pd.DataFrame) -> go.Figure:
    """Plotly heatmap of asset return correlations."""
    fig = go.Figure(data=go.Heatmap(
        z=corr_df.values, x=corr_df.columns, y=corr_df.index,
        colorscale="RdYlGn", zmin=-1, zmax=1,
        text=corr_df.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 12},
    ))
    fig.update_layout(
        height=400, template="plotly_dark",
        title="Asset Correlation Matrix",
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of XGBoost feature importances."""
    top = importance_df.head(15)
    fig = go.Figure(go.Bar(
        x=top["importance"], y=top["feature"],
        orientation="h",
        marker=dict(
            color=top["importance"],
            colorscale="Viridis",
        ),
    ))
    fig.update_layout(
        height=400, template="plotly_dark",
        title="ML Feature Importance (XGBoost)",
        xaxis_title="Importance",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig
