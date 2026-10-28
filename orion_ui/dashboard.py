"""
dashboard.py

Purpose:
- Build the Streamlit dashboard UI for Project ORION.

Concepts introduced:
- Streamlit basics (widgets, layout, charts).
- Connecting UI inputs -> data -> indicators -> scoring -> risk -> explanation.
- Handling errors gracefully in the UI.

Why this exists:
- Separates the web UI code from the application logic so
  the rest of the project can be used in other contexts (e.g., CLI, tests).
"""

from typing import Optional

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from orion_data.market_data import download_stock_history
from orion_analysis.indicators import (
    add_all_indicators,
    add_sma,
    add_rsi,
    add_macd,
)
from orion_analysis.scoring_engine import score_stock
from orion_analysis.risk_engine import analyze_risk
from orion_ai.explanation_engine import explain_decision
from sentiment_engine import analyze_sentiment, build_sentiment_summary

from orion_utils.helpers import is_dataframe_valid
from orion_utils.validation import validate_ticker
from config import APP_TITLE


def _plot_price_with_sma(df: pd.DataFrame, ticker: str) -> None:
    """
    Plot closing price with SMA20 using Plotly.

    Parameters:
        df: DataFrame with 'Close' and 'SMA_20' columns.
        ticker: Ticker symbol for the chart title.
    """
    fig = go.Figure()

    # Price line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Close Price",
        )
    )

    # SMA20 line (if it exists)
    if "SMA_20" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["SMA_20"],
                mode="lines",
                name="SMA 20",
            )
        )

    fig.update_layout(
        title=f"{ticker} - Price and SMA20",
        xaxis_title="Date",
        yaxis_title="Price",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def _plot_rsi(df: pd.DataFrame) -> None:
    """
    Plot RSI indicator.

    Parameters:
        df: DataFrame with 'RSI' column.
    """
    if "RSI" not in df.columns:
        st.info("RSI not available.")
        return

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            mode="lines",
            name="RSI",
        )
    )

    # Add horizontal lines for 30 and 70 levels
    fig.add_hline(y=70, line=dict(color="red", dash="dash"))
    fig.add_hline(y=30, line=dict(color="green", dash="dash"))

    fig.update_layout(
        title="RSI (14)",
        xaxis_title="Date",
        yaxis_title="RSI",
        yaxis=dict(range=[0, 100]),
    )

    st.plotly_chart(fig, use_container_width=True)


def _plot_macd(df: pd.DataFrame) -> None:
    """
    Plot MACD line, signal line, and histogram.

    Parameters:
        df: DataFrame with 'MACD', 'MACD_SIGNAL', and 'MACD_HIST' columns.
    """
    missing_cols = [
        c for c in ["MACD", "MACD_SIGNAL", "MACD_HIST"] if c not in df.columns
    ]
    if missing_cols:
        st.info("MACD not available.")
        return

    fig = go.Figure()

    # MACD line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD"],
            mode="lines",
            name="MACD",
        )
    )

    # Signal line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["MACD_SIGNAL"],
            mode="lines",
            name="Signal",
        )
    )

    # Histogram (bar chart)
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df["MACD_HIST"],
            name="MACD Histogram",
            marker_color="grey",
            opacity=0.5,
        )
    )

    fig.update_layout(
        title="MACD",
        xaxis_title="Date",
        yaxis_title="Value",
        barmode="overlay",
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_scoring_and_risk(
    score_result: dict,
    risk_result: dict,
    explanation: str,
) -> None:
    """
    Render the scoring engine output, risk summary, and explanation
    in a beginner-friendly layout.
    """
    st.subheader("Decision Support Summary")
    st.write(f"Recommendation: **{score_result.get('decision', 'HOLD')}**")
    st.write(f"Total Score: {score_result.get('total_score', 0)}")

    # Show component breakdown
    components = score_result.get("components", [])
    with st.expander("Score breakdown (rules used)"):
        for comp in components:
            st.markdown(f"**{comp.get('name', 'Component')}** – score {comp.get('score', 0)}")
            for d in comp.get("details", []):
                st.write(f"- {d}")

    st.subheader("Risk Summary")
    # Pretty JSON display of the risk dict (volatility, drawdown, regime, etc.).[web:61]
    st.json(risk_result)

    st.subheader("Explanation")
    st.write(explanation)


def _render_sentiment_section(ticker: str) -> None:
    """
    Fetch and display news sentiment for a ticker.

    Parameters:
        ticker: Stock ticker symbol.
    """
    st.subheader("📰 News Sentiment Analysis")

    with st.spinner("Fetching recent news headlines..."):
        sentiment_result = analyze_sentiment(ticker, max_headlines=10)

    label = sentiment_result.get("sentiment_label", "neutral")
    avg = sentiment_result.get("average_compound", 0.0)
    count = sentiment_result.get("headline_count", 0)
    interpretation = sentiment_result.get("interpretation", "")
    headline_analysis = sentiment_result.get("headline_analysis", [])

    # Colour-coded metric
    label_colour = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}
    icon = label_colour.get(label, "🟡")

    col1, col2, col3 = st.columns(3)
    col1.metric("Sentiment", f"{icon} {label.capitalize()}")
    col2.metric("Avg Compound Score", f"{avg:.4f}")
    col3.metric("Headlines Analysed", count)

    st.info(interpretation)

    if headline_analysis:
        with st.expander("View individual headline scores"):
            for item in headline_analysis:
                hl_label = item.get("label", "neutral")
                hl_icon = label_colour.get(hl_label, "🟡")
                url = item.get("url", "")
                title = item.get("headline", "")
                compound = item.get("compound", 0.0)

                if url:
                    st.markdown(
                        f"{hl_icon} **[{title}]({url})**  "
                        f"(compound: `{compound:.4f}`)"
                    )
                else:
                    st.markdown(
                        f"{hl_icon} **{title}**  "
                        f"(compound: `{compound:.4f}`)"
                    )
    else:
        st.warning("No recent headlines found for this ticker.")


def show_dashboard() -> None:
    """
    Main function to render the Streamlit dashboard.

    Steps:
    1) Ask user for ticker.
    2) Validate ticker.
    3) Download historical data.
    4) Compute indicators (SMA20/50/200, RSI, MACD, volume, 52-week H/L).
    5) Compute rule-based score and risk.
    6) Generate explanation (optionally using Gemini).
    7) Display charts + decision + risk + explanation.
    """
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.title(APP_TITLE)
    st.caption(
        "Phase 2–5: Technical indicators + rule-based scoring + risk summary + explanation. "
        "This is an educational project, not financial advice."
    )

    # Sidebar inputs
    st.sidebar.header("Settings")
    ticker_input = st.sidebar.text_input(
        "Enter stock ticker (e.g., AAPL, MSFT):",
        value="AAPL",
    )

    use_gemini = st.sidebar.checkbox(
        "Use Gemini for explanation (requires GEMINI_API_KEY in .env)",
        value=True,
    )

    if st.sidebar.button("Run Analysis"):
        is_valid, msg = validate_ticker(ticker_input)
        if not is_valid:
            st.error(msg)
            return

        ticker = ticker_input.strip().upper()
        with st.spinner(f"Downloading data for {ticker}..."):
            df = download_stock_history(ticker)

        if not is_dataframe_valid(df):
            st.error(
                "No data returned. Please check the ticker or try a different one."
            )
            return

        # Compute all indicators (SMA20/50/200, RSI, MACD, volume, 52-week H/L)
        df = add_all_indicators(df)

        st.subheader(f"{ticker} – Price and Indicators")

        # Top row: price + SMA
        _plot_price_with_sma(df, ticker)

        # Second row: RSI and MACD
        col1, col2 = st.columns(2)
        with col1:
            _plot_rsi(df)
        with col2:
            _plot_macd(df)

        # Scoring + risk + explanation
        score_result = score_stock(df)
        risk_result = analyze_risk(df)
        explanation = explain_decision(
            score_result,
            risk_result,
            use_gemini=use_gemini,
        )

        _render_scoring_and_risk(score_result, risk_result, explanation)

        # Sentiment Analysis Section
        _render_sentiment_section(ticker)

        # Show raw data at the bottom (optional)
        with st.expander("Show raw data (last 50 rows)"):
            st.dataframe(df.tail(50))
    else:
        st.info("Enter a ticker in the sidebar and click 'Run Analysis' to begin.")