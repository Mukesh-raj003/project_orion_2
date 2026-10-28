"""
risk_engine.py

Purpose:
- Analyze risk characteristics of a stock:
  - Volatility
  - Trend strength
  - Max drawdown
  - Market regime (bull / bear / sideways)
  - Overall risk level

Concepts introduced:
- Daily returns.
- Standard deviation (volatility).
- Max drawdown.
- Simple classification rules.
"""

from typing import Dict, Any

import pandas as pd
import numpy as np


def _compute_daily_returns(df: pd.DataFrame) -> pd.Series:
    """
    Compute simple daily returns based on closing price.

    Return = (today_close / yesterday_close) - 1
    """
    return df["Close"].pct_change()


def _compute_volatility(returns: pd.Series) -> float:
    """
    Compute volatility as standard deviation of daily returns.

    Parameters:
        returns: Series of daily returns.

    Returns:
        Volatility as a float (e.g., 0.02 ~ 2%).
    """
    return float(returns.std())


def _compute_max_drawdown(df: pd.DataFrame) -> float:
    """
    Compute maximum drawdown from closing prices.

    Max drawdown is the worst peak-to-trough drop.

    Returns:
        Max drawdown as a negative number (e.g., -0.35 = -35%).
    """
    close = df["Close"]
    running_max = close.cummax()
    drawdown = (close / running_max) - 1.0
    return float(drawdown.min())


def _compute_trend_strength(df: pd.DataFrame) -> float:
    """
    Compute a simple trend strength measure.

    Here we use:
        trend_strength = (latest_close - SMA_200_latest) / SMA_200_latest

    Positive values mean price is above long-term average (bullish),
    negative below (bearish).

    Returns:
        trend strength as a float (e.g., 0.1 ~ 10% above SMA200).
    """
    latest = df.iloc[-1]
    close = latest["Close"]
    sma_200 = latest.get("SMA_200", np.nan)

    if pd.isna(sma_200) or sma_200 == 0:
        return 0.0

    return float((close - sma_200) / sma_200)


def _classify_market_regime(trend_strength: float) -> str:
    """
    Classify market regime based on trend strength.

    Very simple rules:
        trend_strength > +0.05 => 'bull'
        trend_strength < -0.05 => 'bear'
        otherwise              => 'sideways'
    """
    if trend_strength > 0.05:
        return "bull"
    elif trend_strength < -0.05:
        return "bear"
    else:
        return "sideways"


def _classify_risk_level(volatility: float, max_drawdown: float) -> str:
    """
    Classify overall risk level based on volatility and drawdown.

    Example simple rules:
        - If volatility < 0.015 and drawdown > -0.2 => 'low'
        - Else if volatility < 0.03 and drawdown > -0.4 => 'medium'
        - Else => 'high'

    Volatility is std dev of daily returns.
    Max drawdown is negative (e.g., -0.35).
    """
    if volatility < 0.015 and max_drawdown > -0.20:
        return "low"
    elif volatility < 0.03 and max_drawdown > -0.40:
        return "medium"
    else:
        return "high"


def analyze_risk(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Main function to compute risk metrics.

    Parameters:
        df: Historical price DataFrame with at least 'Close' and SMA_200.

    Returns:
        Dict with:
            - 'volatility': float
            - 'max_drawdown': float
            - 'trend_strength': float
            - 'market_regime': 'bull' | 'bear' | 'sideways'
            - 'risk_level': 'low' | 'medium' | 'high'
    """
    returns = _compute_daily_returns(df)

    # Drop initial NaN from pct_change
    clean_returns = returns.dropna()
    if clean_returns.empty:
        # Fallback: no data -> neutral values
        volatility = 0.0
    else:
        volatility = _compute_volatility(clean_returns)

    max_dd = _compute_max_drawdown(df)
    trend_strength = _compute_trend_strength(df)
    regime = _classify_market_regime(trend_strength)
    risk_level = _classify_risk_level(volatility, max_dd)

    return {
        "volatility": volatility,
        "max_drawdown": max_dd,
        "trend_strength": trend_strength,
        "market_regime": regime,
        "risk_level": risk_level,
    }