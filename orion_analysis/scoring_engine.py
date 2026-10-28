"""
scoring_engine.py

Purpose:
- Implement a rule-based, weighted scoring system that takes indicator
  values and outputs a numeric score and a recommendation (SELL/HOLD/BUY).

Core principle:
- 100% deterministic Python logic.
- No AI is used to make the decision.

Concepts introduced:
- Rule tables.
- Combining multiple conditions into a score.
- Returning structured results with explanation details.
"""

from typing import Dict, Any

import pandas as pd


def _get_latest_row(df: pd.DataFrame) -> pd.Series:
    """
    Helper to safely get the last row of a DataFrame.

    Parameters:
        df: Price + indicator DataFrame.

    Returns:
        Last row as a pandas Series.
    """
    return df.iloc[-1]


def _price_trend_score(latest: pd.Series) -> Dict[str, Any]:
    """
    Score based on price vs moving averages.

    Rules:
        - Price above SMA200 => +3
        - Price above SMA50  => +2
        - Price above SMA20  => +1

    Returns:
        Dict with 'score' and 'details' list.
    """
    score = 0
    details = []

    close = latest["Close"]

    if "SMA_200" in latest and pd.notna(latest["SMA_200"]):
        if close > latest["SMA_200"]:
            score += 3
            details.append("Price above SMA200 (+3)")
        else:
            details.append("Price below SMA200 (0)")

    if "SMA_50" in latest and pd.notna(latest["SMA_50"]):
        if close > latest["SMA_50"]:
            score += 2
            details.append("Price above SMA50 (+2)")
        else:
            details.append("Price below SMA50 (0)")

    if "SMA_20" in latest and pd.notna(latest["SMA_20"]):
        if close > latest["SMA_20"]:
            score += 1
            details.append("Price above SMA20 (+1)")
        else:
            details.append("Price below SMA20 (0)")

    return {"score": score, "details": details}


def _momentum_score(latest: pd.Series) -> Dict[str, Any]:
    """
    Score based on RSI and MACD.

    Rules:
        - RSI between 40 and 60 => +1 (healthy)
        - RSI > 70 or < 30 => -1 (overbought/oversold, caution)
        - MACD_HIST > 0 => +2 (bullish momentum)
        - MACD_HIST < 0 => -1 (bearish momentum)

    Returns:
        Dict with 'score' and 'details' list.
    """
    score = 0
    details = []

    rsi = latest.get("RSI", None)
    macd_hist = latest.get("MACD_HIST", None)

    if rsi is not None and pd.notna(rsi):
        if 40 <= rsi <= 60:
            score += 1
            details.append("RSI in healthy range 40-60 (+1)")
        elif rsi > 70 or rsi < 30:
            score -= 1
            details.append("RSI overbought/oversold (-1)")
        else:
            details.append("RSI neutral (0)")

    if macd_hist is not None and pd.notna(macd_hist):
        if macd_hist > 0:
            score += 2
            details.append("MACD histogram positive (+2)")
        elif macd_hist < 0:
            score -= 1
            details.append("MACD histogram negative (-1)")
        else:
            details.append("MACD histogram ~0 (0)")

    return {"score": score, "details": details}


def _volume_score(latest: pd.Series) -> Dict[str, Any]:
    """
    Score based on relative volume.

    Rules:
        - Relative volume > 1.5 => +2 (strong interest)
        - Relative volume between 1.2 and 1.5 => +1 (above average)
        - Relative volume < 0.8 => -1 (low interest)

    Returns:
        Dict with 'score' and 'details' list.
    """
    score = 0
    details = []

    rel_vol = latest.get("VOL_REL_20", None)

    if rel_vol is not None and pd.notna(rel_vol):
        if rel_vol > 1.5:
            score += 2
            details.append("Relative volume > 1.5 (+2)")
        elif 1.2 < rel_vol <= 1.5:
            score += 1
            details.append("Relative volume moderately high (+1)")
        elif rel_vol < 0.8:
            score -= 1
            details.append("Relative volume low (-1)")
        else:
            details.append("Relative volume normal (0)")

    return {"score": score, "details": details}


def _position_in_52w_range_score(latest: pd.Series) -> Dict[str, Any]:
    """
    Score based on where price sits between 52-week high and low.

    Rules (simple version):
        - Closer to 52-week high => +2
        - Middle of range        => 0
        - Close to 52-week low   => -2

    Returns:
        Dict with 'score' and 'details' list.
    """
    score = 0
    details = []

    close = latest["Close"]
    high_52w = latest.get("HIGH_52W", None)
    low_52w = latest.get("LOW_52W", None)

    if high_52w is None or low_52w is None:
        details.append("52-week high/low not available.")
        return {"score": score, "details": details}

    if not pd.notna(high_52w) or not pd.notna(low_52w):
        details.append("52-week high/low not available.")
        return {"score": score, "details": details}

    range_span = high_52w - low_52w
    if range_span <= 0:
        details.append("Invalid 52-week range.")
        return {"score": score, "details": details}

    # Position between 0 and 1 (0=low, 1=high)
    position = (close - low_52w) / range_span

    if position > 0.7:
        score += 2
        details.append("Price near 52-week high (+2)")
    elif position < 0.3:
        score -= 2
        details.append("Price near 52-week low (-2)")
    else:
        details.append("Price in middle of 52-week range (0)")

    return {"score": score, "details": details}


def _map_score_to_decision(total_score: int) -> str:
    """
    Convert numeric score to a textual decision.

    Example mapping (you can tweak later):
        score <= -2 => 'SELL'
        -1 to +2    => 'HOLD'
        >= +3       => 'BUY'
    """
    if total_score <= -2:
        return "SELL"
    elif total_score >= 3:
        return "BUY"
    else:
        return "HOLD"


def score_stock(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Main function to score a stock and produce a recommendation.

    Parameters:
        df: DataFrame with price + indicators (from indicators.add_all_indicators).

    Returns:
        Dict with:
            - 'total_score': int
            - 'decision': 'SELL' | 'HOLD' | 'BUY'
            - 'components': list of per-rule breakdowns
            - 'latest_date': date of the last data point
    """
    latest = _get_latest_row(df)

    # Compute scores from each group of rules
    trend = _price_trend_score(latest)
    momentum = _momentum_score(latest)
    volume = _volume_score(latest)
    pos_52w = _position_in_52w_range_score(latest)

    total_score = trend["score"] + momentum["score"] + volume["score"] + pos_52w["score"]
    decision = _map_score_to_decision(total_score)

    components = [
        {"name": "Price vs SMAs", "score": trend["score"], "details": trend["details"]},
        {"name": "Momentum (RSI, MACD)", "score": momentum["score"], "details": momentum["details"]},
        {"name": "Volume", "score": volume["score"], "details": volume["details"]},
        {"name": "52-week Range", "score": pos_52w["score"], "details": pos_52w["details"]},
    ]

    return {
        "total_score": int(total_score),
        "decision": decision,
        "components": components,
        "latest_date": latest.name,
    }