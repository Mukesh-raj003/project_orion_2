"""
sentiment_engine.py

Purpose:
- Fetch recent news headlines for a stock ticker.
- Analyze the sentiment of those headlines.
- Turn the result into a simple, beginner-friendly summary.

Concepts introduced:
- Working with APIs through Python libraries.
- Lists and dictionaries.
- Sentiment analysis using VADER.
- Defensive programming with error handling.

Why this exists:
- News sentiment gives ORION another signal to explain
  how market news may affect the stock.
"""

from typing import Dict, Any, List
import math

import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Create one reusable sentiment analyzer object.
# VADER is rule-based and works well on short headline-style text.[web:97][web:102]
SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()


def _safe_text(value: Any) -> str:
    """
    Convert a value to a safe string.

    Parameters:
        value: Any input value.

    Returns:
        A cleaned string.
    """
    if value is None:
        return ""
    return str(value).strip()


def fetch_news_items(ticker: str, max_headlines: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch recent news items (headline + source URL) for a ticker using yfinance.

    Parameters:
        ticker: Stock ticker symbol like 'AAPL'.
        max_headlines: Maximum number of items to return.

    Returns:
        A list of dicts with 'title' and 'url' keys.
    """
    items: List[Dict[str, Any]] = []

    try:
        stock = yf.Ticker(ticker)
        news_items = stock.news or []

        for item in news_items:
            # yfinance >= 0.2.x nests data under item["content"].
            content = item.get("content") or {}

            title = _safe_text(content.get("title") or item.get("title"))
            if not title:
                continue

            # Try canonical URL first, then clickThrough, then top-level link.
            canonical = content.get("canonicalUrl") or {}
            clickthrough = content.get("clickThroughUrl") or {}
            url = (
                _safe_text(canonical.get("url"))
                or _safe_text(clickthrough.get("url"))
                or _safe_text(item.get("link"))
            )

            items.append({"title": title, "url": url})

            if len(items) >= max_headlines:
                break

    except Exception:
        return []

    return items


def fetch_news_headlines(ticker: str, max_headlines: int = 10) -> List[str]:
    """
    Fetch recent news headlines for a ticker using yfinance.

    Parameters:
        ticker: Stock ticker symbol like 'AAPL'.
        max_headlines: Maximum number of headlines to return.

    Returns:
        A list of headline strings (backward-compat wrapper).
    """
    return [i["title"] for i in fetch_news_items(ticker, max_headlines)]


def analyze_headline_sentiment(headline: str) -> Dict[str, Any]:
    """
    Analyze the sentiment of one headline.

    Parameters:
        headline: A news headline string.

    Returns:
        Dictionary with compound, positive, neutral, negative, and label.
    """
    scores = SENTIMENT_ANALYZER.polarity_scores(headline)
    compound = scores["compound"]

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "headline": headline,
        "compound": compound,
        "positive": scores["pos"],
        "neutral": scores["neu"],
        "negative": scores["neg"],
        "label": label,
    }


def _summarize_overall_sentiment(avg_compound: float) -> str:
    """
    Convert average compound score to a simple sentiment label.

    Parameters:
        avg_compound: Average compound score across headlines.

    Returns:
        'positive', 'neutral', or 'negative'
    """
    if avg_compound >= 0.05:
        return "positive"
    elif avg_compound <= -0.05:
        return "negative"
    return "neutral"


def analyze_sentiment(ticker: str, max_headlines: int = 10) -> Dict[str, Any]:
    """
    Fetch news headlines and analyze overall sentiment.

    Parameters:
        ticker: Stock ticker symbol.
        max_headlines: Maximum number of headlines to analyze.

    Returns:
        Dictionary containing:
            - ticker
            - headline_count
            - headlines
            - headline_analysis
            - average_compound
            - sentiment_label
            - interpretation
    """
    news_items = fetch_news_items(ticker, max_headlines=max_headlines)
    headline_analysis: List[Dict[str, Any]] = []

    if not news_items:
        return {
            "ticker": ticker.upper(),
            "headline_count": 0,
            "headlines": [],
            "headline_analysis": [],
            "average_compound": 0.0,
            "sentiment_label": "neutral",
            "interpretation": "No recent news headlines were available for sentiment analysis.",
        }

    compound_scores: List[float] = []
    headlines: List[str] = []

    for news_item in news_items:
        headline = news_item["title"]
        url = news_item["url"]
        headlines.append(headline)
        result = analyze_headline_sentiment(headline)
        result["url"] = url          # attach source URL to each result
        headline_analysis.append(result)
        compound_scores.append(result["compound"])

    avg_compound = sum(compound_scores) / len(compound_scores)
    sentiment_label = _summarize_overall_sentiment(avg_compound)

    if sentiment_label == "positive":
        interpretation = (
            "Recent news sentiment is positive. This may support the stock's outlook "
            "because headlines are generally favorable."
        )
    elif sentiment_label == "negative":
        interpretation = (
            "Recent news sentiment is negative. This may pressure the stock because "
            "the latest headlines are generally unfavorable."
        )
    else:
        interpretation = (
            "Recent news sentiment is neutral. News flow does not currently give a "
            "strong positive or negative push to the stock."
        )

    return {
        "ticker": ticker.upper(),
        "headline_count": len(headlines),
        "headlines": headlines,
        "headline_analysis": headline_analysis,
        "average_compound": round(avg_compound, 4),
        "sentiment_label": sentiment_label,
        "interpretation": interpretation,
    }


def build_sentiment_summary(sentiment_result: Dict[str, Any]) -> str:
    """
    Build a plain-language summary for the dashboard or AI explainer.

    Parameters:
        sentiment_result: Output from analyze_sentiment().

    Returns:
        A beginner-friendly summary string.
    """
    ticker = sentiment_result.get("ticker", "UNKNOWN")
    label = sentiment_result.get("sentiment_label", "neutral")
    avg = sentiment_result.get("average_compound", 0.0)
    count = sentiment_result.get("headline_count", 0)
    interpretation = sentiment_result.get("interpretation", "")

    return (
        f"Sentiment for {ticker}: {label} "
        f"(average compound score {avg}, from {count} headlines). "
        f"{interpretation}"
    )
    