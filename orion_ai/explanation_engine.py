"""
explanation_engine.py

Purpose:
- Take the rule-based decision (score + BUY/HOLD/SELL) and risk summary,
  and generate a human-readable explanation that tells the user *why*
  the system reached that recommendation.

Critical rule:
- This module MUST NOT change the decision.
- It only explains the decision.

Concepts introduced:
- Using environment variables (.env) for API keys.
- Optional integration with Gemini API for natural language generation.
"""

from typing import Dict, Any, List
import os

from dotenv import load_dotenv

# Load environment variables from .env (GEMINI_API_KEY)
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass



def _build_structured_explanation(
    score_result: Dict[str, Any],
    risk_result: Dict[str, Any],
) -> str:
    """
    Build a plain-text structured explanation using only Python.

    This serves two purposes:
    - As a fallback when Gemini is not available.
    - As the "truth" that AI should rephrase but not change.
    """
    decision = score_result.get("decision", "HOLD")
    total_score = score_result.get("total_score", 0)
    components: List[Dict[str, Any]] = score_result.get("components", [])

    volatility = risk_result.get("volatility", None)
    max_dd = risk_result.get("max_drawdown", None)
    trend_strength = risk_result.get("trend_strength", None)
    regime = risk_result.get("market_regime", "unknown")
    risk_level = risk_result.get("risk_level", "unknown")

    lines: List[str] = []

    # High-level summary
    lines.append(
        f"Project ORION assigned a total score of {total_score} and "
        f"recommends to {decision} this stock."
    )

    # Break down scoring components
    for comp in components:
        name = comp.get("name", "Component")
        score = comp.get("score", 0)
        details = comp.get("details", [])
        lines.append(f"{name}: score {score}.")
        for d in details:
            lines.append(f"  - {d}")

    # Risk summary (include both raw values and simple impact hints)
    lines.append("")
    lines.append("Risk summary (values and impact hints):")
    if volatility is not None:
        lines.append(
            f"  - Volatility (daily return std dev): {volatility:.4f}. "
            "Lower values mean the price has been moving more smoothly day to day; "
            "higher values mean the price has been jumping around more."
        )
    if max_dd is not None:
        lines.append(
            f"  - Maximum drawdown: {max_dd:.2%}. "
            "This is the worst peak-to-trough drop in the period, showing how much "
            "the stock could fall during a bad stretch."
        )
    if trend_strength is not None:
        lines.append(
            f"  - Trend strength vs SMA200: {trend_strength:.2%}. "
            "Positive values mean the price is above its long-term average "
            "(more bullish), negative values mean it is below (more bearish)."
        )

    lines.append(f"  - Market regime (based on trend strength): {regime}.")
    lines.append(
        f"  - Overall risk level (combined view of volatility and drawdown): {risk_level}."
    )

    lines.append("")
    lines.append(
        "Interpretation summary: ORION uses these risk metrics to judge how stable "
        "the stock has been, how severe past drops were, whether the current price "
        "is in a bullish or bearish zone relative to its long-term trend, and how "
        "much overall risk it sees when combining these signals."
    )

    return "\n".join(lines)


def _explain_with_gemini(structured_text: str) -> str:
    """
    Use Gemini API to turn the structured explanation into
    a friendly, beginner-readable report.

    Note:
    - The input `structured_text` is the source of truth.
    - Gemini is allowed to rephrase and elaborate, not to change
      the decision, scores, or numeric values.
    """
    if not GEMINI_API_KEY:
        # No key configured -> return structured text directly
        return structured_text

    try:
        # Using the Google GenAI SDK ('google-genai').[web:49][web:82]
        # Make sure you installed: pip install google-genai
        from google import genai

        client = genai.Client(api_key=GEMINI_API_KEY)

        prompt = f"""
You are a financial analysis assistant.
You are ORION AI.

Behave simultaneously as:
• Financial Analyst
• Quantitative Analyst
• Data Scientist
• Market Educator
• Risk Analyst
• AI Explainer

Your purpose is to transform the structured ORION output into insights
that any beginner can understand.

ALWAYS:
- Explain your reasoning step by step.
- Separate clearly between observed facts, analysis, forecasts, and uncertainty.
- Use only the data provided in the structured explanation below.
- Never fabricate information or provide financial advice.
- Never change the underlying decision (BUY/HOLD/SELL), scores, or numeric values.

CRITICAL FOCUS FOR RISK METRICS:
You are given:
- volatility
- max_drawdown
- trend_strength
- market_regime
- risk_level

For these, you MUST explain:
- What each value means for how the stock behaves (stability vs bumpiness,
  how bad past drops were, whether the stock is in a bullish or bearish zone,
  how risky ORION thinks the stock is).
- How each metric affects the overall risk view and the final recommendation.
- Focus on their IMPACT on the stock and on the decision, not textbook definitions.

Generate the report using the following sections whenever the data supports them:

1. Market Summary
2. Trend Analysis
3. Technical Indicators
4. Price Forecast
5. Sentiment Analysis
6. Risk Analysis
7. Overall Interpretation
8. Confidence Level
9. Learning Corner
10. Disclaimer

For every metric, do not simply restate its value.

Explain:
• what the current value indicates for this stock,
• whether it is positive, negative, or neutral for the situation,
• how it affects the current market outlook,
• and how it influenced the final conclusion.

If a section (for example, Sentiment Analysis or detailed Price Forecast)
cannot be supported by the data in the structured explanation, include it
with a short note saying the information is not available and you cannot
speculate beyond the provided data.

Use clear, beginner-friendly language while remaining technically accurate.

Below is the structured explanation generated by the deterministic ORION engine.
Use it as your only source of truth:

{structured_text}

Now write a concise, well-structured report using the 10 sections above.
Make sure that the Risk Analysis section explicitly interprets:
- volatility
- max_drawdown
- trend_strength
- market_regime
- risk_level
in terms of how they affect the stock's behavior and the overall decision.
"""

        chat = client.chats.create(model="gemini-2.0-flash")
        response = chat.send_message(prompt)
        return response.text

    except Exception as e:
        # If anything goes wrong with Gemini, fall back
        print(f"Gemini explanation error: {e}")
        return structured_text


def explain_decision(
    score_result: Dict[str, Any],
    risk_result: Dict[str, Any],
    use_gemini: bool = True,
) -> str:
    """
    Public function to generate an explanation.

    Parameters:
        score_result: Output from scoring_engine.score_stock(...).
        risk_result: Output from risk_engine.analyze_risk(...).
        use_gemini: If True, try to use Gemini; otherwise, use plain Python.

    Returns:
        Explanation string.
    """
    structured = _build_structured_explanation(score_result, risk_result)

    if use_gemini:
        return _explain_with_gemini(structured)
    else:
        return structured