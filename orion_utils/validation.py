"""
validation.py

Purpose:
- Centralize input validation (e.g., ticker symbol).

Concepts introduced:
- Basic string checks.
- Raising and handling simple errors.
"""

from typing import Tuple


def validate_ticker(ticker: str) -> Tuple[bool, str]:
    """
    Validate a stock ticker symbol.

    Parameters:
        ticker: The user-provided ticker string.

    Returns:
        (is_valid, message)
        - is_valid: True if ticker looks okay.
        - message: Error message if invalid, or empty string if valid.
    """
    if not ticker:
        return False, "Ticker cannot be empty."

    ticker = ticker.strip()

    if len(ticker) < 1 or len(ticker) > 10:
        return False, "Ticker length should be between 1 and 10 characters."

    if " " in ticker:
        return False, "Ticker should not contain spaces."

    # This is a very basic check; real validation would hit an API/exchange.
    return True, ""