"""
helpers.py

Purpose:
- General helper functions used across the project.

Concepts introduced:
- Simple "utility" functions that keep app.py and dashboard.py clean.
"""

from typing import Optional

import pandas as pd


def is_dataframe_valid(df: Optional[pd.DataFrame]) -> bool:
    """
    Check if a DataFrame is usable (not None and not empty).

    Parameters:
        df: A pandas DataFrame or None.

    Returns:
        True if df is not None and has at least one row, False otherwise.
    """
    if df is None:
        return False
    if df.empty:
        return False
    return True