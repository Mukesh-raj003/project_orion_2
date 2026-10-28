"""
config.py

Purpose:
- Store configuration values in one place so the rest of the app
  does not have "magic numbers" or hard-coded strings everywhere.

Concepts introduced:
- Module-level constants.
- Simple configuration pattern.
"""

from typing import List

# Default settings for data download
DEFAULT_LOOKBACK_DAYS: int = 365  # 1 year of daily data
DEFAULT_INTERVAL: str = "1d"      # 1 day candles

# Technical indicator settings
SMA_WINDOWS: List[int] = [20, 50, 200]  # We start with 20 in Phase 1
RSI_PERIOD: int = 14                    # Classic Welles Wilder RSI period[web:2]
MACD_FAST: int = 12                     # Standard MACD fast EMA length[web:10]
MACD_SLOW: int = 26                     # Standard MACD slow EMA length[web:10]
MACD_SIGNAL: int = 9                    # Standard MACD signal EMA length[web:10]

# Streamlit UI settings
APP_TITLE: str = "Project ORION: Educational Stock Analysis"
APP_THEME_COLOR: str = "#3498db"  # just a pretty blue