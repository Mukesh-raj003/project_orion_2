"""
market_data.py

Purpose:
- Download historical stock price data using yfinance.
- Provide clean, reusable functions for other parts of the app.

Concepts introduced:
- Importing libraries.
- Functions and parameters.
- Return values.
- Basic error handling with try/except.
- Working with pandas DataFrame.

Why this exists:
- Separating data access logic makes the rest of the code easier to read and test.
"""

from typing import Optional
import datetime as dt

import pandas as pd
import yfinance as yf

from config import DEFAULT_LOOKBACK_DAYS, DEFAULT_INTERVAL


def get_date_range(days_back: int = DEFAULT_LOOKBACK_DAYS) -> tuple[dt.date, dt.date]:
    """
    Calculate the start and end dates for data download.

    Parameters:
        days_back: How many days of history you want.

    Returns:
        A tuple (start_date, end_date).
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=days_back)
    return start, end


def download_stock_history(
    ticker: str,
    days_back: int = DEFAULT_LOOKBACK_DAYS,
    interval: str = DEFAULT_INTERVAL,
) -> Optional[pd.DataFrame]:
    """
    Download historical price data for a single stock.

    Parameters:
        ticker: The stock symbol, e.g. "AAPL".
        days_back: How many days of data to fetch.
        interval: Candle interval, e.g. "1d", "1h".

    Returns:
        A pandas DataFrame with columns like: Open, High, Low, Close, Volume.
        Returns None if download fails or no data is available.
    """
    start, end = get_date_range(days_back)

    try:
        # yfinance uses a Ticker object as a simple interface
        yf_ticker = yf.Ticker(ticker)

        # history() returns a DataFrame indexed by date
        df = yf_ticker.history(start=start, end=end, interval=interval)

        # Sometimes the API returns an empty DataFrame
        if df.empty:
            return None

        # Ensure the index is a normal DatetimeIndex (sometimes it's something else)
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        return df

    except Exception as e:
        # In a real app, you might want to log this error.
        print(f"Error downloading data for {ticker}: {e}")
        return None