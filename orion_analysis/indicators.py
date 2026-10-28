"""
indicators.py

Purpose:
- Compute technical indicators such as SMA, RSI, MACD, volume stats,
  and 52-week high/low on price data.

Concepts introduced:
- Using pandas to add new columns to a DataFrame.
- Rolling window calculations.
- Exponential moving averages.
- Writing small, focused functions.
"""

from typing import List

import pandas as pd

from config import (
    RSI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    SMA_WINDOWS,
)


def add_sma(df: pd.DataFrame, window: int, price_column: str = "Close") -> pd.DataFrame:
    """
    Add a Simple Moving Average (SMA) column to the DataFrame.

    SMA = average of the last N closing prices.

    Parameters:
        df: Price DataFrame with at least a 'Close' column.
        window: Number of periods for the SMA (e.g., 20).
        price_column: Column to average, usually 'Close'.

    Returns:
        The same DataFrame with a new column named f"SMA_{window}".
    """
    column_name = f"SMA_{window}"
    df[column_name] = df[price_column].rolling(window=window).mean()
    return df


def add_multiple_smas(
    df: pd.DataFrame,
    windows: List[int] = SMA_WINDOWS,
    price_column: str = "Close",
) -> pd.DataFrame:
    """
    Add several SMA columns (e.g., 20, 50, 200) in one call.

    Parameters:
        df: Price DataFrame.
        windows: List of window sizes.
        price_column: Column to average.

    Returns:
        DataFrame with additional SMA columns.
    """
    for w in windows:
        df = add_sma(df, window=w, price_column=price_column)
    return df


def add_rsi(
    df: pd.DataFrame,
    period: int = RSI_PERIOD,
    price_column: str = "Close",
) -> pd.DataFrame:
    """
    Add a Relative Strength Index (RSI) column.

    RSI measures the size of recent gains vs losses and moves
    between 0 and 100, with common overbought/oversold levels
    at 70 and 30.[web:2][web:5]

    Parameters:
        df: Price DataFrame.
        period: Lookback period for RSI (default 14).
        price_column: Column to use for price.

    Returns:
        DataFrame with a new 'RSI' column.
    """
    delta = df[price_column].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    avg_loss = avg_loss.replace(0, 1e-10)

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    df["RSI"] = rsi
    return df


def _ema(series: pd.Series, span: int) -> pd.Series:
    """
    Helper function to compute an Exponential Moving Average (EMA).

    EMA is similar to SMA but gives more weight to recent prices.

    Parameters:
        series: The data to smooth.
        span: Controls how much weight is given to recent data.

    Returns:
        A pandas Series containing the EMA.
    """
    return series.ewm(span=span, adjust=False).mean()


def add_macd(
    df: pd.DataFrame,
    price_column: str = "Close",
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> pd.DataFrame:
    """
    Add MACD, Signal, and Histogram columns.

    MACD Line   = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD, signal)
    Histogram   = MACD - Signal[web:10][web:15]

    Parameters:
        df: Price DataFrame.
        price_column: Column with price data.
        fast: Fast EMA span (default 12).
        slow: Slow EMA span (default 26).
        signal: Signal EMA span (default 9).

    Returns:
        DataFrame with 'MACD', 'MACD_SIGNAL', 'MACD_HIST' columns.
    """
    close = df[price_column]

    ema_fast = _ema(close, span=fast)
    ema_slow = _ema(close, span=slow)

    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, span=signal)
    histogram = macd_line - signal_line

    df["MACD"] = macd_line
    df["MACD_SIGNAL"] = signal_line
    df["MACD_HIST"] = histogram

    return df


def add_volume_features(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Add simple volume analysis columns.

    - VOL_SMA_{window}: average volume over the last N days.
    - VOL_REL_{window}: today's volume / VOL_SMA_{window}
                         (values > 1 mean higher than average).

    Parameters:
        df: Price DataFrame with 'Volume' column.
        window: Lookback for average volume (default 20 days).

    Returns:
        DataFrame with volume SMA and relative volume.
    """
    sma_col = f"VOL_SMA_{window}"
    rel_col = f"VOL_REL_{window}"

    df[sma_col] = df["Volume"].rolling(window=window).mean()
    df[rel_col] = df["Volume"] / df[sma_col]
    return df


def add_52_week_high_low(df: pd.DataFrame, price_column: str = "Close") -> pd.DataFrame:
    """
    Add 52-week high and low columns.

    Assumes df covers at least 252 trading days (~1 year).

    Parameters:
        df: Price DataFrame.
        price_column: Column to use for high/low calculation.

    Returns:
        DataFrame with 'HIGH_52W' and 'LOW_52W' columns filled
        with the same values for all rows (for convenience).
    """
    if df.empty:
        return df

    rolling_window = 252  # ~1 year of trading days
    high_52w = df[price_column].rolling(window=rolling_window).max()
    low_52w = df[price_column].rolling(window=rolling_window).min()

    # For simplicity, we fill forward the last known high/low
    df["HIGH_52W"] = high_52w.ffill()
    df["LOW_52W"] = low_52w.ffill()

    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to compute all indicators used by ORION.

    Parameters:
        df: Price DataFrame with 'Close' and 'Volume'.

    Returns:
        DataFrame with SMA20/50/200, RSI, MACD, volume stats,
        and 52-week high/low.
    """
    df = add_multiple_smas(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_volume_features(df, window=20)
    df = add_52_week_high_low(df)
    return df