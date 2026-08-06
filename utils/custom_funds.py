"""
Custom fund price data module.

Drop-in replacement for investpy fund lookups.
Fetches fund (managed fund / ETF) price data via yfinance.
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger('pt_logger.custom_funds')

# Map known APIR/ISIN codes to yfinance tickers where possible.
# Falls back to ISIN lookup via yfinance if not mapped here.
KNOWN_FUND_TICKERS = {
    # Vanguard
    'VAN0100AU': 'VAN0100AU.AX',   # Vanguard High Growth Index
    'VAN0101AU': 'VAN0101AU.AX',   # Vanguard Growth Index
    'VAN0102AU': 'VAN0102AU.AX',   # Vanguard Balanced Index
    'VAN0103AU': 'VAN0103AU.AX',   # Vanguard Conservative Index
    'VAN1111AU': 'VAN1111AU.AX',   # Vanguard Diversified High Growth
    'VAN1112AU': 'VAN1112AU.AX',   # Vanguard Diversified Growth
    'VAN1113AU': 'VAN1113AU.AX',   # Vanguard Diversified Balanced
    'VAN1114AU': 'VAN1114AU.AX',   # Vanguard Diversified Conservative
    # Fidelity
    'AU60FID00151': 'AU60FID00151.AX',  # Fidelity India Fund (tested)
    # Magellan
    'AU60MGE00028': 'AU60MGE00028.AX',  # Magellan Global Fund
    # Regal
    'AU60RGL00047': 'AU60RGL00047.AX',  # Regal Investment Fund (RF1)
    'AU60RGL00039': 'AU60RGL00039.AX',  # Regal Partners Ltd
}


def get_custom_fund_data(isin: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """Fetch fund price data via yfinance.

    Args:
        isin: APIR code or ISIN (e.g. "AU60FID00151")
        start_date: Start date for price history
        end_date: End date for price history

    Returns:
        DataFrame with Date index and Close/Splits/Dividends columns,
        or None if data could not be retrieved.
    """
    # Determine the best ticker to try
    ticker_candidates = []

    if isin in KNOWN_FUND_TICKERS:
        ticker_candidates.append(KNOWN_FUND_TICKERS[isin])

    # Try the raw ISIN on .AX
    ticker_candidates.append(f"{isin}.AX")

    # Try the raw ISIN without suffix
    ticker_candidates.append(isin)

    for ticker in ticker_candidates:
        try:
            df = yf.Ticker(ticker).history(
                start=start_date,
                end=end_date,
                auto_adjust=False,
                rounding=False,
            )
            if df is not None and not df.empty:
                df = df.tz_localize(None)
                df.rename(columns={'Stock Splits': 'Splits'}, inplace=True)

                # Ensure required columns
                if 'Splits' not in df.columns:
                    df['Splits'] = 0
                if 'Dividends' not in df.columns:
                    df['Dividends'] = 0

                logger.info(f"Got fund data for {isin} via yfinance ticker={ticker}: {len(df)} records")
                return df
        except Exception as e:
            logger.debug(f"Fund ticker {ticker} ({isin}) failed: {e}")
            continue

    logger.warning(f"No fund data found for {isin} via any yfinance ticker path")
    return None
