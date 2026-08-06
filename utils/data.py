from datetime import datetime
import logging
from multiprocessing.pool import ThreadPool
from typing import List, Tuple

import pandas as pd
import yfinance as yf
import yahooquery as yq

from utils.crypto import get_crypto_price
from utils.custom_funds import get_custom_fund_data

logger = logging.getLogger('pt_logger.Stock')


def get_price_data_ticker(ticker: str, start_date: datetime, end_date: datetime, currency: str) -> pd.DataFrame:
    """Get price data for a single ticker for the specified period.

    Routes to the correct data source based on ticker type suffix.

    Args:
        ticker: Ticker symbol (may include .LOAN/.CASH/.FUND/.CRYPTO/.FX suffix)
        start_date: Start date
        end_date: End date
        currency: Base currency for price conversion

    Returns:
        DataFrame with price data, or empty DataFrame on failure
    """
    raw_ticker, ticker_type = split_ticker(ticker=ticker)

    if ticker_type == 'LOAN':
        dl_data = get_loan_data(start_date, end_date)
    elif ticker_type == 'CASH':
        dl_data = get_cash_data(start_date, end_date)
    elif ticker_type == 'FUND':
        dl_data = get_fund_data(raw_ticker, start_date, end_date)
    elif ticker_type == 'CRYPTO':
        currency = 'USD'
        dl_data = get_crypto_price(raw_ticker, start_date, end_date, currency)
    elif ticker_type == 'FX':
        dl_data = get_currency_data(raw_ticker, start_date, end_date)
    else:
        dl_data = get_yf_price(ticker, start_date, end_date)

    if isinstance(dl_data, pd.DataFrame):
        logger.debug(f'Data downloaded for {ticker}: Start: {start_date.date()} | End: {end_date.date()}')
    else:
        dl_data = pd.DataFrame()
        logger.debug(f'-------  No data found for ticker: {ticker} -------')
    return dl_data


def get_price_data(tickers: List, start_dates: List, end_dates: List, currency: str) -> pd.DataFrame:
    """Get price data for a list of tickers in parallel.

    Args:
        tickers: List of ticker symbols
        start_dates: List of start dates
        end_dates: List of end dates
        currency: Base currency

    Returns:
        Concatenated DataFrame with Ticker/Date multi-index
    """
    tickers = list(tickers)
    start_dates = list(start_dates)
    end_dates = list(end_dates)

    if len(tickers) == 0:
        logger.info('No tickers provided for price lookup; returning empty DataFrame')
        return pd.DataFrame()

    try:
        with ThreadPool(processes=10) as pool:
            all_data = pool.starmap(get_price_data_ticker, zip(
                tickers, start_dates, end_dates, [currency] * len(tickers)))
            logger.debug('Obtained data, concatenating')
            filtered = [(df, ticker) for df, ticker in zip(all_data, tickers) if not df.empty]
            if not filtered:
                logger.info('Price lookup returned no data for provided tickers')
                return pd.DataFrame()
            all_data, tickers = zip(*filtered)
            concat_data = pd.concat(all_data, keys=tickers, names=['Ticker', 'Date'])
    except ValueError as e:
        raise ValueError(f'Please provide at least one ticker. Error: {e}')
    return concat_data


def get_loan_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Create synthetic price data for loan assets (fixed -1 close price)."""
    df = pd.DataFrame({'Date': pd.date_range(start_date, end_date, freq='D')})
    df['Close'] = -1
    df['Splits'] = 0
    df['Dividends'] = 0
    df.set_index('Date', inplace=True)
    return df


def get_cash_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Create synthetic price data for cash assets (fixed 1 close price)."""
    df = pd.DataFrame({'Date': pd.date_range(start_date, end_date, freq='D')})
    df['Close'] = 1
    df['Splits'] = 0
    df['Dividends'] = 0
    df.set_index('Date', inplace=True)
    return df


def get_fund_data(isin: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Get fund price data.

    Uses custom_funds module (yfinance-backed), with investpy fallback removed
    (investpy is unmaintained since 2022).

    Args:
        isin: APIR code or ISIN (e.g. "AU60FID00151")
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with Close/Splits/Dividends columns, or None
    """
    # Try custom_funds (yfinance-backed)
    try:
        df = get_custom_fund_data(isin, start_date, end_date)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception as e:
        logger.warning(f'custom_funds lookup failed for {isin}: {e}')

    # Fallback: try yfinance with a plain `.AX` suffix
    try:
        logger.info(f'Attempting yfinance direct lookup for {isin}')
        df = yf.Ticker(f'{isin}.AX').history(
            start=start_date, end=end_date, auto_adjust=False, rounding=False
        )
        if df is not None and not df.empty:
            df = df.tz_localize(None)
            df.rename(columns={'Stock Splits': 'Splits'}, inplace=True)
            for col in ['Splits', 'Dividends']:
                if col not in df.columns:
                    df[col] = 0
            return df
    except Exception as e:
        logger.debug(f'yfinance fallback failed for {isin}.AX: {e}')

    return None


def get_yf_price(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Get price data from yfinance.

    Args:
        ticker: Yahoo Finance ticker
        start_date: Start date
        end_date: End date

    Returns:
        DataFrame with yfinance price data, or None
    """
    try:
        df = yf.Ticker(ticker).history(
            start=start_date, end=end_date, auto_adjust=False, rounding=False)
        df.rename(columns={'Stock Splits': 'Splits', 'Adj Close': 'Adjclose'}, inplace=True)
        df = df.tz_localize(None)
        if 'Capital Gains' in df.columns:
            df.drop(columns=['Capital Gains'], inplace=True)
    except Exception as e:
        logger.error(f'yfinance error for {ticker}: {e}')
        df = None
    return df


def get_yq_price(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Get price data from yahooquery (legacy — prefer yfinance).

    Kept for backward compatibility but yfinance is preferred.
    """
    try:
        df = yq.Ticker(ticker).history(start=start_date, end=end_date).reset_index()
        df.drop(columns='symbol', inplace=True)
        df = df.rename(str.capitalize, axis=1).set_index('Date')
        df.index = pd.to_datetime(df.index)
        df.index = df.index.tz_localize(None)
        df.index = pd.Index(df.index.date)
        if 'Capital Gains' in df.columns:
            df.drop(columns=['Capital Gains'], inplace=True)
        df.index.names = ['Date']
    except Exception as e:
        logger.error(f'yahooquery error for {ticker}: {e}')
        df = None
    return df


def get_name(ticker: str) -> str:
    """Get the human-readable name for a ticker.

    Uses yfinance; removes investpy dependency.
    """
    raw_ticker, ticker_type = split_ticker(ticker=ticker)

    if ticker_type in ('STOCK', 'LOAN', 'CASH', 'FX', 'CRYPTO'):
        if ticker_type == 'STOCK':
            try:
                stock = yf.Ticker(ticker)
                name = stock.info.get('shortName', 'NA')
            except Exception:
                name = 'NA'
        else:
            name = raw_ticker.replace('=X', '')
    elif ticker_type == 'FUND':
        # Try yfinance info first
        for candidate in [f'{raw_ticker}.AX', raw_ticker]:
            try:
                info = yf.Ticker(candidate).info
                name = info.get('shortName') or info.get('longName') or 'NA'
                if name != 'NA':
                    break
            except Exception:
                name = 'NA'
    else:
        name = 'NA'

    return name


def get_currency(ticker: str) -> str:
    """Get the trading currency for a ticker."""
    raw_ticker, ticker_type = split_ticker(ticker=ticker)

    if ticker_type == 'STOCK':
        try:
            stock = yf.Ticker(ticker)
            currency = stock.info.get('currency', 'NA')
        except Exception:
            currency = 'NA'
    elif ticker_type == 'FUND':
        currency = 'AUD'
    elif ticker_type == 'CRYPTO':
        currency = 'USD'
    else:
        currency = 'NA'

    return str(currency).upper()


def get_ticker_type(ticker_type: str) -> str:
    """Map a ticker type suffix to a canonical type."""
    if ticker_type not in ['LOAN', 'CASH', 'FUND', 'CRYPTO', 'FX']:
        return 'STOCK'
    return ticker_type


def split_ticker(ticker: str) -> Tuple[str, str]:
    """Split a ticker symbol into raw ticker and type suffix.

    E.g. 'BHP.AX' -> ('BHP.AX', 'STOCK'), 'AU60FID00151.FUND' -> ('AU60FID00151', 'FUND')
    """
    if len(ticker.split('.')) < 2:
        ticker_type = None
    else:
        ticker_type = ticker.split('.')[1]
    raw_ticker = ticker.split('.')[0]
    ticker_type = get_ticker_type(ticker_type)
    return raw_ticker, ticker_type


def get_currency_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Get FX rate data from yfinance."""
    return get_yf_price(ticker=ticker, start_date=start_date, end_date=end_date)
