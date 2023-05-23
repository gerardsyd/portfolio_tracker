from datetime import datetime, timedelta
import logging
from multiprocessing.pool import ThreadPool
from typing import List, Tuple
import traceback

import investpy
from json.decoder import JSONDecodeError
import pandas as pd
import yfinance as yf
import yahooquery as yq


from utils.crypto import get_crypto_price

logger = logging.getLogger('pt_logger.Stock')


def get_price_data_ticker(ticker: str, start_date: datetime, end_date: datetime, currency: str) -> pd.DataFrame:
    """
    Gets price data for ticker for specified period

    Args:
        ticker (str): String ticker in format that is acceptable to Yahoo Finance
        start_date (datetime): Start date to get price data
        end_date (datetime): End date to get price data
        currency (str): currency to convert pricing

    Returns:
        pd.DataFrame: Dataframe containing open, close, high, low, split, dividend data for ticker from start_date to end_date
    """

    logger.debug(f'-------  Ticker is {ticker}  -------')
    raw_ticker, ticker_type = split_ticker(ticker=ticker)

    if ticker_type == 'LOAN':
        dl_data = get_loan_data(start_date, end_date)
    elif ticker_type == 'CASH':
        dl_data = get_cash_data(start_date, end_date)
    elif ticker_type == 'FUND':
        dl_data = get_fund_data(raw_ticker, start_date, end_date)
    elif ticker_type == 'CRYPTO':
        dl_data = get_crypto_price(raw_ticker, start_date, end_date, currency)
    elif ticker_type == 'FX':
        dl_data = get_currency_data(raw_ticker, start_date, end_date)
    else:
        # assumes that ticker is YQ acceptable ticker and attempts to obtain price from yahooquery
        dl_data = get_yq_price(ticker, start_date, end_date)

    if isinstance(dl_data, pd.DataFrame):
        logger.debug(
            f'Data downloaded for {ticker}: Start: {start_date.date()} | End: {end_date.date()}')
    else:
        dl_data = pd.DataFrame()
        logger.debug(
            f'-------  No data found for ticker: {ticker} -------')
    # print(dl_data)
    return dl_data


def get_price_data(tickers: List, start_dates: List, end_dates: List, currency: str) -> pd.DataFrame:
    """
    Gets price data for a list of tickers for period specified

    Args:
        tickers (List): String list of tickers in format that is acceptable to Yahoo Finance
        startdate (datetime): Start date to get price data
        end_date (datetime): End date to get price data

    Returns:
        pd.DataFrame: Dataframe containing open, close, high, low, split, dividend data for each ticker from start_date to end_date
    """
    try:
        with ThreadPool(processes=10) as pool:
            all_data = pool.starmap(get_price_data_ticker, zip(
                tickers, start_dates, end_dates, currency))
            logger.debug('Obtained data, concatenating')
            concat_data = pd.concat(
                all_data, keys=tickers, names=['Ticker', 'Date'])
    except ValueError as e:
        raise ValueError('Please provide at least one ticker')
    return concat_data


def get_duplicates(lst):
    counts = {}
    duplicates = []

    for item in lst:
        if item in counts:
            counts[item] += 1
        else:
            counts[item] = 1

    for key, value in counts.items():
        if value > 1:
            duplicates.append(key)

    return duplicates


def get_loan_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Creates loan data dataframe containing -1 as close price and no stock splits or dividends

    Args:
        start_date (datetime): Start date to get price data
        end_date (datetime): End date to get price data

    Returns:
        pd.DataFrame: Dataframe close, split, dividend data for loans from start_date to end_date
    """

    df = pd.DataFrame(
        {'Date': pd.date_range(start_date, end_date, freq='D')})
    df['Close'] = -1
    df['Splits'] = 0
    df['Dividends'] = 0
    df.set_index('Date', inplace=True)
    return df


def get_cash_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Creates loan data dataframe containing -1 as close price and no stock splits or dividends

    Args:
        start_date (datetime): Start date to get price data
        end_date (datetime): End date to get price data

    Returns:
        pd.DataFrame: Dataframe close, split, dividend data for loans from start_date to end_date
    """

    df = pd.DataFrame(
        {'Date': pd.date_range(start_date, end_date, freq='D')})
    df['Close'] = 1
    df['Splits'] = 0
    df['Dividends'] = 0
    df.set_index('Date', inplace=True)
    return df


def get_fund_data(isin: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Gets price data for fund for specified period

    Args:
        ticker (str): String ticker with ISIN
        start_date (datetime): Start date to get price data
        end_date (datetime): End date to get price data

    Returns:
        pd.DataFrame: Dataframe containing close, split, dividend data for ticker from start_date to end_date
    """
    # check if there is a custom funds module, import and execute custom fund function
    df = None
    try:
        from utils.custom_funds import get_custom_fund_data
        df = get_custom_fund_data(isin, start_date, end_date)
    except ImportError:
        print('No custom funds module available')

    # If not data loaded from custom funds, try investpy
    if not isinstance(df, pd.DataFrame):
        try:
            fund_search = investpy.search_funds(by='isin', value=isin)
            name = fund_search.at[0, 'name']
            country = fund_search.at[0, 'country']
            df = investpy.get_fund_historical_data(
                fund=name, country=country, from_date=start_date.strftime('%d/%m/%Y'), to_date=end_date.strftime('%d/%m/%Y'))
            df.drop('Currency', axis=1, inplace=True)
            df.reset_index(inplace=True)
        except (RuntimeError, ValueError, ConnectionError):
            df = None

    # if data downloaded, include nil stock splits and dividends (as info is not available)
    if isinstance(df, pd.DataFrame):
        df['Splits'] = 0
        df['Dividends'] = 0
        df.set_index(['Date'], inplace=True, drop=True)

    return df


def get_yq_price(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Gets price data for ticker for specified period from yfinance

    Args:
        ticker (str): String ticker in format that is acceptable to Yahoo Finance
        start_date (datetime): Start date to get price data
        end_date (datetime): End date to get price data

    Returns:
        pd.DataFrame: Dataframe containing open, close, high, low, split, dividend data for ticker from start_date to end_date
    """
    try:
        df = yq.Ticker(ticker).history(start=start_date, end=end_date).reset_index()
        df.drop(columns='symbol', inplace=True)  # drop symbol column
        df = df.rename(str.capitalize, axis=1).set_index('Date')
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.tz_localize(None)  # remove TZ aware from downloaded data
            df.index = pd.Index(df.index.date)  # remove times from dowloaded data to get clean dataset
        if 'Capital Gains' in df.columns:
            df.drop(columns=["Capital Gains"], inplace=True)
        df.index.names = ['Date']
    except KeyError:
        df = None
    except RuntimeError:
        logger.debug(
            f'-------  Yahoo! Finance is not working (ticker: {ticker}) -------')
        df = None
    except ConnectionError:
        logger.debug(
            f'-------  Connection error with Yahoo! Finance (ticker: {ticker}) -------')
        df = None
    except JSONDecodeError as e:
        logger.debug(
            f'-------  Connection error with Yahoo! Finance (ticker: {ticker}) -------')
        logger.debug(e)
        df = None
    return df


def get_yf_price(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Gets price data for ticker for specified period from yfinance

    Args:
        ticker (str): String ticker in format that is acceptable to Yahoo Finance
        start_date (datetime): Start date to get price data
        end_date (datetime): End date to get price data

    Returns:
        pd.DataFrame: Dataframe containing open, close, high, low, split, dividend data for ticker from start_date to end_date
    """
    try:
        df = yf.Ticker(ticker).history(
            start=start_date, end=end_date, auto_adjust=False, rounding=False, debug=False)
        df.rename(columns={'Stock Splits': 'Splits',
                  'Adj Close': 'Adjclose'}, inplace=True)
        df = df.tz_localize(None)  # remove TZ aware from downloaded data
        if 'Capital Gains' in df.columns:
            df.drop(columns=["Capital Gains"], inplace=True)
    except KeyError:
        df = None
    except RuntimeError:
        logger.debug(
            f'-------  Yahoo! Finance is not working (ticker: {ticker}) -------')
        df = None
    except ConnectionError:
        logger.debug(
            f'-------  Connection error with Yahoo! Finance (ticker: {ticker}) -------')
        df = None
    except JSONDecodeError as e:
        logger.debug(
            f'-------  Connection error with Yahoo! Finance (ticker: {ticker}) -------')
        logger.debug(e)
        df = None
    return df


def get_name(ticker: str) -> str:
    """
    Gets name of ticker

    Args:
        ticker (str): String ticker in format that is acceptable to Yahoo Finance

    Returns:
        str: Full name of stock based on ticker
    """

    logger.debug(f'Getting name for {ticker}')
    raw_ticker, ticker_type = split_ticker(ticker=ticker)
    logger.debug([raw_ticker, ticker_type])

    if ticker_type == 'STOCK':
        try:
            stock = yq.Ticker(ticker)
            name = stock.quote_type[ticker]['longName']
        except (IndexError, KeyError, Exception, ValueError, AttributeError) as e:
            logger.info(f'-------  Ticker name {ticker} not found -------')
            logger.debug(f'-------  Error is {e} -------')
            name = "NA"
    elif ticker_type == 'FUND':
        try:
            fund_search = investpy.search_funds(
                by='isin', value=raw_ticker)
            name = fund_search.at[0, 'name']
        except (RuntimeError, ValueError):
            name = "NA"
    elif ticker_type == 'FX':
        name = raw_ticker.replace('=X', '')
    else:
        # catch all for all other types including CRYPTO
        name = raw_ticker
    return name


def get_currency(ticker: str) -> str:
    """
    Gets quoted currency of ticker

    Args:
        ticker (str): String ticker in format that is acceptable to Yahoo Finance

    Returns:
        str: Full name of stock based on ticker
    """

    logger.debug(f'Getting quoted currency for {ticker}')
    raw_ticker, ticker_type = split_ticker(ticker=ticker)

    if ticker_type == 'STOCK':
        try:
            stock = yq.Ticker(ticker)
            try:
                currency = stock.price[ticker]['currency']
            except (IndexError, KeyError, Exception) as e:
                logger.info(
                    f'-------  Currency for {ticker} not found -------')
                logger.debug(f'-------  Error is {e} -------')
                currency = "NA"
        except (ValueError, AttributeError):
            logger.debug(
                f'-------  Ticker {ticker} not found (currency) -------')
            currency = "NA"
    elif ticker_type == 'FUND':
        try:
            df = investpy.search_funds(by='isin', value=raw_ticker)
            currency = df.at[0, 'currency']
        except (RuntimeError, ValueError, ConnectionError):
            currency = "NA"
    elif ticker_type == 'CRYPTO':
        currency = raw_ticker
    else:
        currency = "NA"
    return str(currency).upper()


def get_ticker_type(ticker_type: str) -> str:
    if ticker_type not in ['LOAN', 'CASH', 'FUND', 'CRYPTO', 'FX']:
        return 'STOCK'
    else:
        return ticker_type


def split_ticker(ticker: str) -> Tuple[str, str]:
    if len(ticker.split('.')) < 2:
        ticker_type = None
    else:
        ticker_type = ticker.split('.')[1]
    raw_ticker = ticker.split('.')[0]
    ticker_type = get_ticker_type(ticker_type)
    return raw_ticker, ticker_type


def get_currency_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    return get_yq_price(ticker=ticker, start_date=start_date, end_date=end_date)


if __name__ == '__main__':
    print(get_name('GMVD'))
