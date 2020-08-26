from datetime import datetime
import logging
from multiprocessing.pool import ThreadPool
from typing import List

import investpy
import pandas as pd
import numpy as np
import yfinance as yf
from yfinance import ticker

from utils.crypto import get_crypto_price

logger = logging.getLogger('pt_logger.Stock')
yf.pdr_override()


def get_price_data_ticker(ticker: str, start_date: np.datetime64, end_date: np.datetime64, currency: str) -> pd.DataFrame:
    """
    Gets price data for ticker for specified period

    Args:
        ticker (str): String ticker in format that is acceptable to Yahoo Finance
        start_date (np.datetime64): Start date to get price data
        end_date (np.datetime64): End date to get price data
        currency (str): currency to convert pricing

    Returns:
        pd.DataFrame: Dataframe containing open, close, high, low, split, dividend data for ticker from start_date to end_date
    """

    logger.debug(f'-------  Ticker is {ticker}  -------')
    try:
        raw_ticker = ticker.split('.')[0]
        ticker_type = ticker.split('.')[1]
    except:
        raw_ticker = None
        ticker_type = None

    if ticker_type == 'LOAN':
        dl_data = get_loan_data(start_date, end_date)
    elif ticker_type == 'FUND':
        dl_data = get_fund_data(raw_ticker, start_date, end_date)
    elif ticker_type == 'CRYPTO':
        dl_data = get_crypto_price(raw_ticker, start_date, end_date, currency)
    else:
        # assumes that ticker is YF acceptable ticker and attempts to obtain price from Yfinance
        dl_data = get_yf_price(ticker, start_date, end_date)

    if isinstance(dl_data, pd.DataFrame):
        logger.debug(
            f'Data downloaded for {ticker}: Start: {start_date.date()} | End: {end_date.date()}')
    else:
        dl_data = pd.DataFrame()
        logger.debug(
            f'-------  No data found for ticker: {ticker} -------')

    return dl_data


def get_price_data(tickers: List, start_dates: List, end_dates: List, currency: str) -> pd.DataFrame:
    """
    Gets price data for a list of tickers for period specified

    Args:
        tickers (List): String list of tickers in format that is acceptable to Yahoo Finance
        startdate (np.datetime64): Start date to get price data
        end_date (np.datetime64): End date to get price data

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
    except ValueError:
        raise ValueError('Please provide at least one ticker')
    return concat_data


def get_loan_data(start_date: np.datetime64, end_date: np.datetime64) -> pd.DataFrame:
    """
    Creates loan data dataframe containing -1 as close price and no stock splits or dividends

    Args:
        start_date (np.datetime64): Start date to get price data
        end_date (np.datetime64): End date to get price data

    Returns:
        pd.DataFrame: Dataframe close, split, dividend data for loans from start_date to end_date
    """

    df = pd.DataFrame(
        {'Date': pd.date_range(start_date, end_date)})
    df['Close'] = -1
    df['Stock Splits'] = 0
    df['Dividends'] = 0
    df.set_index('Date', inplace=True)
    return df


def get_fund_data(isin: str, start_date: np.datetime64, end_date: np.datetime64) -> pd.DataFrame:
    """
    Gets price data for fund for specified period

    Args:
        ticker (str): String ticker with ISIN
        start_date (np.datetime64): Start date to get price data
        end_date (np.datetime64): End date to get price data

    Returns:
        pd.DataFrame: Dataframe containing close, split, dividend data for ticker from start_date to end_date
    """

    try:
        fund_search = investpy.search_funds(by='isin', value=isin)
        name = fund_search.at[0, 'name']
        country = fund_search.at[0, 'country']
        df = investpy.get_fund_historical_data(
            fund=name, country=country, from_date=start_date.strftime('%d/%m/%Y'), to_date=end_date.strftime('%d/%m/%Y'))
        df.drop('Currency', axis=1, inplace=True)
        df.reset_index(inplace=True)
    except RuntimeError:
        # if not in investpy database, check if there is a custom funds module, import and execute custom fund function
        try:
            from utils.custom_funds import get_custom_fund_data
            df = get_custom_fund_data(
                isin, start_date, end_date)
        except ImportError:
            print('No custom funds module available')
            df = None
    except ValueError:
        df = None

    if isinstance(df, pd.DataFrame):
        df['Stock Splits'] = 0
        df['Dividends'] = 0
        df.set_index(['Date'], inplace=True, drop=True)

    return df


def get_yf_price(ticker: str, start_date: np.datetime64, end_date: np.datetime64) -> pd.DataFrame:
    """
    Gets price data for ticker for specified period from yfinance

    Args:
        ticker (str): String ticker in format that is acceptable to Yahoo Finance
        start_date (np.datetime64): Start date to get price data
        end_date (np.datetime64): End date to get price data

    Returns:
        pd.DataFrame: Dataframe containing open, close, high, low, split, dividend data for ticker from start_date to end_date
    """
    try:
        df = yf.Ticker(ticker).history(
            start=start_date, end=end_date, auto_adjust=False, rounding=False, debug=False)
    except KeyError:
        df = None
    except RuntimeError:
        logger.debug(f'-------  Yahoo! Finance is not working -------')
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
    try:
        stock = yf.Ticker(ticker)
        try:
            name = stock.info['longName']
        except (IndexError, KeyError) as e:
            print(f'-------  Ticker {ticker} not found -------')
            name = "NA"
    except (ValueError, AttributeError):
        logger.debug(f'-------  Ticker name {ticker} not found -------')
        name = "NA"
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
    try:
        stock = yf.Ticker(ticker)
        try:
            currency = stock.info['currency']
        except (IndexError, KeyError) as e:
            logger.debug(f'-------  Currency for {ticker} not found -------')
            currency = "NA"
    except (ValueError, AttributeError):
        logger.debug(f'-------  Ticker {ticker} not found (currency) -------')
        currency = "NA"
    return currency
