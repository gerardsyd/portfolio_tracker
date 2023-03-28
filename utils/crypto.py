import logging
import pandas as pd
import numpy as np
import os
from datetime import datetime
import requests

from binance.client import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger('pt_logger.Stock')

# get binance api key and secret from environment variables
binance_api_key = os.environ.get('BINANCE_API_KEY')
binance_api_secret = os.environ.get('BINANCE_API_SECRET')
nomic_api_key = os.environ.get('NOMIC_API_KEY')


def get_crypto_price(symbol: str, start_date: datetime, end_date: datetime, currency: str = 'AUD') -> pd.DataFrame:
    """
    Get crypto asset prices in required currency, returns dataframe in format required for data module

    Args:
        symbol (str): String ticker of crypto asset (e.g ETH)
        start_date (datetime): start date for historical price data
        end_date (datetime): start date for historical price data
        currency (str, optional): currency pair for conversion. Defaults to 'AUD'.

    Returns:
        pd.DataFrame: Dataframe containing open, close, high, low, split, dividend data for crypto-currency pair from start_date to end_date
    """

    # check if required crypto is BTC in which case get pair with currency else get fx conversion
    currency = 'USDT' if currency == 'USD' else currency
    if symbol == 'BTC':
        crypto_pair = symbol + currency
        crypto_prices = get_prices_from_API(crypto_pair, start_date, end_date)
        fx_prices = None
    else:
        crypto_pair = symbol + 'BTC'
        if currency == 'BTC':
            crypto_prices = get_prices_from_API(
                crypto_pair, start_date, end_date)
            fx_prices = None
        else:
            crypto_prices = get_prices_from_API(
                crypto_pair, start_date, end_date)
            fx_pair = 'BTC' + currency
            fx_prices = get_prices_from_API(fx_pair, start_date, end_date)
            fx_prices = fx_prices[['Close']]

    if isinstance(crypto_prices, pd.DataFrame) and isinstance(fx_prices, pd.DataFrame):
        # if both dataframes exist, then get crypto-currency pair
        data_df = pd.merge(left=crypto_prices, right=fx_prices, right_index=True,
                           left_index=True, how='inner')
        data_df['Close_x'] = data_df['Close_x'] * data_df['Close_y']
        data_df['Open'] = data_df['Open'] * data_df['Close_y']
        data_df['High'] = data_df['High'] * data_df['Close_y']
        data_df['Low'] = data_df['Low'] * data_df['Close_y']
        data_df.rename(columns={'Close_x': 'Close'}, inplace=True)
        data_df.drop(['close_time', 'quote_av', 'trades', 'tb_base_av',
                      'tb_quote_av', 'ignore', 'Close_y'], axis=1, inplace=True)
        data_df['Splits'] = 0
        data_df['Dividends'] = 0
    elif isinstance(crypto_prices, pd.DataFrame) and (symbol == 'BTC' or currency == 'BTC'):
        # if required crypto is BTC (or BTC is currency), get BTC-currency / crypto-BTC pair
        data_df = crypto_prices.drop(['close_time', 'quote_av', 'trades', 'tb_base_av',
                                      'tb_quote_av', 'ignore'], axis=1)
        data_df['Splits'] = 0
        data_df['Dividends'] = 0
    else:
        # crypto (or FX) not available in Binance and return nothing
        data_df = None
    return data_df


def get_prices_from_API(symbol_pair: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Get crypto asset price history from Binance

    Args:
        symbol (str): String ticker of crypto asset (e.g ETH)
        start_date (datetime): start date for historical price data
        end_date (datetime): start date for historical price data

    Returns:
        pd.DataFrame: Dataframe containing binance data for crypto-currency pair from start_date to end_date
    """

    client = Client(api_key=binance_api_key, api_secret=binance_api_secret)
    try:
        data = client.get_historical_klines(symbol_pair, client.KLINE_INTERVAL_1DAY, start_date.strftime(
            "%d %b %Y %H:%M:%S"), end_date.strftime("%d %b %Y %H:%M:%S"))
    except BinanceAPIException as e:
        print(f'{symbol_pair} is not available on Binance')
        return None
    else:
        data_df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close',
                                              'Volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        data_df['Date'] = pd.to_datetime(data_df['Date'], unit='ms')
        data_df.set_index(['Date'], inplace=True, drop=True)
        data_df[data_df.columns] = data_df[data_df.columns].apply(
            pd.to_numeric, errors='coerce')
        return data_df


def get_FX_rates(from_currency: str, to_currency: str, start_date: datetime, end_date: datetime):
    if to_currency == "USD":
        return get_prices_from_nomic(from_currency, start_date, end_date)
    if from_currency == "USD":
        df = get_prices_from_nomic(to_currency, start_date, end_date)
        df['Close'] = 1 / df['Close']
        return df
    else:
        from_df = get_prices_from_nomic(from_currency, start_date, end_date)
        to_df = get_prices_from_nomic(to_currency, start_date, end_date)
        merged_df = pd.merge(left=from_df, right=to_df, on='Date', how='inner')
        merged_df['Close'] = merged_df['Close_x'] / merged_df['Close_y']
        merged_df.drop(["Close_x", "Close_y"], inplace=True, axis=1)
        return merged_df


def get_prices_from_nomic(currency: str, start_date: datetime, end_date: datetime):
    start_date = start_date.isoformat("T") + "Z"
    end_date = end_date.isoformat("T") + "Z"
    url = f"https://api.nomics.com/v1/exchange-rates/history?key={nomic_api_key}&currency={currency}&start={start_date}&end={end_date}"
    json = requests.get(url).content
    df = pd.read_json(json)
    df.rename(columns={"timestamp": "Date", "rate": "Close"}, inplace=True)
    return df


if __name__ == '__main__':
    df = get_FX_rates('XRP', 'USD', datetime(
        2019, 6, 1), datetime(2020, 8, 22))
    # df = get_crypto_price('AUD', datetime(2019, 6, 1),
    #                       datetime(2020, 8, 22), 'USDT')
    print(df)
