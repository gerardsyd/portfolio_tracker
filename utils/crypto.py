# IMPORTS
import pandas as pd
import numpy as np
import os
from datetime import datetime

from binance.client import Client
from binance.exceptions import BinanceAPIException

# get binance api key and secret from environment variables
binance_api_key = os.environ.get('BINANCE_API_KEY')
binance_api_secret = os.environ.get('BINANCE_API_SECRET')


def get_crypto_price(symbol: str, start_date: np.datetime64, end_date: np.datetime64, currency: str = 'AUD') -> pd.DataFrame:
    """
    Get crypto asset prices in required currency, returns dataframe in format required for data module

    Args:
        symbol (str): String ticker of crypto asset (e.g ETH)
        start_date (np.datetime64): start date for historical price data
        end_date (np.datetime64): start date for historical price data
        currency (str, optional): currency pair for conversion. Defaults to 'AUD'.

    Returns:
        pd.DataFrame: Dataframe containing open, close, high, low, split, dividend data for crypto-currency pair from start_date to end_date 
    """

    # check if required crypto is BTC in which case get pair with currency else get fx conversion
    if symbol == 'BTC':
        crypto_pair = symbol + currency
        crypto_prices = get_prices_from_API(crypto_pair, start_date, end_date)
        fx_prices = None
    else:
        crypto_pair = symbol + 'BTC'
        crypto_prices = get_prices_from_API(crypto_pair, start_date, end_date)
        fx_pair = 'BTC' + currency
        fx_prices = get_prices_from_API(fx_pair, start_date, end_date)
        fx_prices = fx_prices[['Close']]

    if isinstance(crypto_prices, pd.DataFrame) and isinstance(fx_prices, pd.DataFrame):
        # if both dataframes exist, then get crypto-currency pair
        data_df = pd.merge(left=crypto_prices, right=fx_prices, right_index=True,
                           left_index=True, how='inner')
        data_df['Close_x'] = data_df['Close_x']*data_df['Close_y']
        data_df['Open'] = data_df['Open']*data_df['Close_y']
        data_df['High'] = data_df['High']*data_df['Close_y']
        data_df['Low'] = data_df['Low']*data_df['Close_y']
        data_df.rename(columns={'Close_x': 'Close'}, inplace=True)
        data_df.drop(['close_time', 'quote_av', 'trades', 'tb_base_av',
                      'tb_quote_av', 'ignore', 'Close_y'], axis=1, inplace=True)
        data_df['Stock Splits'] = 0
        data_df['Dividends'] = 0
    elif isinstance(crypto_prices, pd.DataFrame) and symbol == 'BTC':
        # if required crypto is BTC, get BTC-currency pair
        data_df = crypto_prices.drop(['close_time', 'quote_av', 'trades', 'tb_base_av',
                                      'tb_quote_av', 'ignore'], axis=1)
        data_df['Stock Splits'] = 0
        data_df['Dividends'] = 0
    else:
        # crypto (or FX) not available in Binance and return nothing
        data_df = None
    return data_df


def get_prices_from_API(symbol_pair: str, start_date: np.datetime64, end_date: np.datetime64) -> pd.DataFrame:
    """
    Get crypto asset price history from Binance

    Args:
        symbol (str): String ticker of crypto asset (e.g ETH)
        start_date (np.datetime64): start date for historical price data
        end_date (np.datetime64): start date for historical price data

    Returns:
        pd.DataFrame: Dataframe containing binance data for crypto-currency pair from start_date to end_date 
    """

    client = Client(api_key=binance_api_key, api_secret=binance_api_secret)
    try:
        data = client.get_historical_klines(symbol_pair, client.KLINE_INTERVAL_1DAY, start_date.strftime(
            "%d %b %Y %H:%M:%S"), end_date.strftime("%d %b %Y %H:%M:%S"))
    except BinanceAPIException as e:
        print(f'symbol_pair is not available on Binance')
        return None
    else:
        data_df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close',
                                              'Volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
        data_df['Date'] = pd.to_datetime(data_df['Date'], unit='ms')
        data_df.set_index(['Date'], inplace=True, drop=True)
        data_df[data_df.columns] = data_df[data_df.columns].apply(
            pd.to_numeric, errors='coerce')
        return data_df


if __name__ == '__main__':
    df = get_crypto_price('XRP', datetime(2020, 8, 1),
                          datetime(2020, 8, 22), 'AUD')
    print(df)
