from datetime import datetime
from itertools import repeat
import logging
from os import path
from typing import List, Union

import pandas as pd
import numpy as np

from utils import data, irr

logger = logging.getLogger('pt_logger.Stock')
pd.options.display.float_format = '{:,.2f}'.format


class Portfolio():
    """
    Creates a Portfolio object which tracks information on the portfolio
    """

    TD_COLUMNS = ['Date', 'Ticker', 'Quantity', 'Price', 'Fees', 'Direction']
    INFO_COLUMNS = ['Ticker', 'Name', 'Quantity', 'LastPrice', '%LastChange', '$LastChange', 'CurrVal', 'IRR', '%UnRlGain', '%PF',
                    'AvgCost', 'Cost', '%CostPF', 'Dividends', 'RlGain', 'UnRlGain', 'TotalGain', 'Date']
    DEFAULT_FILE = 'data/data.pkl'
    DEFAULT_NAME_FILE = DEFAULT_FILE.split(".pkl")[0] + "_names.pkl"

    def __init__(self, trades: pd.DataFrame = None, currency: str = 'AUD', filename: str = DEFAULT_FILE, names_filename: str = DEFAULT_NAME_FILE):
        """
        Creates a new portfolio. Can accept a dataframe of trades


        Args:
            trades (pd.DataFrame, optional): Dataframe containing stock trades with the following columns:
            [Date, Ticker, Quantity, Price, Fees, Direction]. Defaults to None.
            filename (str, optional): File name and location to save pricing data. Defaults to 'data/data.pkl'.
            names_filename (str, optional): File name and location to save ticker / name data. Defaults to 'data/data_names.pkl'.

        Raises:
            ValueError: Raised if columns of dataframe passed in do not match required columns
        """

        self.positions = []
        self.trades_df = pd.DataFrame(columns=self.TD_COLUMNS)
        if trades is not None:
            self.add_trades(trades)
        self.filename = filename
        self.name_file = names_filename
        self.currency = currency

    def add_trades(self, trades: pd.DataFrame):
        """
        Adds dataframe of trades to portfolio

        Arguments:
            trade_df {pd.DataFrame} -- Dataframe containing stock trades with
            the following columns: [Date, Ticker, Quantity, Price, Fees, Direction]

        Raises:
            ValueError: Raised if columns of dataframe passed in do not match required columns
        """

        logger.debug(f'-------  Check if DF has correct columns  -------')

        if all(trades.columns == self.TD_COLUMNS):
            logger.debug('Concatenating trades to trade_df')
            self.trades_df = pd.concat(
                [self.trades_df, trades])
            self.trades_df.sort_values('Date', inplace=True)
        else:
            raise ValueError(
                f'Dataframe has incorrect columns. Please make sure dataframe has following columns in order: {self.TD_COLUMNS}')

    @property
    def info(self) -> pd.DataFrame:
        """
        Updates portfolio and returns portfolio dataframe

        Returns:
            Dataframe containing following information for each stock held in portfolio

        ['Ticker', 'Name', 'Quantity', 'LastPrice', '%LastChange','$LastChange', 'CurrVal', 'IRR', '%UnRlGain', '%PF',
                    'AvgCost', 'Cost', '%CostPF', 'Dividends', 'RlGain', 'UnRlGain', 'TotalGain', 'Date']
        """
        return self.info_date()

    def info_date(self, as_at_date: str = None,  min_days: int = -1, hide_zero_pos: bool = False, no_update: bool = False) -> pd.DataFrame:
        """
        Updates portfolio and returns portfolio dataframe as at a specified date (or as at today if no date provided)

        Args:
            as_at_date ({str}, optional): String representation of date in '%Y-%m-%d' format. Defaults to None.

        Returns:
            Dataframe: Portfolio information as at specified date containing following information for each stock held in portfolio

        ['Ticker', 'Name', 'Quantity', 'LastPrice', '%LastChange', '$LastChange', 'CurrVal', 'IRR', '%UnRlGain', '%PF',
                    'AvgCost', 'Cost', '%CostPF', 'Dividends', 'RlGain', 'UnRlGain', 'TotalGain', 'Date']
        """

        if as_at_date == None:
            as_at_date = pd.to_datetime('today')
        else:
            as_at_date = datetime.strptime(as_at_date, '%Y-%m-%d')

        logger.debug(
            'Get historical and current positions and merge with info dataframe')

        start = datetime.now()
        curr_df, split_df, div_df = self.curr_positions(
            self.trades_df['Ticker'].unique(), as_at_date, min_days, no_update)
        logger.info(f'curr_positions took {(datetime.now()-start)} to run')
        start = datetime.now()
        hist_df = self.hist_positions(as_at_date, split_df, div_df)
        logger.info(f'hist_positions took {(datetime.now()-start)}s to run')
        start = datetime.now()

        # calculate IRR and save in DF
        irr_df = self.calc_IRR(hist_df[['Date', 'Ticker', 'CF', 'CumQuan']].copy(), curr_df[[
                               'Date', 'Ticker', 'Close']].copy())
        logger.info(f'calc_IRR took {(datetime.now()-start)}s to run')
        start = datetime.now()

        # clean-up dataframe
        hist_df.drop(['Date', 'Quantity', 'Price', 'Fees', 'Direction', 'AdjQuan',
                      'CFBuy', 'CumCost', 'QBuy', 'CumBuyQuan', 'RlGain', 'CF', 'Dividends'], axis=1, inplace=True)
        hist_df = hist_df.groupby('Ticker').last().reset_index()
        hist_df.rename(columns={'CumQuan': 'Quantity',
                                'TotalRlGain': 'RlGain', 'CumDiv': 'Dividends'}, inplace=True)

        # drop rows where quantity is zero if argument passed is true
        if hide_zero_pos:
            hist_df = hist_df[hist_df['Quantity'].round(2) != 0]

        # Calculate total cost of each stock in portfolio
        hist_df['Cost'] = hist_df.Quantity * hist_df.AvgCost

        # merge hist_df, curr_df and irr_df
        info_df = hist_df.merge(curr_df, on='Ticker', how='left')
        info_df.sort_values('Ticker', inplace=True)
        info_df = self._add_total_row(
            info_df, 'Ticker', ['RlGain', 'Cost', 'Dividends'])
        info_df['Date'] = pd.to_datetime(info_df['Date'].fillna(pd.NaT))

        info_df = info_df.merge(irr_df, on='Ticker')
        info_df.reset_index(inplace=True, drop=True)

        logger.debug('Perform calculations on info dataframe and return')
        tot_index = len(info_df.index)-1

        info_df.rename(columns={'Close': 'LastPrice'}, inplace=True)
        info_df['%CostPF'] = info_df['Cost'] / info_df['Cost'][:-1].sum()
        info_df['CurrVal'] = info_df['Quantity'] * info_df['LastPrice']
        info_df.at[tot_index, 'CurrVal'] = info_df['CurrVal'].sum()
        info_df['$LastChange'] = info_df['CurrVal'] * \
            (1 - 1 / (1 + info_df['%LastChange']))
        info_df.at[tot_index, '$LastChange'] = info_df['$LastChange'].sum()
        info_df['%PF'] = info_df['CurrVal'] / info_df['CurrVal'][:-1].sum()
        info_df['UnRlGain'] = info_df['CurrVal'] + info_df['Cost']
        info_df['UnRlGain'].fillna(0, inplace=True)
        info_df['TotalGain'] = info_df['UnRlGain'] + \
            info_df['RlGain'] + info_df['Dividends']
        info_df['%UnRlGain'] = info_df['UnRlGain'] / -info_df['Cost']
        logger.info(f'clean-up took {(datetime.now()-start)} to run')
        start = datetime.now()

        # get full names of stock from ticker
        info_df.loc[0:tot_index-1,
                    'Name'] = self.stock_names(info_df.loc[0:tot_index-1, 'Ticker'])
        logger.info(f'stock_names took {(datetime.now()-start)} to run')

        # set up column in order of INFO_COLUMNS
        info_df = info_df[self.INFO_COLUMNS]
        return info_df

    def _add_total_row(self, df: pd.DataFrame, index: str, list_cols: List) -> pd.DataFrame:
        """
        Creates a total row at the end of given dataframe with totals for specified list of columns

        Args:
            df (pd.DataFrame): dataframe on which to provide totals row
            index (str): Index in string format. Total row will have index as 'Total'
            list_cols (List): List of columns for which totals need to be calculated

        Returns:
            pd.DataFrame: Returns df with a total row with totals for specified list_cols and 'Total' as index
        """

        df = df.append(pd.Series(name='Total'))
        df.loc['Total'] = df.loc[:, list_cols].sum(axis=0,)
        df.at['Total', index] = 'Total'
        return df

    def hist_positions(self, as_at_date: datetime, split_df: pd.DataFrame, div_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate historical positions for all stocks in Portfolio object (based on trades_df) as at given date

        Args:
            as_at_date (datetime): Date as at which to calculate the position of portfolio
            split_df (pd.DataFrame): Dataframe containing split data for stocks in portfolio
            div_df (pd.DataFrame): Dataframe containing dividend data for stocks in portfolio

        Returns:
            pd.DataFrame: Dataframe containing following information for each stock held in portfolio

        ['Date', 'Ticker', 'Quantity', 'Price', 'Fees', 'Direction', 'CF', 'AdjQuan', 'CumQuan', 'CFBuy', 'CumCost'
                    'QBuy', 'QBuyQuan', 'AvgCost', 'RlGain', 'Dividends', 'CumDiv', 'TotalRlGain']
        """

        # make copy of trades_df with trades only looking at trades before or equal to as_at_date
        start = datetime.now()
        hist_pos = self.trades_df[self.trades_df.Date <= as_at_date].copy()

        hist_pos.sort_values(['Date', 'Ticker'], inplace=True)
        hist_pos.Quantity = pd.to_numeric(hist_pos.Quantity)
        hist_pos.Price = pd.to_numeric(hist_pos.Price)
        hist_pos.Fees = pd.to_numeric(hist_pos.Fees)
        logger.info(
            f'hist_pos load and clean took {(datetime.now()-start)} to run')
        start = datetime.now()

        # adjust trades for splits
        for ticker in hist_pos['Ticker'].unique():
            splits = split_df[split_df['Ticker'] == ticker].copy()
            if not splits.empty:
                splits.sort_values('Date', ascending=False, inplace=True)
                for _, row in splits.iterrows():
                    hist_pos['Quantity'] = np.where(
                        (hist_pos['Date'] <= row['Date']) & (hist_pos['Ticker'] == ticker), round(hist_pos['Quantity'] * row['Stock Splits'], 0), hist_pos['Quantity'])
                    hist_pos['Price'] = np.where(
                        (hist_pos['Date'] <= row['Date']) & (hist_pos['Ticker'] == ticker), hist_pos['Price'] / row['Stock Splits'], hist_pos['Price'])
                    div_df['Dividends'] = np.where(
                        (div_df['Date'] <= row['Date']) & (div_df['Ticker'] == ticker), div_df['Dividends'] / row['Stock Splits'], div_df['Dividends'])

        logger.info(f'adjust for splits took {(datetime.now()-start)} to run')
        start = datetime.now()

        # create new columns to include cashflow, quantity (with buy / sell) and cumulative quantity by ticker
        hist_pos['CF'] = np.where(
            hist_pos.Direction == 'Buy', -1, 1)*(hist_pos.Quantity * hist_pos.Price) - hist_pos.Fees
        hist_pos['AdjQuan'] = np.where(
            hist_pos.Direction == 'Sell', -1, np.where(hist_pos.Direction == 'Div', 0, 1))*hist_pos.Quantity
        hist_pos['CumQuan'] = hist_pos.groupby('Ticker')['AdjQuan'].cumsum()

        # add dividend information
        for ticker in hist_pos['Ticker'].unique():
            dividends = div_df[div_df['Ticker'] == ticker].copy()
            if not dividends.empty:
                # add dividend info if shares held when dividends paid
                for _, row in dividends.iterrows():
                    try:
                        dt_div = hist_pos[(hist_pos['Date'] <= row['Date']) & (
                            hist_pos['Ticker'] == ticker)]['Date'].idxmax()
                        div_qty = hist_pos.loc[dt_div]['CumQuan']
                        # only add dividend if more than 0 shares held
                        if div_qty != 0:
                            hist_pos = hist_pos.append(pd.DataFrame([[row['Date'], ticker, div_qty,
                                                                      row['Dividends'], 0, 'Div', (div_qty*row['Dividends']), 0, div_qty]], columns=hist_pos.columns), ignore_index=True)
                    except ValueError:
                        pass  # do nothing if no shares are held during dividend period
        hist_pos.sort_values(['Ticker', 'Date'], inplace=True)
        logger.info(f'add divs took {(datetime.now()-start)} to run')
        start = datetime.now()

        # Calculate realised gains calculated as trade value less cost base + dividends. Cost base calculated based on average entry price (not adjusted for sales)
        hist_pos['CFBuy'] = np.where(
            hist_pos.Direction == 'Buy', hist_pos.CF, 0)
        hist_pos['CumCost'] = hist_pos.groupby('Ticker')['CFBuy'].cumsum()
        hist_pos['QBuy'] = np.where(
            hist_pos.Direction == 'Buy', hist_pos.Quantity, 0)
        hist_pos['CumBuyQuan'] = hist_pos.groupby('Ticker')['QBuy'].cumsum()

        # calculate average cost
        hist_pos['AvgCostRaw'] = hist_pos['CumCost'] / hist_pos['CumBuyQuan']
        # calculate average buy cost for stock (adjusting for sales to zero)
        hist_pos_grouped = hist_pos.groupby('Ticker')
        hist_pos = hist_pos_grouped.apply(self.calc_avg_price)
        hist_pos.reset_index(drop=True, inplace=True)
        # use AvgCost Adjusted where position completely sold out to reset costbase, otherwise use raw number
        hist_pos['AvgCost'] = np.where(
            hist_pos['grouping'] == 0, hist_pos['AvgCostRaw'], hist_pos['AvgCostAdj'])

        hist_pos['RlGain'] = np.where(
            hist_pos.Direction == 'Sell', hist_pos.CF + (hist_pos.AvgCost * hist_pos.Quantity), 0)
        hist_pos['Dividends'] = np.where(
            hist_pos.Direction == 'Div', hist_pos.CF, 0)
        hist_pos['CumDiv'] = hist_pos.groupby('Ticker')['Dividends'].cumsum()
        hist_pos['TotalRlGain'] = hist_pos.groupby('Ticker')['RlGain'].cumsum()
        logger.info(f'hist_pos clean up took {(datetime.now()-start)} to run')

        return hist_pos

    def calc_avg_price(self, df: pd.DataFrame) -> pd.DataFrame:
        df['grouping'] = df['CumQuan'].eq(0).shift().cumsum().fillna(
            0.)  # create group for each group of shares bought / sold
        DF = df.groupby('grouping', as_index=False).apply(
            lambda x: x.CFBuy.sum()/x.QBuy.sum()).reset_index(drop=True)
        DF.columns = ['grouping', 'AvgCostAdj']
        df = pd.merge(df, DF, how='left', on='grouping')
        return df

    def curr_positions(self, tickers: List, as_at_date: datetime, min_days: int, no_update: bool = False) -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Calculate current position for all stocks in Portfolio object (based on trades_df) as at given date

        Args:
            tickers (List): List of tickers in Portfolio
            as_at_date (datetime): Date as at  which to calculate portfolio position
            min_days (int, optional): Checks saved pickl file with price data and if price data was updated within min_days, then will not update data.
                Defaults to -1 which will update day prior and today's information (to update any intra-day information that may have been saved).

        Returns:
            Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Returns three dataframes:
                1. curr_df with current position / prices and value of stocks
                2. split_df containing split data for stocks in portfolio
                3. div_df containing dividend data for stocks in portfolio
        """
        start = datetime.now()
        min_date = self.trades_df["Date"].min()
        # check if file exists
        if path.isfile(self.filename):
            # if exists, load data into df. Run get price data for each ticker from last date to as_at_date + new tickers
            curr_df = pd.read_pickle(self.filename)
            logger.info(
                f'read price data took {(datetime.now()-start)} to run')
            start = datetime.now()
            # get tickers and dates for info in current database
            curr_df_data = curr_df.groupby('Ticker').agg(
                Min=('Date', 'min'), Max=('Date', 'max'))
            curr_df_data.reset_index(inplace=True)

            if not no_update:
                tickers_data, start_date_data, end_date_data = [], [], []

                # update for tickers in current database and in current trade list
                tickers_to_update = set(tickers).intersection(
                    set(curr_df['Ticker']))
                for _, row in curr_df_data.iterrows():
                    # check if current ticker in trade list
                    if row.Ticker in tickers_to_update:
                        last_date = row['Max']
                        logger.debug(
                            f'Ticker is {row.Ticker} with last date of {last_date} and as at date of {as_at_date} and days since update is {(as_at_date - last_date).days}')
                        if (as_at_date - last_date).days >= min_days:
                            tickers_data.append(row.Ticker)
                            start_date_data.append(last_date)
                            end_date_data.append(as_at_date)

                price_data = data.get_price_data(
                    tickers_data, start_date_data, end_date_data, repeat(self.currency, len(tickers_data)))
                price_data.reset_index(inplace=True)
                try:
                    price_data.set_index(['Ticker', 'Date'],
                                         inplace=True, verify_integrity=True)
                except ValueError:
                    price_data.drop_duplicates(
                        keep='last', inplace=True, subset=['Ticker', 'Date'])
                    price_data.set_index(['Ticker', 'Date'],
                                         inplace=True, verify_integrity=True)
                curr_df.set_index(['Ticker', 'Date'], inplace=True)
                curr_df = curr_df.combine_first(
                    price_data)  # adds new data / rows
                curr_df.update(price_data)  # updates data for existing rows
                curr_df.reset_index(inplace=True)

                # update for tickers not in existing dataset
                tickers_data, start_date_data, end_date_data = [], [], []
                tickers_to_update = set(tickers).difference(
                    set(curr_df['Ticker']))
                for ticker in tickers_to_update:
                    tickers_data.append(ticker)
                    start_date_data.append(min_date)
                    end_date_data.append(as_at_date)

                if(len(tickers_data) > 0):
                    price_data = data.get_price_data(
                        tickers_data, start_date_data, end_date_data, repeat(self.currency, len(tickers_data)))
                    price_data.reset_index(inplace=True)
                    curr_df = curr_df.append(price_data, ignore_index=True)
                curr_df.reset_index(inplace=True, drop=True)
        else:
            # if does not exist, run data.get_price_data for full period
            curr_df = data.get_price_data(
                tickers, repeat(min_date, len(tickers)), repeat(
                    as_at_date, len(tickers)), repeat(self.currency, len(tickers)))
            curr_df.reset_index(inplace=True)

        logger.info(f'update prices took {(datetime.now()-start)} to run')
        start = datetime.now()
        # Calculate last price change
        curr_df['PrevPrice'] = curr_df.groupby(
            'Ticker').shift(1, fill_value=0)['Close']
        curr_df['%LastChange'] = curr_df['Close'] / curr_df['PrevPrice'] - 1
        curr_df['%LastChange'].replace([np.inf, -np.inf], np.nan, inplace=True)
        logger.info(f'last change took {(datetime.now()-start)} to run')
        start = datetime.now()

        # sort and save file
        curr_df.sort_values(['Ticker', 'Date'], inplace=True)
        curr_df.to_pickle(self.filename)
        logger.info(f'save took {(datetime.now()-start)} to run')
        start = datetime.now()

        # create df with current positions as at the as_at_date
        price_df = curr_df[curr_df['Date'] <= as_at_date].copy()
        price_df = price_df[['Ticker', 'Date', 'Close', '%LastChange']]
        price_df = price_df.groupby('Ticker').last().reset_index()

        split_df = curr_df[[
            'Ticker', 'Date', 'Stock Splits']]
        split_df = curr_df[curr_df['Stock Splits'] != 0.0]

        div_df = curr_df[[
            'Ticker', 'Date', 'Dividends']]
        div_df = curr_df[curr_df['Dividends'] != 0.0]
        logger.info(f'curr_pos clean up took {(datetime.now()-start)} to run')
        start = datetime.now()

        return price_df, split_df, div_df

    def calc_IRR(self, hist_pos: pd.DataFrame, curr_p: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates IRR given two dataframes containing historical trades / cash flows and current position / value of stocks

        Args:
            hist_pos (pd.DataFrame): Dataframe containing historical trades. Should have ['Ticker', 'Date', 'CF']. CF should be cash flow where negative represents an outlay and positive an inflow
            curr_p (pd.DataFrame): Dataframe with current position by ticker. Should have ['Ticker, 'Date', 'Close'] where Close represents close price as at the date for relevant ticker

        Returns:
            pd.DataFrame: Returns Dataframe with ticker and IRRs for each stock held
        """

        # get current position for each ticker (i.e. current number of shares held)
        curr_pos = hist_pos.groupby('Ticker').last().reset_index()

        curr_p.set_index('Ticker', inplace=True)

        # add current value in new column in curr_pos dataframe
        for _, row in curr_pos.iterrows():
            ticker = row['Ticker']
            try:
                curr_p.at[ticker, 'CF'] = row['CumQuan'] * \
                    curr_p.loc[ticker, 'Close']
            except KeyError:
                logger.debug(f'No stock data for {ticker}')
                curr_p.at[ticker, 'CF'] = np.nan

        # clean up dataframes and reset indices before merge
        hist_pos.drop(['CumQuan'], axis=1, inplace=True)
        curr_p.drop(['Close'], axis=1, inplace=True)
        curr_p.reset_index(inplace=True)

        # merge curr_p into hist_pos as transactions
        CF_df = hist_pos.append(curr_p)
        CF_df.sort_values(['Date'], inplace=True)
        CF_df.reset_index(inplace=True, drop=True)

        # extract CFs and dates by ticker and pass through IRR function, store in dataframe
        grouped_CF_df = CF_df.groupby('Ticker')[['Date', 'CF']]
        IRR_df = pd.DataFrame(columns=['Ticker', 'IRR'])

        for name, _ in grouped_CF_df:
            stock_irr = irr.irr(grouped_CF_df.get_group(name).values.tolist())
            IRR_df = IRR_df.append(pd.DataFrame(
                [[name, stock_irr]], columns=IRR_df.columns))

        CF_df.drop('Ticker', axis=1, inplace=True)
        CF_df.dropna(inplace=True)
        total_irr = irr.irr(CF_df.values.tolist())
        IRR_df = IRR_df.append(pd.DataFrame(
            [['Total', total_irr]], columns=IRR_df.columns), ignore_index=True)

        # return DF with ticker and IRR
        return IRR_df

    def stock_names(self, tickers):
        # load pickle with names. If does not exist, create new dataframe
        if path.isfile(self.name_file):
            name_df = pd.read_pickle(self.name_file)
        else:
            name_df = pd.DataFrame(
                columns=['Ticker', 'Name'])
            name_df.set_index('Ticker', inplace=True)

        # check if ticker exists in loaded file. If not, get from yahoo finance
        for ticker in tickers:
            try:
                ticker_type = ticker.split('.')[1]
            except IndexError:
                ticker_type = None

            try:
                name = name_df.loc[ticker, 'Name']
            except KeyError:
                if ticker_type != 'CRYPTO':
                    name = data.get_name(ticker)
                else:
                    name = ticker.split('.')[0] + self.currency
                name_df.loc[ticker] = [name]

        # sort all except total row
        name_df.sort_values('Ticker', inplace=True)

        # Save pickle to allow for faster load times
        name_df.to_pickle(self.name_file)

        # deletes any names that are in the file but not in portfolio
        for ticker in set(name_df.index.tolist()).difference(set(tickers)):
            name_df.drop(index=ticker, inplace=True)

        return name_df['Name'].tolist()
