from datetime import datetime, timedelta
from flask_login import UserMixin
import logging
import pandas as pd
import numpy as np
import traceback
from typing import List, Union

from sqlalchemy import func, select
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import aliased
from werkzeug.security import generate_password_hash, check_password_hash

from app import db, login
from utils import data, irr

logger = logging.getLogger('pt_logger')
TYPE_CATEGORIES = ['STOCK', 'FUND', 'CRYPTO', 'LOAN', 'CASH', '', 'FX']
INFO_COLUMNS = ['Ticker', 'Name', 'Quantity', 'LastPrice', '%LastChange', '$LastChange', 'CurrVal', 'IRR', '%UnRlGain', '%PF',
                'AvgCost', 'Cost', '%CostPF', 'Dividends', 'RlGain', 'UnRlGain', 'TotalGain', 'Date', 'Type']


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    trades = db.relationship('Trades', backref='user', lazy='dynamic')
    default_currency = db.Column(
        db.String(10), index=True, nullable=False, server_default="AUD")
    last_accessed = db.Column(db.DateTime, index=True)

    def __repr__(self):
        return f'<User {self.username} with default currency of {self.default_currency}>'

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    @login.user_loader
    def load_user(id):
        return User.query.get(int(id))

    def get_trades(self) -> pd.DataFrame:
        df = pd.read_sql(self.trades.statement, db.engine, index_col='id')
        df.drop(columns='user_id', inplace=True)
        df.rename(str.capitalize, axis=1, inplace=True)
        return df

    def get_ticker_trades(self, ticker) -> pd.DataFrame:
        df = pd.read_sql(Trades.query.filter(Trades.user_id == self.id, Trades.ticker == ticker).statement, db.engine)
        if df.empty:
            return None
        else:
            return df

    def add_trades(self, df: pd.DataFrame, append: bool = True):
        if append:
            # If append is true, get existing trades and append passed df to existing trades
            exist_df = self.get_trades()
            df.rename(str.capitalize, axis=1, inplace=True)
            if not exist_df.empty:
                df = pd.concat([exist_df, df], ignore_index=True, join='inner')

        # Update user_id to be current id
        df['User_id'] = self.id

        # Checks if tickers already in db and, if not, insert into DB
        df_tickers = df['Ticker'].unique()
        current_tickers = Stocks.current_tickers()
        for ticker in df_tickers:
            if ticker not in current_tickers:
                logger.debug(f'{ticker} not found, adding... ')
                stock = Stocks(ticker=ticker.upper())
                stock.update_name()
                currency = stock.update_currency()
                stock.update_last_updated(datetime.now())
                db.session.add(stock)
                if Stocks.check_stock_exists(f'{currency}{self.default_currency}=X.FX') is None:
                    logger.debug(f'{currency} not found, adding... ')
                    curr = Stocks(ticker=f'{currency.upper()}{self.default_currency.upper()}=X.FX')
                    curr.update_name()
                    curr.currency = self.default_currency
                    curr.update_last_updated(datetime.now())
                    db.session.add(curr)
                db.session.commit()

        # Remove existing trades and add all trades to DB. Rollback changes if any errors
        try:
            Trades.query.filter_by(user_id=self.id).delete()
            db.session.commit()
            df.to_sql('trades', db.engine, if_exists='append', index=False)
        except Exception as e:
            # db.session.rollback()
            logger.debug(f'-------------- Exception {traceback.print_exc()} --------------')

    def drop_trades(self):
        Trades.query.filter_by(user_id=self.id).delete()
        db.session.commit()

    def currencies(self) -> List:
        """
        Generates list of all currencies in User's portfolio other than crypto and default currency

        Returns:
            List: List of currencies (str) in User's portfolio other than crypto and default currency
        """
        stock_info = self.get_stock_info()
        stock_info['Raw'], stock_info['Type'] = zip(*stock_info['Ticker'].apply(data.split_ticker))
        stock_info = stock_info[stock_info['Raw'] != stock_info['Currency']]
        fx = stock_info['Currency'].unique()
        mask = (fx != 'NA') & (fx != self.default_currency)
        currencies = [f'{currency}{self.default_currency}=X.FX' for currency in list(fx[mask])]
        return currencies

    def update_last_accessed(self, date):
        self.last_accessed = date
        db.session.commit()

    def get_stock_info(self):
        tickers = self.get_trades()['Ticker'].unique()
        df = pd.read_sql(Stocks.query.filter(Stocks.ticker.in_(tickers)).statement, db.engine).rename(str.capitalize, axis=1)
        return df

    def get_tickers(self):
        return Trades.query.distinct(Trades.ticker).all()

    def info_date(self, start_date: datetime = None, as_at_date: datetime = None, hide_zero_pos: bool = False) -> pd.DataFrame:
        """
        Returns portfolio dataframe as at a specified date (or as at today if no date provided). Relies on stock price data in StockPrices and does not perform an update

        Args:
            start_date(datetime, optional): Datetime for the starting date of trades within portfolio. Defaults to None.
            as_at_date(datetime, optional): Datetime for the last trade date of trades within portfolio. Defaults to None.
            hide_zero_pos(bool, optional): Hide nil stock positions. Defaults to False.

        Returns:
            Dataframe: Portfolio information as at specified date containing following information for each stock held in portfolio
            ['Ticker', 'Name', 'Quantity', 'LastPrice', '%LastChange', '$LastChange', 'CurrVal', 'IRR', '%UnRlGain', '%PF',
            'AvgCost', 'Cost', '%CostPF', 'Dividends', 'RlGain', 'UnRlGain', 'TotalGain', 'Date']
        """
        # Set up variables
        # trades_df = self.get_trades()
        # tickers = list(trades_df['Ticker'].unique())
        tickers = [t[0] for t in db.session.query(Trades.ticker).distinct().all()]
        tickers.extend(self.currencies())  # add currencies to ticker list

        if start_date is None:
            start_date = db.session.query(func.min(StockPrices.date)).first()[0]
        if as_at_date is None:
            as_at_date = pd.to_datetime('today')

        logger.debug('Get splits and dividend information from stockprices for tickers')
        start = datetime.now()
        splits = StockPrices.query.filter(
            StockPrices.ticker.in_(tickers),
            StockPrices.splits != 0).order_by(StockPrices.date.asc()).all()
        divs = StockPrices.query.filter(
            StockPrices.ticker.in_(tickers),
            StockPrices.dividends != 0).order_by(StockPrices.date.asc()).all()
        logger.info(f'Splits and divs data took {(datetime.now()-start)} to run')

        logger.debug('Getting latest prices for tickers')
        curr_df = self.current_prices(tickers=tickers, as_at_date=as_at_date, last_change=True)
        logger.debug('Getting historical positions and calculating current holdings')
        hist_df = self.hist_positions(start_date=start_date, as_at_date=as_at_date, splits=splits, divs=divs)

        logger.debug('Getting IRR for each position')
        start = datetime.now()
        irr_df = self.calc_IRR(hist_df[['Date', 'Ticker', 'CF', 'CumQuan']].copy(), curr_df[['Date', 'Ticker', 'Close']].copy())
        logger.info(f'IRR data took {(datetime.now()-start)} to run')

        # clean-up dataframe
        logger.debug('Cleaning up dataframe')
        start = datetime.now()
        hist_df.drop(['Date', 'Quantity', 'Price', 'Fees', 'Direction', 'AdjQuan', 'CFBuy', 'CumCost', 'QBuy', 'CumBuyQuan', 'RlGain', 'CF', 'Dividends'], axis=1, inplace=True)
        hist_df = hist_df.groupby('Ticker').last().reset_index()
        hist_df.rename(columns={'CumQuan': 'Quantity', 'TotalRlGain': 'RlGain', 'CumDiv': 'Dividends'}, inplace=True)

        # drop rows where quantity is zero if argument passed is true
        if hide_zero_pos:
            hist_df = hist_df[hist_df['Quantity'].round(2) != 0]

        # Calculate total cost of each stock in portfolio
        hist_df['Cost'] = hist_df.Quantity * hist_df.AvgCost

        # merge hist_df and curr_df. Drop duplicate currency and fx columns and rename current fx column
        logger.debug('Merging histoical and current dataframes, adding total row and irr information')
        info_df = hist_df.merge(curr_df, on='Ticker', how='left').drop(columns=['Fx_x', 'Currency_x', 'Currency_y']).rename(columns={'Fx_y': 'Fx'})
        info_df.sort_values('Ticker', inplace=True)

        # Add total row to info_df
        info_df = self._add_total_row(info_df, 'Ticker', ['RlGain', 'Cost', 'Dividends'])
        info_df['Date'] = pd.to_datetime(info_df['Date'].fillna(pd.NaT))

        # merge with irr_df
        info_df = info_df.merge(irr_df, on='Ticker')
        info_df = info_df.reset_index(drop=True)

        logger.debug('Perform calculations on info dataframe and return')
        tot_index = len(info_df.index) - 1

        # create relevant columns including % of portfolio, current value, last change, unrealised gains, total gains
        info_df.rename(columns={'Close': 'LastPrice'}, inplace=True)
        info_df['%CostPF'] = info_df['Cost'] / info_df['Cost'][:-1].sum()
        info_df['CurrVal'] = info_df['Quantity'] * info_df['LastPrice'] * info_df['Fx']
        info_df.at[tot_index, 'CurrVal'] = info_df['CurrVal'].sum()
        info_df['$LastChange'] = info_df['CurrVal'] * (1 - 1 / (1 + info_df['%LastChange']))
        info_df.at[tot_index, '$LastChange'] = info_df['$LastChange'].sum()
        info_df['%PF'] = info_df['CurrVal'] / info_df['CurrVal'][:-1].sum()
        info_df['UnRlGain'] = info_df['CurrVal'] + info_df['Cost']
        info_df['UnRlGain'].fillna(0, inplace=True)
        info_df['TotalGain'] = info_df['UnRlGain'] + info_df['RlGain'] + info_df['Dividends']
        info_df['%UnRlGain'] = info_df['UnRlGain'] / -info_df['Cost']

        # get type of stock from ticker. Add names / date last accessed etc to info_df
        info_df = pd.merge(info_df, self.get_stock_info(), on='Ticker', sort=False, how='left')
        info_df['Raw'], info_df['Type'] = zip(*info_df['Ticker'].apply(data.split_ticker))
        info_df['Type'] = pd.Categorical(info_df['Type'], TYPE_CATEGORIES)

        # set up column in order of INFO_COLUMNS
        info_df = info_df[INFO_COLUMNS]
        logger.info(f'Clean up data took {(datetime.now()-start)} to run')
        return info_df

    def hist_positions(self, start_date: datetime, as_at_date: datetime, splits: List, divs: List, tickers: List = None) -> pd.DataFrame:
        """
        Calculate historical positions for all stocks in trades for user as at given date

        Args:
            start_date(datetime): Date as at which to calculate limit capital gains and dividends calculations(i.e. will return capital gains and dividends between start_date and as_at_date)
            as_at_date(datetime): Date as at which to calculate the position of portfolio
            splits(List): List containing split items from StockPrices for stocks in portfolio
            divs(List): List containing dividend items from StockPricesfor stocks in portfolio
            tickers(List): List of tickers for which to return historic positions

        Returns:
            pd.DataFrame: Dataframe containing following information for each stock held in portfolio

        ['Date', 'Ticker', 'Quantity', 'Price', 'Fees', 'Direction', 'CF', 'AdjQuan', 'CumQuan', 'CFBuy', 'CumCost'
                    'QBuy', 'QBuyQuan', 'AvgCost', 'RlGain', 'Dividends', 'CumDiv', 'TotalRlGain']
        """

        # make copy of trades_df with trades only looking at trades before or equal to as_at_date
        start = datetime.now()
        hist_pos_statement = db.session.query(Trades, Stocks.currency).where(Trades.date <= as_at_date, Trades.user_id == 1, Trades.ticker == Stocks.ticker).statement
        hist_pos = pd.read_sql(hist_pos_statement, db.engine).drop(columns=['id', 'user_id']).rename(str.capitalize, axis=1)
        if tickers is not None:
            hist_pos = hist_pos[hist_pos['Ticker'].isin(tickers)].copy()

        hist_pos.sort_values(['Date', 'Ticker'], inplace=True)
        logger.info(f'hist_pos load and clean took {(datetime.now()-start)} to run')

        start = datetime.now()
        # adjust trades for splits
        for split in splits:
            hist_pos['Quantity'] = np.where(
                (hist_pos['Date'] <= split.date) & (hist_pos['Ticker'] == split.ticker),
                round(hist_pos['Quantity'] * float(split.splits), 0),
                hist_pos['Quantity'])
            hist_pos['Price'] = np.where(
                (hist_pos['Date'] <= split.date) & (hist_pos['Ticker'] == split.ticker),
                hist_pos['Price'] / float(split.splits),
                hist_pos['Price'])
            # div_df['Dividends'] = np.where(
            #             (div_df['Date'] <= split.date) & (div_df['Ticker'] == split.ticker), div_df['Dividends'] / split.splits, div_df['Dividends'])

        logger.info(f'adjust for splits took {(datetime.now()-start)} to run')
        # logger.info(hist_pos)
        start = datetime.now()
        # create new columns to include cashflow, quantity (with buy / sell) and cumulative quantity by ticker
        # hist_pos['CF'] = np.where(hist_pos.Direction == 'Buy', -1, 1) * (hist_pos.Quantity * hist_pos.Price * hist_pos.Fx) - (hist_pos.Fees * hist_pos.Fx)
        hist_pos['AdjQuan'] = np.where(hist_pos.Direction == 'Sell', -1, np.where(hist_pos.Direction == 'Div', 0, 1)) * hist_pos.Quantity
        hist_pos['CumQuan'] = hist_pos.groupby('Ticker')['AdjQuan'].cumsum()

        # add dividend information
        for dividend in divs:
            dt_div = hist_pos[(hist_pos['Date'] <= dividend.date) & (hist_pos['Ticker'] == dividend.ticker)]['Date'].index
            if not dt_div.empty:
                div_qty = hist_pos.at[dt_div[-1], 'CumQuan']

                # only add dividend if more than 0 shares held
                if div_qty != 0:
                    div_data = {'Date': dividend.date,
                                'Ticker': dividend.ticker,
                                'Quantity': div_qty,
                                'Price': float(dividend.dividends),
                                'Fees': 0,
                                'Direction': 'Div',
                                'CF': (div_qty * float(dividend.dividends)),
                                'AdjQuan': 0,
                                'CumQuan': div_qty,
                                'Currency': hist_pos.at[dt_div[0], 'Currency']}
                    hist_pos = pd.concat([hist_pos, pd.DataFrame(div_data, index=[0])], ignore_index=True)
                    hist_pos.sort_values(['Ticker', 'Date'], inplace=True)
        logger.info(f'add divs took {(datetime.now()-start)} to run')
        # logger.info(hist_pos)

        start = datetime.now()
        hist_pos = self.get_fx(hist_pos)
        # Adjust for currency
        # hist_pos['Fx'] = np.where(hist_pos['Currency'] == self.default_currency, float(1), np.NaN)
        # for index, row in hist_pos.iterrows():
        #     type = data.split_ticker(row['Ticker'])[1]
        #     if np.isnan(row['Fx']) and type not in ['CRYPTO', 'CASH', 'LOAN']:
        #         hist_pos.at[index, 'Fx'] = float(db.session.query(StockPrices.close).filter(
        #             StockPrices.ticker == f'{row["Currency"]}{self.default_currency}=X.FX').filter(StockPrices.date == row['Date']).scalar())
        #     elif np.isnan(row['Fx']) and type in ['CRYPTO', 'CASH', 'LOAN']:
        #         hist_pos.at[index, 'Fx'] = float(1)
        logger.info(f'hist_pos fx took {(datetime.now()-start)} to run')

        start = datetime.now()
        # Calculate realised gains calculated as trade value less cost base + dividends. Cost base calculated based on average entry price (not adjusted for sales)
        hist_pos['CF'] = np.where(hist_pos.Direction == 'Buy', -1, 1) * (hist_pos.Quantity * hist_pos.Price * hist_pos.Fx) - (hist_pos.Fees * hist_pos.Fx)
        hist_pos['CFBuy'] = np.where(hist_pos.Direction == 'Buy', hist_pos.CF, 0)
        hist_pos['CumCost'] = hist_pos.groupby('Ticker')['CFBuy'].cumsum()
        hist_pos['QBuy'] = np.where(hist_pos.Direction == 'Buy', hist_pos.Quantity, 0)
        hist_pos['CumBuyQuan'] = hist_pos.groupby('Ticker')['QBuy'].cumsum()

        # calculate average cost
        hist_pos['AvgCostRaw'] = hist_pos['CumCost'] / hist_pos['CumBuyQuan']
        # calculate average buy cost for stock (adjusting for sales to zero)
        hist_pos_grouped = hist_pos.groupby('Ticker')
        hist_pos = hist_pos_grouped.apply(self.calc_avg_price)
        hist_pos.reset_index(drop=True, inplace=True)
        # use AvgCost Adjusted where position completely sold out to reset costbase, otherwise use raw number
        hist_pos['AvgCost'] = np.where(hist_pos['grouping'] == 0, hist_pos['AvgCostRaw'], hist_pos['AvgCostAdj'])
        hist_pos.to_csv('data/Cf_avgcost.csv')

        hist_pos['RlGain'] = np.where(((hist_pos.Direction == 'Sell') & (hist_pos.Date >= start_date)), hist_pos.CF + (hist_pos.AvgCost * hist_pos.Quantity), 0)
        hist_pos['Dividends'] = np.where(hist_pos.Direction == 'Div', hist_pos.CF, 0)
        hist_pos['CumDiv'] = hist_pos.groupby('Ticker')['Dividends'].cumsum()
        hist_pos['TotalRlGain'] = hist_pos.groupby('Ticker')['RlGain'].cumsum()
        logger.info(f'hist_pos clean up took {(datetime.now()-start)} to run')
        # logger.info(hist_pos)

        return hist_pos

    def get_fx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns dataframe with Fx column

        Args:
            df (pd.DataFrame): dataframe with Currency, Ticker and Date columns

        Returns:
            pd.DataFrame: returns df with a Fx column

        """
        df['Fx'] = np.where(df['Currency'] == self.default_currency, float(1), np.NaN)
        for index, row in df.iterrows():
            type = data.split_ticker(row['Ticker'])[1]
            if np.isnan(row['Fx']) and type not in ['CRYPTO', 'CASH', 'LOAN']:
                fx_rate = db.session.query(StockPrices.close).filter(
                    StockPrices.ticker == f'{row["Currency"]}{self.default_currency}=X.FX').filter(StockPrices.date == row['Date']).scalar() or float(1)
                df.at[index, 'Fx'] = float(fx_rate)
            elif np.isnan(row['Fx']) and type in ['CRYPTO', 'CASH', 'LOAN']:
                df.at[index, 'Fx'] = float(1)
        return df

    def calc_IRR(self, hist_pos: pd.DataFrame, curr_p: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates IRR given two dataframes containing historical trades / cash flows and current position / value of stocks

        Args:
            hist_pos(pd.DataFrame): Dataframe containing historical trades. Should have['Ticker', 'Date', 'CF']. CF should be cash flow where negative represents an outlay and positive an inflow
            curr_p(pd.DataFrame): Dataframe with current position by ticker. Should have['Ticker, 'Date', 'Close'] where Close represents close price as at the date for relevant ticker

        Returns:
            pd.DataFrame: Returns Dataframe with ticker and IRRs for each stock held
        """

        start = datetime.now()
        # get current position for each ticker (i.e. current number of shares held)
        curr_pos = hist_pos.groupby('Ticker').last().reset_index()
        curr_p = curr_p.set_index('Ticker').astype({'Close': 'float'})

        # add current value in new column in curr_pos dataframe
        for _, row in curr_pos.iterrows():
            ticker = row['Ticker']
            try:
                curr_p.at[ticker, 'CF'] = row['CumQuan'] * curr_p.loc[ticker, 'Close']
            except KeyError:
                logger.debug(f'IRR Calculation: No stock data for {ticker}')
                curr_p.at[ticker, 'CF'] = np.nan

        # clean up dataframes and reset indices before merge
        hist_pos.drop(['CumQuan'], axis=1, inplace=True)
        curr_p.drop(['Close'], axis=1, inplace=True)
        curr_p.reset_index(inplace=True)

        # merge curr_p into hist_pos as transactions
        CF_df = pd.concat([hist_pos, curr_p])
        CF_df.sort_values(['Date'], inplace=True)
        CF_df.reset_index(inplace=True, drop=True)

        # extract CFs and dates by ticker and pass through IRR function, store in dataframe
        grouped_CF_df = CF_df.groupby('Ticker')[['Date', 'CF']]
        IRR_df = pd.DataFrame(columns=['Ticker', 'IRR'])

        for name, _ in grouped_CF_df:
            stock_irr = irr.irr(grouped_CF_df.get_group(name).values.tolist())
            IRR_df = pd.concat([IRR_df, pd.DataFrame([[name, stock_irr]], columns=IRR_df.columns)])

        CF_df.drop('Ticker', axis=1, inplace=True)
        CF_df.dropna(inplace=True)
        total_irr = irr.irr(CF_df.values.tolist())
        IRR_df = pd.concat([IRR_df, pd.DataFrame([['Total', total_irr]], columns=IRR_df.columns)], ignore_index=True)

        logger.info(f'IRR calc took {(datetime.now()-start)} to run')
        return IRR_df

    def current_prices(self, tickers: List, as_at_date: datetime, last_change: bool = False) -> pd.DataFrame:
        """
        Gets the latest prices for tickers as at the given date. If last_change is true, also returns the change in price from the previous price for a given ticker

        Args:
            tickers (List): List of tickers for which to get current prices
            as_at_date (datetime): Date at which to generate prices
            last_change (bool, optional): If true, returns change in price from the previous price in a column. Defaults to False.

        Returns:
            pd.DataFrame: Dataframe with columns: Ticker, Date, Close as at Date and, if last_change is true, %LastChange which shows % change in price
        """

        start = datetime.now()
        # add curr_fx col
        curr_p = self._current_prices(tickers, [as_at_date] * len(tickers), ['Ticker', 'Close', 'Date', 'Currency'])

        if last_change:
            logger.debug("Getting previous day price")
            tickers = curr_p['Ticker'].to_list()
            prev_dates = [(d - timedelta(days=1)) for d in curr_p['Date']]
            prev_df = self._current_prices(tickers, prev_dates, ['Ticker', 'Close']).rename(columns={'Close': 'PrevClose'})
            curr_p = curr_p.merge(prev_df, on='Ticker', how='left')
            curr_p['%LastChange'] = curr_p['Close'] / curr_p['PrevClose'] - 1
            curr_p.drop(columns='PrevClose', inplace=True)

        curr_p = self.get_fx(curr_p)
        logger.info(f'current prices read took {(datetime.now()-start)} to run')
        return curr_p

    def _current_prices(self, tickers: List[str], as_at_dates: List[datetime], columns: List[str]) -> pd.DataFrame:
        COL_TYPES = {
            'Ticker': 'str',
            'Open': 'float',
            'High': 'float',
            'Low': 'float',
            'Close': 'float',
            'Volume': 'float',
            'Adjclose': 'float',
            'Dividends': 'float',
            'Splits': 'float',
            'Currency': 'str'
        }
        col_types = {key: value for key, value in COL_TYPES.items() if key in columns}

        latest_prices = []
        for ticker, as_at_date in zip(tickers, as_at_dates):
            max_date = db.session.query(func.max(StockPrices.date)).filter(
                StockPrices.ticker == ticker,
                StockPrices.date <= as_at_date).scalar()

            if max_date:
                latest_price = db.session.query(StockPrices, Stocks.currency).join(
                    Stocks, StockPrices.ticker == Stocks.ticker).filter(
                    StockPrices.ticker == ticker,
                    StockPrices.date == max_date).first()

                if latest_price:
                    latest_prices.append(latest_price)
                else:
                    logger.debug(f'No price for {ticker} on {as_at_date}')
            else:
                logger.debug(f'No price for {ticker} on {as_at_date}')
        lp = [latest_price[0].__dict__ for latest_price in latest_prices]
        for i, latest_price in enumerate(latest_prices):
            lp[i].update({'currency': latest_price[1]})
        df = pd.DataFrame(lp).rename(str.capitalize, axis=1)[columns].astype(col_types)
        return df

    @staticmethod
    def calc_avg_price(df: pd.DataFrame) -> pd.DataFrame:
        # create group for each group of shares bought / sold
        df['grouping'] = df['CumQuan'].eq(0).shift().cumsum().fillna(0)
        avg_price_df = df.groupby('grouping', as_index=False).apply(lambda x: x.CFBuy.sum() / x.QBuy.sum()).reset_index(drop=True)
        avg_price_df.columns = ['grouping', 'AvgCostAdj']
        df = df.merge(avg_price_df, how='left', on='grouping')
        return df

    def _add_total_row(self, df: pd.DataFrame, index: str, list_cols: List) -> pd.DataFrame:
        """
        Creates a total row at the end of given dataframe with totals for specified list of columns

        Args:
            df(pd.DataFrame): dataframe on which to provide totals row
            index(str): Index in string format. Total row will have index as 'Total'
            list_cols(List): List of columns for which totals need to be calculated

        Returns:
            pd.DataFrame: Returns df with a total row with totals for specified list_cols and 'Total' as index
        """
        df = pd.concat([df, pd.Series(name='Total', dtype=float)])
        df.loc['Total'] = df.loc[:, list_cols].sum(axis=0)
        df.at['Total', index] = 'Total'
        return df

    def price_history(self, ticker: str, start_date: datetime, as_at_date: datetime, period: str) -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        prices_df = pd.read_sql(StockPrices.query.filter(StockPrices.ticker == ticker).statement, db.engine).rename(str.capitalize, axis=1)

        splits = StockPrices.query.filter(StockPrices.ticker == ticker, StockPrices.splits != 0).order_by(StockPrices.date.asc()).all()
        divs = StockPrices.query.filter(StockPrices.ticker == ticker, StockPrices.dividends != 0).order_by(StockPrices.date.asc()).all()
        logger.info(divs)

        hist_df = self.hist_positions(start_date=start_date, as_at_date=as_at_date, splits=splits, divs=divs, tickers=[ticker])
        prices_df = prices_df[prices_df['Date'] >= hist_df['Date'].min()]
        if period == 'A':
            p_hist_df = prices_df.groupby(prices_df['Date'].dt.year).tail(1).copy()
        elif period == 'M':
            p_hist_df = prices_df.groupby([prices_df['Date'].dt.year, prices_df['Date'].dt.month]).tail(1).copy()
        elif period == 'D':
            p_hist_df = prices_df.copy()
        else:
            raise ValueError('Please insert either A (annual), M (monthly) or D (daily) for period')

        p_hist_df['Quantity'] = None
        p_hist_df['AvgCost'] = None
        p_hist_df['Dividends'] = None
        p_hist_df['RlGain'] = None

        for idx, row in p_hist_df.iterrows():
            try:
                pos_at_date = hist_df[hist_df['Date'] <= row['Date']].iloc[-1]
                p_hist_df.loc[idx, 'Quantity'] = pos_at_date['CumQuan']
                p_hist_df.loc[idx, 'AvgCost'] = pos_at_date['AvgCost']
                p_hist_df.loc[idx, 'Dividends'] = pos_at_date['Dividends'] if pos_at_date['Date'] == row['Date'] else 0
                p_hist_df.loc[idx, 'CumDiv'] = pos_at_date['CumDiv']
                p_hist_df.loc[idx, 'RlGain'] = pos_at_date['TotalRlGain']
            except IndexError:
                logger.info(f'No price data for {ticker} prior to {row["Date"]}')

        p_hist_df['CurrVal'] = p_hist_df['Close'] * p_hist_df['Quantity']
        p_hist_df['Cost'] = p_hist_df['AvgCost'] * p_hist_df['Quantity']
        p_hist_df['UnRlGain'] = p_hist_df['CurrVal'] + p_hist_df['Cost']
        p_hist_df['TotalGain'] = p_hist_df['UnRlGain'] + p_hist_df['RlGain'] + p_hist_df['Dividends']

        div_df = p_hist_df[['Ticker', 'Date', 'Dividends']]
        div_df = div_df[div_df['Dividends'] != 0.0].dropna(subset='Dividends')
        split_df = prices_df[['Ticker', 'Date', 'Splits']]
        split_df = split_df[split_df['Splits'] != 0.0].dropna(subset='Splits')
        p_hist_df = p_hist_df.drop(columns=['Dividends'])

        return p_hist_df, div_df, split_df

    def update_prices(self, as_at_date: datetime, tickers: List = None, min_days: int = -1) -> pd.DataFrame:
        start = datetime.now()
        if tickers is None:
            tickers = list(self.get_trades()['Ticker'].unique())
            tickers.extend(self.currencies())
            logger.debug(tickers)

        pf_min_date = self.get_trades()['Date'].min()
        # for tickers already in stockprices, work out update period
        dates_statement = db.session.query(StockPrices.ticker, func.max(StockPrices.date).label('max_date')) \
            .filter(StockPrices.ticker.in_(tickers)) \
            .group_by(StockPrices.ticker).statement
        df = pd.read_sql(dates_statement, db.engine)
        df['end_date'] = as_at_date
        df['start_date'] = df['max_date'] + timedelta(days=min_days)
        df = df[df['start_date'] < df['max_date']]
        df.drop(columns=['max_date'], inplace=True)

        tickers_in_db = set(StockPrices.current_tickers())
        new_tickers = set(tickers).difference(tickers_in_db)
        for ticker in new_tickers:
            if self.get_ticker_trades(ticker) is None:
                if data.split_ticker(ticker)[1] == 'FX':
                    curr = ticker.replace(f'{self.default_currency}=X.FX', '')
                    min_date = db.session.query(func.min(Trades.date)).\
                        join(Stocks, Trades.ticker == Stocks.ticker).\
                        filter(Trades.user_id == self.id).\
                        filter(Stocks.currency == curr).\
                        first()[0]
                else:
                    min_date = pf_min_date
            else:
                min_date = pf_min_date if self.get_ticker_trades(ticker) is None else self.get_ticker_trades(ticker)['date'].min()
            df = pd.concat([df, pd.DataFrame({'ticker': ticker, 'start_date': min_date, 'end_date': as_at_date}, index=[0])], ignore_index=True)

        # get price data for all tickers as needed and reset index. Replace NaN with None for insertion into SQL
        prices = data.get_price_data(df['ticker'], df['start_date'], df['end_date'], [self.default_currency] * len(df['ticker'])).reset_index()
        prices = prices.replace(np.NaN, None)

        # iterate through rows in prices to update SQL database with updated prices where already existing or to append new data
        for _, row in prices.iterrows():
            price_data = {
                'ticker': row['Ticker'],
                'date': row['Date'],
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Close'],
                'adjclose': row['Adjclose'],
                'volume': row['Volume'],
                'dividends': row['Dividends'],
                'splits': row['Splits']
            }
            stock_price = StockPrices.query.filter_by(ticker=price_data['ticker'], date=price_data['date']).first()
            if stock_price:
                # stock price exists, updating with new information
                logger.debug(f'Updating {row["Ticker"]=} with {row["Date"]=}')
                for key, value in price_data.items():
                    setattr(stock_price, key, value)
            else:
                # Stock price doesn't exist so creating new price and adding to db
                logger.debug(f'Adding {row["Ticker"]=} with {row["Date"]=}')
                stock_price = StockPrices(**price_data)
                db.session.add(stock_price)

        # commit all changes
        db.session.commit()

        logger.info(f'price update took {(datetime.now()-start)} to run')
        return prices


class Trades(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', name='fk_trades_user_id'))
    date = db.Column(db.DateTime, index=True)
    ticker = db.Column(db.String(20), db.ForeignKey('stocks.ticker', name='fk_trades_ticker'), nullable=False)
    quantity = db.Column(db.Numeric(20, 10), index=True)
    price = db.Column(db.Numeric(20, 10), index=True)
    fees = db.Column(db.Numeric(20, 10), index=True)
    direction = db.Column(db.String(10), index=True)
    pf_price = db.Column(db.Numeric(20, 10), index=True)
    pf_shares = db.Column(db.Numeric(20, 10), index=True)
    # fx = db.Column(db.Numeric(20, 10), index=True)

    def __repr__(self):
        return f'<{self.id}: {self.direction} trade on {self.date} for {self.quantity} of {self.ticker} at {self.price}>'


class Stocks(db.Model):
    ticker = db.Column(db.String(20), primary_key=True)
    name = db.Column(db.String(60), index=True)
    currency = db.Column(db.String(10), index=True)
    last_updated = db.Column(db.DateTime(), index=True)

    def __repr__(self):
        return f'<{self.ticker}: {self.name} and quoted in {self.currency} and last updated on {self.last_updated}>'

    def update_name(self):
        name = data.get_name(self.ticker)
        if name is None:
            self.name = "NA"
        else:
            self.name = name[:60]
        db.session.commit()

    def update_currency(self):
        self.currency = data.get_currency(self.ticker)
        db.session.commit()
        return self.currency

    def update_last_updated(self, date: datetime):
        self.last_updated = date
        db.session.commit()

    def check_stock_exists(ticker):
        return Stocks.query.get(ticker.upper())

    def return_currency(ticker):
        return Stocks.query.get(ticker).currency

    @classmethod
    def current_tickers(cls):
        return [stock.ticker for stock in cls.query.all()]


class StockPrices(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), db.ForeignKey('stocks.ticker', name='fk_prices_ticker'), nullable=False)
    date = db.Column(db.DateTime, index=True)
    open = db.Column(db.Numeric(40, 20), index=True)
    high = db.Column(db.Numeric(40, 20), index=True)
    low = db.Column(db.Numeric(40, 20), index=True)
    close = db.Column(db.Numeric(40, 20), index=True)
    volume = db.Column(db.Numeric(40, 20), index=True)
    adjclose = db.Column(db.Numeric(40, 20), index=True)
    dividends = db.Column(db.Numeric(40, 20), index=True)
    splits = db.Column(db.Numeric(20, 10), index=True)

    def __repr__(self):
        return f'<{self.ticker} price on {self.date}: {self.close}>'

    @classmethod
    def current_tickers(cls):
        return [result[0] for result in cls.query.with_entities(cls.ticker).distinct().all()]

    @hybrid_property
    def prev_close(self):
        """Calculate the previous close price for this ticker"""
        prev_price = StockPrices.query.filter(
            StockPrices.ticker == self.ticker,
            StockPrices.date < self.date
        ).order_by(StockPrices.date.desc()).first()

        if prev_price is None:
            return None
        else:
            return prev_price.close

    @prev_close.expression
    def prev_close(cls):
        max_date = select(StockPrices.date) \
            .where(StockPrices.ticker == cls.ticker) \
            .where(StockPrices.date < cls.date) \
            .as_scalar()
        pp = aliased(StockPrices)
        prev_close = (
            select(pp.close)
            .where(pp.ticker == cls.ticker)
            .where(pp.date == max_date)
            .as_scalar()
        )

        return prev_close
