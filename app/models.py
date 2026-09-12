from datetime import datetime, timedelta, date
from collections import Counter
from flask_login import UserMixin
import logging
import math
import pandas as pd
import numpy as np
import traceback
from typing import List, Union, Optional, Tuple, Dict, Set

from sqlalchemy import func, select, UniqueConstraint
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.dialects.mysql import insert
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

    def set_password(self, password: str):
        """
        Set password for user

        Args:
            password (string): password for user
        """
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """
        Checks if password for user is correct

        Args:
            password (str): string password to check

        Returns:
            bool: True if password is correct, False if incorrect
        """
        return check_password_hash(self.password_hash, password)

    @login.user_loader
    def load_user(id):
        return User.query.get(int(id))

    def get_trades(self) -> pd.DataFrame:
        """
        Gets trades for user

        Returns:
            pd.DataFrame: dataframe containing all trades for user
        """
        df = pd.read_sql(self.trades.statement, db.engine, index_col='id')
        df.drop(columns='user_id', inplace=True)
        df.rename(str.capitalize, axis=1, inplace=True)
        return df

    def get_ticker_trades(self, ticker: str) -> pd.DataFrame:
        """
        Gets trades for user for a specified ticker

        Args:
            ticker (str): ticker for which to obtain user trades

        Returns:
            pd.DataFrame: trades completed by the user for the specified ticker
        """
        df = pd.read_sql(Trades.query.filter(
            Trades.user_id == self.id, Trades.ticker == ticker).statement, db.engine)
        if df.empty:
            return None
        else:
            return df

    def add_trades(self, df: pd.DataFrame, append: bool = True):
        """
        Add trades to user

        Args:
            df (pd.DataFrame): dataframe taking trades for user
            append (bool, optional): If true, append to existing trades, otherwise replace. Defaults to True.
        """
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
                    curr = Stocks(
                        ticker=f'{currency.upper()}{self.default_currency.upper()}=X.FX')
                    curr.update_name()
                    curr.currency = self.default_currency
                    curr.update_last_updated(datetime.now())
                    db.session.add(curr)
                db.session.commit()

        # Remove existing trades and add all trades to DB. Rollback changes if any errors
        min_trade_date = None
        if 'Date' in df.columns:
            try:
                min_trade_date = pd.to_datetime(df['Date']).min()
            except Exception:
                min_trade_date = None

        try:
            Trades.query.filter_by(user_id=self.id).delete()
            db.session.commit()
            df.to_sql('trades', db.engine, if_exists='append', index=False)
            self.invalidate_nav_from(min_trade_date)
        except Exception:
            # db.session.rollback()
            logger.debug(
                f'-------------- Exception {traceback.print_exc()} --------------')

    def drop_trades(self):
        """
        Drops all trades for user
        """
        Trades.query.filter_by(user_id=self.id).delete()
        db.session.commit()
        self.mark_all_monthly_nav_cache_stale()

    def mark_all_monthly_nav_cache_stale(self):
        """
        Flag all cached monthly NAV rows for this user as needing refresh.
        """
        updated = PortfolioMonthlyNav.query.filter_by(user_id=self.id).update(
            {'needs_refresh': True},
            synchronize_session=False
        )
        if updated:
            db.session.commit()

    def mark_monthly_nav_cache_stale(self, start_date: datetime):
        """
        Flag cached monthly NAV rows from the month containing start_date onwards.
        """
        if start_date is None:
            self.mark_all_monthly_nav_cache_stale()
            return

        month_end = pd.Timestamp(start_date).to_period('M').to_timestamp('M').date()
        updated = PortfolioMonthlyNav.query.filter(
            PortfolioMonthlyNav.user_id == self.id,
            PortfolioMonthlyNav.month_end >= month_end
        ).update({'needs_refresh': True}, synchronize_session=False)
        if updated:
            db.session.commit()

    def invalidate_nav_from(self, start_date: Optional[datetime]) -> None:
        """
        Clear stored trade-level NAV data from start_date onwards and mark cache rows stale.
        """
        if start_date is not None and isinstance(start_date, pd.Timestamp):
            start_date = start_date.to_pydatetime()

        if start_date is None:
            logger.info('Invalidating NAV data for all trades (start_date=None) for user %s', self.id)
            query = Trades.query.filter(Trades.user_id == self.id)
        else:
            logger.info('Invalidating NAV data for user %s from %s onwards', self.id, start_date)
            query = Trades.query.filter(
                Trades.user_id == self.id,
                Trades.date >= start_date
            )

        affected = query.update(
            {
                Trades.pf_price: None,
                Trades.pf_shares: None
            },
            synchronize_session=False
        )

        if affected:
            logger.debug('Cleared stored NAV columns for %s trades', affected)
        else:
            logger.debug('No trade rows required NAV invalidation for user %s', self.id)

        # Commit via the cache stale marking to keep operations in one transaction
        self.mark_monthly_nav_cache_stale(start_date)

    def currencies(self) -> List:
        """
        Generates list of all currencies in User's portfolio other than crypto and default currency

        Returns:
            List: List of currencies (str) in User's portfolio other than crypto and default currency
        """
        stock_info = self.get_stock_info()
        stock_info['Raw'], stock_info['Type'] = zip(
            *stock_info['Ticker'].apply(data.split_ticker))
        stock_info = stock_info[stock_info['Raw'] != stock_info['Currency']]
        fx = stock_info['Currency'].unique()
        mask = (fx != 'NA') & (fx != self.default_currency)
        currencies = [
            f'{currency}{self.default_currency}=X.FX' for currency in list(fx[mask])]
        return currencies

    def update_last_accessed(self, date):
        self.last_accessed = date
        db.session.commit()

    def get_stock_info(self):
        tickers = self.get_tickers()
        df = pd.read_sql(Stocks.query.filter(Stocks.ticker.in_(
            tickers)).statement, db.engine).rename(str.capitalize, axis=1)
        return df

    def get_tickers(self):
        return [t[0] for t in db.session.query(Trades.ticker).filter(Trades.user_id == self.id).distinct().all()]

    def _latest_price_timestamp(self) -> Optional[datetime]:
        """Return the most common quoted date across current holdings (ex FX)."""
        tickers = [
            t[0] for t in db.session.query(Trades.ticker)
            .filter(Trades.user_id == self.id)
            .distinct()
            .all()
        ]

        filtered_tickers = [
            ticker for ticker in tickers
            if data.split_ticker(ticker)[1] != "FX"
        ]

        if not filtered_tickers:
            return None

        rows = db.session.query(
            StockPrices.ticker,
            func.max(StockPrices.date).label("max_date")
        ).filter(
            StockPrices.ticker.in_(filtered_tickers)
        ).group_by(
            StockPrices.ticker
        ).all()

        if not rows:
            return None

        date_list = []
        for _, max_date in rows:
            if max_date is None:
                continue
            if isinstance(max_date, datetime):
                date_list.append(max_date.date())
            else:
                date_list.append(max_date)

        if not date_list:
            return None

        counts = Counter(date_list)
        most_common_date, _ = max(counts.items(), key=lambda x: (x[1], x[0]))
        return datetime.combine(most_common_date, datetime.min.time())

    def info_date(self, start_date: datetime = None, as_at_date: datetime = None, hide_zero_pos: bool = False, limit_divs_by_date: bool = False) -> Union[pd.DataFrame, pd.DataFrame]:
        """
        Returns portfolio dataframe as at a specified date (or as at today if no date provided). Relies on stock price data in StockPrices and does not perform an update

        Args:
            start_date(datetime, optional): Datetime for the starting date of trades within portfolio. Defaults to None.
            as_at_date(datetime, optional): Datetime for the last trade date of trades within portfolio. Defaults to None.
            hide_zero_pos(bool, optional): Hide nil stock positions. Defaults to False.
            limit_divs_by_date(bool, optional): If true, limits dividends to only those between start and end dates, otherwise shows dividends for period. Defaults to False.

        Returns:
            Dataframe: Portfolio information as at specified date containing following information for each stock held in portfolio
            ['Ticker', 'Name', 'Quantity', 'LastPrice', '%LastChange', '$LastChange', 'CurrVal', 'IRR', '%UnRlGain', '%PF',
            'AvgCost', 'Cost', '%CostPF', 'Dividends', 'RlGain', 'UnRlGain', 'TotalGain', 'Date']
            Dataframe: Portfolio trade information up to specified date containing following information for each stock held in portfolio
            ['Ticker', 'Currency', 'CumQuan', 'Fx', 'AvgCostRaw', 'grouping', 'AvgCostAdj', 'AvgCost', 'CumDiv', 'TotalRlGain']
        """
        # Set up variables
        start = datetime.now()
        tickers = self.get_tickers()
        tickers.extend(self.currencies())  # add currencies to ticker list
        logger.info(f'Get tickers took {(datetime.now()-start)} to run')

        start = datetime.now()
        if start_date is None:
            start_date = db.session.query(
                func.min(StockPrices.date)).first()[0]
        if as_at_date is None:
            as_at_date = pd.to_datetime('today')
        logger.info(
            f'Get start and as at date took {(datetime.now()-start)} to run')

        logger.debug(
            'Get splits and dividend information from stockprices for tickers')
        start = datetime.now()
        splits = StockPrices.query.filter(
            StockPrices.ticker.in_(tickers),
            StockPrices.splits != 0).order_by(StockPrices.date.asc()).all()
        if limit_divs_by_date:
            divs = StockPrices.query.filter(
                StockPrices.ticker.in_(tickers),
                StockPrices.dividends != 0,
                StockPrices.date.between(start_date, as_at_date)).order_by(StockPrices.date.asc()).all()
        else:
            divs = StockPrices.query.filter(
                StockPrices.ticker.in_(tickers),
                StockPrices.dividends != 0).order_by(StockPrices.date.asc()).all()
        logger.debug(divs)
        logger.info(
            f'Splits and divs data took {(datetime.now()-start)} to run')

        start = datetime.now()
        logger.debug('Getting latest prices for tickers')
        curr_df = self.current_prices(
            tickers=tickers, as_at_date=as_at_date, last_change=True)
        logger.info(
            f'Current prices took {(datetime.now()-start)} to run')

        start = datetime.now()
        logger.debug(
            'Getting historical positions and calculating current holdings')
        hist_df = self.hist_positions(start_date=start_date, as_at_date=as_at_date,
                                      splits=splits, divs=divs, include_dividends=True, calculate_gains=True, limit_divs_by_date=limit_divs_by_date)
        hist_trades = hist_df.copy(deep=True)
        logger.info(
            f'Historical positions took {(datetime.now()-start)} to run')

        logger.debug('Getting IRR for each position')
        start = datetime.now()
        irr_df = self.calc_IRR(hist_df[['Date', 'Ticker', 'CF', 'CumQuan']].copy(
        ), curr_df[['Date', 'Ticker', 'Close']].copy())
        logger.info(f'IRR data took {(datetime.now()-start)} to run')

        # clean-up dataframe
        logger.debug('Cleaning up dataframe')
        start = datetime.now()
        hist_df.drop(['Date', 'Quantity', 'Price', 'Fees', 'Direction', 'AdjQuan', 'CFBuy',
                     'CumCost', 'QBuy', 'CumBuyQuan', 'RlGain', 'CF', 'Dividends'], axis=1, inplace=True)
        hist_df = hist_df.groupby('Ticker').last().reset_index()
        hist_df.rename(columns={
                       'CumQuan': 'Quantity', 'TotalRlGain': 'RlGain', 'CumDiv': 'Dividends'}, inplace=True)

        # drop rows where quantity is zero if argument passed is true
        if hide_zero_pos:
            hist_df = hist_df[hist_df['Quantity'].round(2) != 0]

        # Calculate total cost of each stock in portfolio
        hist_df['Cost'] = hist_df.Quantity * hist_df.AvgCost

        # merge hist_df and curr_df. Drop duplicate currency and fx columns and rename current fx column
        logger.debug(
            'Merging histoical and current dataframes, adding total row and irr information')
        info_df = hist_df.merge(curr_df, on='Ticker', how='left').drop(
            columns=['Fx_x', 'Currency_x', 'Currency_y']).rename(columns={'Fx_y': 'Fx'})
        info_df.sort_values('Ticker', inplace=True)

        # Add total row to info_df
        info_df = self._add_total_row(
            info_df, 'Ticker', ['RlGain', 'Cost', 'Dividends'])
        info_df['Date'] = pd.to_datetime(info_df['Date'].fillna(pd.NaT))

        # merge with irr_df
        info_df = info_df.merge(irr_df, on='Ticker')
        info_df = info_df.reset_index(drop=True)

        logger.debug('Perform calculations on info dataframe and return')
        tot_index = len(info_df.index) - 1

        # create relevant columns including % of portfolio, current value, last change, unrealised gains, total gains
        info_df.rename(columns={'Close': 'LastPrice'}, inplace=True)
        info_df['%CostPF'] = info_df['Cost'] / info_df['Cost'][:-1].sum()
        info_df['CurrVal'] = info_df['Quantity'] * \
            info_df['LastPrice'] * info_df['Fx']
        info_df.at[tot_index, 'CurrVal'] = info_df['CurrVal'].sum()
        info_df['$LastChange'] = info_df['CurrVal'] * \
            (1 - 1 / (1 + info_df['%LastChange'])) * info_df['Fx']
        info_df.at[tot_index, '$LastChange'] = info_df['$LastChange'].sum()
        info_df['%PF'] = info_df['CurrVal'] / info_df['CurrVal'][:-1].sum()
        info_df['UnRlGain'] = info_df['CurrVal'] + info_df['Cost']
        info_df['UnRlGain'].fillna(0, inplace=True)
        info_df['TotalGain'] = info_df['UnRlGain'] + \
            info_df['RlGain'] + info_df['Dividends']
        info_df['%UnRlGain'] = info_df['UnRlGain'] / -info_df['Cost']

        # get type of stock from ticker. Add names / date last accessed etc to info_df
        info_df = pd.merge(info_df, self.get_stock_info(),
                           on='Ticker', sort=False, how='left')
        info_df['Raw'], info_df['Type'] = zip(
            *info_df['Ticker'].apply(data.split_ticker))
        info_df['Type'] = pd.Categorical(info_df['Type'], TYPE_CATEGORIES)

        # set up column in order of INFO_COLUMNS
        info_df = info_df[INFO_COLUMNS]
        logger.info(f'Clean up data took {(datetime.now()-start)} to run')
        return info_df, hist_trades

    def hist_positions(self, start_date: datetime, as_at_date: datetime, splits: List, divs: List, tickers: List = None, include_dividends: bool = True, calculate_gains: bool = True, limit_divs_by_date: bool = False) -> pd.DataFrame:
        """
        Calculate historical positions for all stocks in trades for user as at given date

        Args:
            start_date(datetime): Date as at which to calculate limit capital gains and dividends calculations(i.e. will return capital gains and dividends between start_date and as_at_date)
            as_at_date(datetime): Date as at which to calculate the position of portfolio
            splits(List): List containing split items from StockPrices for stocks in portfolio
            divs(List): List containing dividend items from StockPricesfor stocks in portfolio
            tickers(List): List of tickers for which to return historic positions
            include_dividends(bool, optional): If True, include dividends otherwise exclude them
            limit_divs_by_date(bool, optional): If true, limits dividends to only those between start and end dates, otherwise shows dividends for period. Defaults to False.

        Returns:
            pd.DataFrame: Dataframe containing following information for each stock held in portfolio

        ['Date', 'Ticker', 'Quantity', 'Price', 'Fees', 'Direction', 'CF', 'AdjQuan', 'CumQuan', 'CFBuy', 'CumCost'
                    'QBuy', 'QBuyQuan', 'AvgCost', 'RlGain', 'Dividends', 'CumDiv', 'TotalRlGain']
        """

        # Get historical positions based on dates provided and filter by tickers
        hist_pos_statement = db.session.query(Trades, Stocks.currency).where(
            Trades.date <= as_at_date, Trades.user_id == self.id, Trades.ticker == Stocks.ticker).statement
        hist_pos = pd.read_sql(hist_pos_statement, db.engine).drop(
            columns=['user_id', 'pf_price', 'pf_shares']).rename(str.capitalize, axis=1)
        if tickers is not None:
            hist_pos = hist_pos[hist_pos['Ticker'].isin(tickers)].copy()
        
        # Sort by Date, Ticker AND Id to ensure deterministic order (critical for merge_asof)
        hist_pos.sort_values(['Date', 'Ticker', 'Id'], inplace=True)

        logger.debug(hist_pos[hist_pos['Direction'] == 'Div'])

        # Adjust hist_pos for splits and get quantities at each point along with cumulative quantities
        hist_pos = self.adjust_for_splits(hist_pos, splits)
        hist_pos['AdjQuan'] = np.where(hist_pos.Direction == 'Sell', -1, np.where(
            hist_pos.Direction == 'Div', 0, 1)) * hist_pos.Quantity
        hist_pos['CumQuan'] = hist_pos.groupby('Ticker')['AdjQuan'].cumsum()

        if include_dividends:
            hist_pos = self.add_dividends(hist_pos, divs)  # Add dividends
            if limit_divs_by_date:
                hist_pos = hist_pos[~((hist_pos['Direction'] == 'Div') & ~(
                    hist_pos['Date'].between(start_date, as_at_date)))]
        else:
            # remove dividends from transaction data so as to exclude dividends
            hist_pos = hist_pos[~(hist_pos['Direction'] == 'Div')]

        hist_pos = self.get_fx(hist_pos)  # Add FX
        if calculate_gains:
            hist_pos = self.calculate_gains(
                hist_pos, start_date)  # Calculate gains and losses
        return hist_pos

    def adjust_for_splits(self, hist_pos: pd.DataFrame, splits: List) -> pd.DataFrame:
        """
        Adjusts hist_pos for splits based on splits list

        Args:
            hist_pos (pd.DataFrame): dataframe with historical positions of each trade / stock
            splits (List): List of splits with date and split

        Returns:
            pd.DataFrame: hist_post dataframe with quantity and price updated for splits
        """
        for split in splits:
            hist_pos['Quantity'] = np.where(
                (hist_pos['Date'] <= split.date) & (
                    hist_pos['Ticker'] == split.ticker),
                round(hist_pos['Quantity'] * float(split.splits), 0),
                hist_pos['Quantity'])
            hist_pos['Price'] = np.where(
                (hist_pos['Date'] <= split.date) & (
                    hist_pos['Ticker'] == split.ticker),
                hist_pos['Price'] / float(split.splits),
                hist_pos['Price'])
        return hist_pos

    def add_dividends(self, hist_pos: pd.DataFrame, divs: List) -> pd.DataFrame:
        """
        Adjusts hist_pos for dividends based on dividends list

        Args:
            hist_pos (pd.DataFrame): dataframe with historical positions of each trade / stock
            divs (List): List of dividends with date and quantity of dividends

        Returns:
            pd.DataFrame: hist_post dataframe updated for dividend data
        """
        for dividend in divs:
            dt_div = hist_pos[(hist_pos['Date'] <= dividend.date) & (
                hist_pos['Ticker'] == dividend.ticker)]['Date'].index
            if not dt_div.empty:
                div_qty = hist_pos.at[dt_div[-1], 'CumQuan']
                if div_qty != 0:
                    div_data = {
                        'Date': dividend.date,
                        'Ticker': dividend.ticker,
                        'Quantity': div_qty,
                        'Price': float(dividend.dividends),
                        'Fees': float(0),
                        'Direction': 'Div',
                        'Currency': hist_pos.at[dt_div[0], 'Currency'],
                        'AdjQuan': float(0),
                        'CumQuan': div_qty}

                    div_data_df = pd.DataFrame(div_data, index=[0])
                    div_data_df['Date'] = div_data_df['Date'].astype(
                        'datetime64[ns]')
                    hist_pos = pd.concat(
                        [hist_pos, div_data_df], ignore_index=True)
                    hist_pos.sort_values(['Ticker', 'Date'], inplace=True)
        return hist_pos

    def calculate_gains(self, hist_pos: pd.DataFrame, start_date: datetime) -> pd.DataFrame:
        """
        Calculate realised and unrealised capital gains from  start_date

        Args:
            hist_pos (pd.DataFrame): dataframe with historical positions of each trade / stock
            start_date (datetime): Date as at which to calculate limit capital gains and dividends calculations(i.e. will return capital gains and dividends between start_date and as_at_date)

        Returns:
            pd.DataFrame: Updated hist_pos dataframe including capital gains information
        """
        if start_date is None:
            start_date = hist_pos['Date'].min()
        hist_pos['CF'] = np.where(hist_pos.Direction == 'Buy', -1, 1) * (
            hist_pos.Quantity * hist_pos.Price * hist_pos.Fx) - (hist_pos.Fees * hist_pos.Fx)
        hist_pos['CFBuy'] = np.where(
            hist_pos.Direction == 'Buy', hist_pos.CF, 0)
        hist_pos['CumCost'] = hist_pos.groupby(
            'Ticker', group_keys=False)['CFBuy'].cumsum()
        hist_pos['QBuy'] = np.where(
            hist_pos.Direction == 'Buy', hist_pos.Quantity, 0)
        hist_pos['CumBuyQuan'] = hist_pos.groupby('Ticker')['QBuy'].cumsum()

        hist_pos['AvgCostRaw'] = hist_pos['CumCost'] / hist_pos['CumBuyQuan']
        hist_pos_grouped = hist_pos.groupby('Ticker', group_keys=False)
        hist_pos = hist_pos_grouped.apply(self.calc_avg_price)
        hist_pos.reset_index(drop=True, inplace=True)
        hist_pos['AvgCost'] = np.where(
            hist_pos['grouping'] == 0, hist_pos['AvgCostRaw'], hist_pos['AvgCostAdj'])

        hist_pos['RlGain'] = np.where(((hist_pos.Direction == 'Sell') & (
            hist_pos.Date >= start_date)), hist_pos.CF + (hist_pos.AvgCost * hist_pos.Quantity), 0)
        hist_pos['Dividends'] = np.where(
            hist_pos.Direction == 'Div', hist_pos.CF, 0)
        hist_pos['CumDiv'] = hist_pos.groupby('Ticker')['Dividends'].cumsum()
        hist_pos['TotalRlGain'] = hist_pos.groupby('Ticker')['RlGain'].cumsum()

        return hist_pos

    def get_fx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns dataframe with Fx column

        Args:
            df (pd.DataFrame): dataframe with Currency, Ticker and Date columns

        Returns:
            pd.DataFrame: returns df with a Fx column

        """
        df['Fx'] = np.where(
            df['Currency'] == self.default_currency, float(1), np.nan)

        # Avoid iterrows() to prevent RangeIndex issues
        df = df.reset_index(drop=True)  # Ensure clean integer index

        for i in range(len(df)):
            row = df.iloc[i]
            type = data.split_ticker(row['Ticker'])[1]
            if np.isnan(row['Fx']) and type not in ['CASH', 'LOAN']:
                fx_rate = db.session.query(StockPrices.close).filter(
                    StockPrices.ticker == f'{row["Currency"]}{self.default_currency}=X.FX').filter(StockPrices.date == row['Date']).scalar() or float(1)
                df.iloc[i, df.columns.get_loc('Fx')] = float(fx_rate)
            elif np.isnan(row['Fx']) and type in ['CASH', 'LOAN']:
                df.iloc[i, df.columns.get_loc('Fx')] = float(1)
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

        # get current position for each ticker (i.e. current number of shares held)
        curr_pos = hist_pos.groupby('Ticker').last().reset_index()
        curr_p = curr_p.set_index('Ticker').astype({'Close': 'float'})

        # add current value in new column in curr_pos dataframe
        for _, row in curr_pos.iterrows():
            ticker = row['Ticker']
            try:
                curr_p.at[ticker, 'CF'] = row['CumQuan'] * \
                    curr_p.loc[ticker, 'Close']
            except KeyError:
                logger.debug(f'IRR Calculation: No stock data for {ticker}')
                curr_p.at[ticker, 'CF'] = np.nan

        # clean up dataframes and reset indices before merge
        hist_pos.drop(['CumQuan'], axis=1, inplace=True)
        curr_p.drop(['Close'], axis=1, inplace=True)
        curr_p.reset_index(inplace=True)

        # merge curr_p into hist_pos as transactions
        CF_df = pd.concat([hist_pos, curr_p], ignore_index=True)
        CF_df.sort_values(['Date'], inplace=True)
        CF_df.reset_index(inplace=True, drop=True)

        # extract CFs and dates by ticker and pass through IRR function, store in dataframe
        grouped_CF_df = CF_df.groupby('Ticker')[['Date', 'CF']]
        irr_rows = []

        for name, group in grouped_CF_df:
            stock_irr = irr.irr(group.values.tolist())
            irr_rows.append((name, stock_irr))

        CF_df.drop('Ticker', axis=1, inplace=True)
        CF_df.dropna(inplace=True)
        total_irr = irr.irr(CF_df.values.tolist()) if not CF_df.empty else np.nan
        irr_rows.append(('Total', total_irr))

        IRR_df = pd.DataFrame(irr_rows, columns=['Ticker', 'IRR'])

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

        curr_p = self._current_prices(
            tickers, [as_at_date] * len(tickers), ['Ticker', 'Close', 'Date', 'Currency'])

        if last_change:
            logger.debug("Getting previous day price")
            tickers = curr_p['Ticker'].to_list()
            prev_dates = [(d - timedelta(days=1)) for d in curr_p['Date']]
            prev_df = self._current_prices(tickers, prev_dates, ['Ticker', 'Close']).rename(
                columns={'Close': 'PrevClose'})
            curr_p = curr_p.merge(prev_df, on='Ticker', how='left')
            curr_p['%LastChange'] = curr_p['Close'] / curr_p['PrevClose'] - 1
            curr_p.drop(columns='PrevClose', inplace=True)

        curr_p = self.get_fx(curr_p)
        return curr_p

    def _current_prices(self, tickers: List[str], as_at_dates: List[datetime], columns: List[str]) -> pd.DataFrame:
        """
        Internal method. Gets the latest prices for tickers as at the given date. If last_change is true, also returns the change in price from the previous price for a given ticker

        Args:
            tickers (List): List of tickers for which to get current prices
            as_at_date (datetime): Date at which to generate prices
            columns (List): list of columns for returned dataframe

        Returns:
            pd.DataFrame: Dataframe with required columns with current prices in database
        """
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
        col_types = {key: value for key,
                     value in COL_TYPES.items() if key in columns}

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

        # Convert the result set to a list of dictionaries
        latest_prices_dicts = [
            {
                'ticker': p[0].ticker,
                'date': p[0].date,
                'open': p[0].open,
                'high': p[0].high,
                'low': p[0].low,
                'close': p[0].close,
                'volume': p[0].volume,
                'adjclose': p[0].adjclose,
                'dividends': p[0].dividends,
                'splits': p[0].splits,
                'currency': p[1]
            }
            for p in latest_prices
        ]

        # Create a pandas DataFrame from the list of dictionaries
        df = pd.DataFrame(latest_prices_dicts)
        if df.empty:
            return pd.DataFrame(columns=columns)
        df = df.rename(
            str.capitalize, axis=1).astype(col_types)[columns]
        return df

    @staticmethod
    def calc_avg_price(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate average cost price for each group of shares bought / sold

        Args:
            df (pd.DataFrame): dataframe with trades

        Returns:
            pd.DataFrame: dataframe which includes the average price calculated
        """
        # create group for each group of shares bought / sold
        # Ensure CumQuan is numeric before cumsum (handles NaN/object dtype)
        cum_quan = pd.to_numeric(df['CumQuan'], errors='coerce').fillna(0)
        df['grouping'] = cum_quan.eq(0).shift(1, fill_value=False).cumsum()
        df['grouping'] = pd.to_numeric(df['grouping'], errors='coerce').fillna(0)
        avg_price_df = df.groupby('grouping', as_index=False).apply(
            lambda x: x.CFBuy.sum() / x.QBuy.sum()).reset_index(drop=True)
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
        df = df.copy()
        totals = df.loc[:, list_cols].sum(axis=0)
        df.loc['Total', list_cols] = totals
        df.at['Total', index] = 'Total'
        return df

    def price_history(self, ticker: str, start_date: datetime, as_at_date: datetime, period: str) -> Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Provides price history for a given ticker between start date and end date and resampled based on period

        Args:
            ticker (str): Ticker for stock data to return
            start_date (datetime): start date for stock data
            as_at_date (datetime): end date for stock data
            period (str): A, M or D representing annual, monthly or daily price data

        Raises:
            ValueError: If period is not entered correctly, raises value error

        Returns:
            Union[pd.DataFrame, pd.DataFrame, pd.DataFrame]: price history dataframe, dividend data dataframe and splts dataframe
        """
        logger.info(f'********************{ticker}*************************')
        prices_df = pd.read_sql(StockPrices.query.filter(
            StockPrices.ticker == ticker).statement, db.engine).rename(str.capitalize, axis=1)

        splits = StockPrices.query.filter(
            StockPrices.ticker == ticker, StockPrices.splits != 0).order_by(StockPrices.date.asc()).all()
        divs = StockPrices.query.filter(
            StockPrices.ticker == ticker, StockPrices.dividends != 0).order_by(StockPrices.date.asc()).all()

        hist_df = self.hist_positions(start_date=start_date, as_at_date=as_at_date, splits=splits, divs=divs, tickers=[
                                      ticker], include_dividends=True, calculate_gains=True)
        prices_df = prices_df[prices_df['Date'] >= hist_df['Date'].min()]
        # force date column to be datetime objects
        prices_df['Date'] = pd.to_datetime(prices_df['Date'])
        if period == 'A':
            p_hist_df = prices_df.groupby(
                prices_df['Date'].dt.year).tail(1).copy()
        elif period == 'M':
            p_hist_df = prices_df.groupby(
                [prices_df['Date'].dt.year, prices_df['Date'].dt.month]).tail(1).copy()
        elif period == 'D':
            p_hist_df = prices_df.copy()
        else:
            raise ValueError(
                'Please insert either A (annual), M (monthly) or D (daily) for period')

        p_hist_df['Quantity'] = None
        p_hist_df['AvgCost'] = None
        p_hist_df['Dividends'] = None
        p_hist_df['RlGain'] = None
        p_hist_df['CumDiv'] = None

        for idx, row in p_hist_df.iterrows():
            try:
                pos_at_date = hist_df[hist_df['Date'] <= row['Date']].iloc[-1]
                p_hist_df.loc[idx, 'Quantity'] = pos_at_date['CumQuan']
                p_hist_df.loc[idx, 'AvgCost'] = pos_at_date['AvgCost']
                p_hist_df.loc[idx, 'Dividends'] = pos_at_date['Dividends'] if pos_at_date['Date'] == row['Date'] else 0
                p_hist_df.loc[idx, 'CumDiv'] = pos_at_date['CumDiv']
                p_hist_df.loc[idx, 'RlGain'] = pos_at_date['TotalRlGain']
            except IndexError:
                logger.info(
                    f'No price data for {ticker} prior to {row["Date"]}')

        p_hist_df['CurrVal'] = p_hist_df['Close'] * p_hist_df['Quantity']
        p_hist_df['Cost'] = p_hist_df['AvgCost'] * p_hist_df['Quantity']
        p_hist_df['UnRlGain'] = p_hist_df['CurrVal'] + p_hist_df['Cost']
        p_hist_df['TotalGain'] = p_hist_df['UnRlGain'] + \
            p_hist_df['RlGain'] + p_hist_df['Dividends']

        div_df = p_hist_df[['Ticker', 'Date', 'Dividends']]
        div_df = div_df[div_df['Dividends'] != 0.0].dropna(subset='Dividends')
        split_df = prices_df[['Ticker', 'Date', 'Splits']]
        split_df = split_df[split_df['Splits'] != 0.0].dropna(subset='Splits')
        p_hist_df = p_hist_df.drop(columns=['Dividends'])

        return p_hist_df, div_df, split_df

    def update_prices(self, as_at_date: datetime, tickers: List = None, min_days: int = -1) -> pd.DataFrame:
        """
        Update prices for tickers in the database up to as_at_date with lookback for min_days. This function will also update the prices database

        Args:
            as_at_date (datetime): Date as at which to obtain prices for tickers
            tickers (List, optional): List of tickers for which to update prices. Defaults to None. If None is passed, get unique ticker list from user's trades and obtain prices for those tickers
            min_days (int, optional): Buffer of days to look back past previous trade data. -1 is suggested to ensure that close prices are accurate and mid-day prices are not being used. Defaults to -1.

        Returns:
            pd.DataFrame: dataframe with updated prices for all tickers for the given time period
        """
        start = datetime.now()
        if tickers is None:
            tickers = list(self.get_trades()['Ticker'].unique())
            tickers.extend(self.currencies())

        pf_min_date = self.get_trades()['Date'].min()
        # for tickers already in stockprices, work out update period
        dates_statement = db.session.query(StockPrices.ticker, func.max(StockPrices.date).label('max_date')) \
            .filter(StockPrices.ticker.in_(tickers)) \
            .group_by(StockPrices.ticker).statement
        df = pd.read_sql(dates_statement, db.engine)
        if df.empty:
            logger.info('No existing price records to update')
        else:
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
                        first()[0] or pf_min_date
                else:
                    min_date = pf_min_date
            else:
                min_date = pf_min_date if self.get_ticker_trades(ticker)['date'].min(
                ) is None else self.get_ticker_trades(ticker)['date'].min()
            df = pd.concat([df, pd.DataFrame(
                {'ticker': ticker, 'start_date': min_date, 'end_date': as_at_date}, index=[0])], ignore_index=True)

        if df.empty:
            logger.info('No tickers require price updates')
            return pd.DataFrame()

        # get price data for all tickers as needed and reset index. Replace NaN with None for insertion into SQL
        prices = data.get_price_data(df['ticker'], df['start_date'], df['end_date'], [
                                     self.default_currency] * len(df['ticker']))
        if prices.empty:
            logger.info('Price service returned no data; skipping update')
            return pd.DataFrame()

        prices = prices.reset_index()
        prices = prices.replace(np.nan, None)

        # iterate through rows in prices to update SQL database with updated prices where already existing or to append new data
        price_data_list = []
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
            }

            # Add 'dividends' and 'splits' only if they exist
            if 'Dividends' in prices.columns:
                price_data['dividends'] = row['Dividends']
            if 'Splits' in prices.columns:
                price_data['splits'] = row['Splits']
            price_data_list.append(price_data)

        # Bulk insert/update price data
        stmt = insert(StockPrices).values(price_data_list)
        update_cols = {c.name: c for c in StockPrices.__table__.columns if c.name not in {
            'ticker', 'date'}}
        on_duplicate_key_stmt = stmt.on_duplicate_key_update(
            {key: getattr(stmt.inserted, key) for key in update_cols})
        db.session.execute(on_duplicate_key_stmt)

        update_time = datetime.utcnow()
        tickers_to_update = list(set(df['ticker']))
        if tickers_to_update:
            Stocks.query.filter(Stocks.ticker.in_(tickers_to_update)).update(
                {'last_updated': update_time},
                synchronize_session=False
            )
        db.session.commit()
        earliest_price_date = pd.to_datetime(prices['Date']).min()
        if pd.notna(earliest_price_date):
            self.mark_monthly_nav_cache_stale(earliest_price_date.to_pydatetime())

        logger.info(f'price update took {(datetime.now()-start)} to run')
        return prices

    def monthly_summary(self, start_date: datetime, end_date: datetime, tickers: List = None, exclude_crypto: bool = False, exclude_loans: bool = False, detail: bool = False) -> pd.DataFrame:
        """
        Creates a dataframe summarising portfolio activity showing opening balance, net investment, investment returns, closing balance, and each investment's month-end valuation. Includes row for dividends and, if:
            - exclude_crypto is True, line for crypto balance
            - exclude_loans is True, line for margin loan balance
        Always will create summary as at the end of the month for the periods provided

        Args:
            start_date (datetime): Start date to calculate monthly summary
            end_date (datetime): End date to calculate monthly summary
            tickers (List, optional): Limits to certain tickers if required. Defaults to None.
            exclude_crypto (bool, optional): Excludes any assets with .CRYPTO. Defaults to False.
            exclude_loans (bool, optional): Excludes any assets with .LOAN. Defaults to False.
            detail (bool, optional): If True, return per-investment balances instead of summary rows. Defaults to False.

        Returns:
            pd.DataFrame: Summary rows when detail is False (opening balance, net investment, investment returns, closing balance, dividends and optional crypto/loan rows) or per-investment monthly changes when detail is True.
        """
        start_date = (start_date.replace(day=1) - pd.Timedelta(days=1)
                      )  # add previous period end for closing balance
        month_ends = pd.date_range(start_date, end_date, freq='ME').tolist()
        formatted_columns = [d.strftime('%Y-%m-%d') for d in month_ends]

        df = pd.DataFrame()
        investments_df = pd.DataFrame()
        trades_df = None

        for m_end in month_ends:
            logger.debug(m_end)
            info_date, _ = self.info_date(start_date=start_date,
                                          as_at_date=m_end, hide_zero_pos=True, limit_divs_by_date=True)
            info_date = info_date[:-1]
            if tickers is not None:
                info_date = info_date[info_date['Ticker'].isin(tickers)]
            info_date['Date'] = m_end
            # print(info_date)
            df = pd.concat([df, info_date], ignore_index=True)
            if m_end == month_ends[-1]:
                trades_df = _

        # reshape the dataframe to have dates running along columns and tickers as rows
        reshaped_df = df.pivot(
            index='Ticker', columns='Date', values='CurrVal')

        # Separate crypto and margin loan balances
        crypto_df = reshaped_df[reshaped_df.index.str.endswith('.CRYPTO')]
        crypto_df.loc['Crypto'] = crypto_df.sum(numeric_only=True)
        crypto_df = crypto_df[crypto_df.index == 'Crypto']
        crypto_df.columns = pd.to_datetime(
            crypto_df.columns).strftime('%Y-%m-%d')
        loans_df = reshaped_df[reshaped_df.index.str.endswith('.LOAN')]
        loans_df.loc['Loans'] = loans_df.sum(numeric_only=True)
        loans_df = loans_df[loans_df.index == 'Loans']
        loans_df.columns = pd.to_datetime(
            loans_df.columns).strftime('%Y-%m-%d')

        # Exclude crypto and margin loan balances based on flag
        if exclude_crypto:
            reshaped_df = reshaped_df[~reshaped_df.index.str.endswith(
                '.CRYPTO')]
        if exclude_loans:
            reshaped_df = reshaped_df[~reshaped_df.index.str.endswith('.LOAN')]
        reshaped_df.loc['Total'] = reshaped_df.sum(numeric_only=True)
        investments_df = reshaped_df.drop(index='Total').copy()
        reshaped_df = reshaped_df.reset_index()

        # Calculate the closing balance
        monthly_summ = reshaped_df.loc[reshaped_df['Ticker'] == 'Total']
        monthly_summ.index = ['Closing Balance']
        monthly_summ.drop(columns=['Ticker'], inplace=True)

        # Calculate the opening balance as the previous month's closing balance
        opening_balance = monthly_summ.shift(1, axis=1).iloc[-1]
        opening_balance.name = 'Opening Balance'

        # Insert the opening balance as the first row in the DataFrame
        monthly_summ = pd.concat([opening_balance.to_frame().T, monthly_summ])
        # Convert the numpy array to a pandas Index and then format the dates
        monthly_summ.columns = pd.to_datetime(
            monthly_summ.columns).strftime('%Y-%m-%d')

        # Filter the trades DataFrame to include only rows within the specified date range, then group by month and sum the 'CF' values for each month.
        # Create a new DataFrame with these monthly sums, set the index to 'Capital Flow'
        trades_df = trades_df[trades_df['Date'] >= start_date]
        trades_df = trades_df[trades_df['Date'] <= end_date]
        # separate out dividends and group by month, then aggregate into df
        div_df = trades_df[trades_df['Direction'] == 'Div']
        div_by_month = div_df.groupby(
            div_df['Date'].dt.to_period('M'))['CF'].sum()
        div_df = pd.DataFrame([div_by_month.values],
                              columns=div_by_month.index.strftime('%Y-%m-%d'))
        div_df.index = ['Dividends']

        trades_df = trades_df[trades_df['Direction'] != 'Div']

        # Exclude crypto and margin loan trades based on flags
        if exclude_crypto:
            trades_df = trades_df[~trades_df['Ticker'].str.endswith('.CRYPTO')]
        if exclude_loans:
            trades_df = trades_df[~trades_df['Ticker'].str.endswith('.LOAN')]

        trades_df['CF'] = trades_df['CF'] * - \
            1  # change CF flow for this context
        total_CF_by_month = trades_df.groupby(
            trades_df['Date'].dt.to_period('M'))['CF'].sum()
        total_CF_row = pd.DataFrame(
            [total_CF_by_month.values], columns=total_CF_by_month.index.strftime('%Y-%m-%d'))
        total_CF_row.index = ['Net Investment']

        # Merge the monthly_summ DataFrame with the total_CF_row DataFrame, using the 'Date' columns as the key
        monthly_summ = pd.concat(
            [monthly_summ, total_CF_row, div_df], axis=0)
        if exclude_crypto:
            monthly_summ = pd.concat([monthly_summ, crypto_df], axis=0)
        if exclude_loans:
            monthly_summ = pd.concat([monthly_summ, loans_df], axis=0)
        monthly_summ.fillna(0, inplace=True)

        # Create a new row in monthly_summ which takes Closing Balance less opening balance less capital flow as investment returns
        investment_returns = monthly_summ.loc['Closing Balance'] - \
            monthly_summ.loc['Opening Balance'] - \
            monthly_summ.loc['Net Investment']
        investment_returns.name = 'Investment Returns'
        monthly_summ = pd.concat(
            [monthly_summ, investment_returns.to_frame().T], axis=0)

        # Reorder the rows in monthly_summ
        summary_order = ['Opening Balance', 'Net Investment',
                         'Investment Returns', 'Closing Balance', 'Dividends']
        if exclude_loans:
            summary_order.append('Loans')
        if exclude_crypto:
            summary_order.append('Crypto')
        monthly_summ = monthly_summ.loc[summary_order]
        monthly_summ = monthly_summ.reindex(
            columns=formatted_columns, fill_value=0)

        if not investments_df.empty:
            investments_df = investments_df.fillna(0)
            investments_df.columns = pd.to_datetime(
                investments_df.columns).strftime('%Y-%m-%d')
            investments_df = investments_df.reindex(
                columns=formatted_columns, fill_value=0)
            investments_df.index.name = None
        else:
            investments_df = pd.DataFrame(columns=formatted_columns)

        drop_col = formatted_columns[0] if formatted_columns else None

        monthly_summ.fillna(0, inplace=True)
        investments_df.fillna(0, inplace=True)

        if detail:
            value_df = investments_df.copy()

            contrib_df = pd.DataFrame(
                0, index=value_df.index, columns=formatted_columns)
            if 'trades_df' in locals() and trades_df is not None and not trades_df.empty:
                ticker_cf = trades_df.copy()
                ticker_cf['Month'] = ticker_cf['Date'].dt.to_period(
                    'M').dt.to_timestamp('M')
                ticker_cf = ticker_cf.groupby(
                    ['Ticker', 'Month'])['CF'].sum().unstack(fill_value=0)
                ticker_cf.columns = ticker_cf.columns.strftime('%Y-%m-%d')
                contrib_df = ticker_cf.reindex(
                    index=value_df.index, columns=formatted_columns, fill_value=0)

            change_df = value_df.diff(axis=1).fillna(0)
            if drop_col:
                change_df.drop(
                    columns=[drop_col], inplace=True, errors='ignore')
                contrib_df.drop(
                    columns=[drop_col], inplace=True, errors='ignore')
                value_df.drop(
                    columns=[drop_col], inplace=True, errors='ignore')

            detail_df = (change_df - contrib_df).fillna(0)

            if not value_df.empty and not detail_df.empty:
                final_col = value_df.columns[-1]
                open_mask = ~np.isclose(value_df[final_col], 0)
                detail_df = detail_df.loc[open_mask]

            if not detail_df.empty:
                detail_df.loc['Total'] = detail_df.sum(numeric_only=True)
            return detail_df

        if drop_col:
            monthly_summ.drop(columns=[drop_col], inplace=True, errors='ignore')

        monthly_summ.fillna(0, inplace=True)
        return monthly_summ

    def _portfolio_value_on(self, as_at_date: datetime) -> Optional[float]:
        """
        Retrieve the total portfolio value for the user as at a given date using
        positions and prices already stored in the database.

        This intentionally avoids ``info_date()``: month-end NAV needs neither
        performance calculations nor IRR, and normal NAV views must not trigger
        a market-data refresh.

        Returns:
            float: Portfolio value, or None if unavailable.
        """
        if as_at_date is None:
            return None

        as_of_ts = pd.Timestamp(as_at_date)
        holdings = db.session.query(
            Trades.ticker,
            Stocks.currency,
            Trades.date,
            Trades.quantity,
            Trades.price,
            Trades.direction
        ).join(
            Stocks, Trades.ticker == Stocks.ticker
        ).filter(
            Trades.user_id == self.id,
            Trades.date <= as_of_ts.to_pydatetime()
        ).order_by(Trades.ticker, Trades.date, Trades.id).all()

        if not holdings:
            return None

        positions: Dict[str, Dict] = {}
        for ticker, currency, trade_date, quantity, trade_price, direction in holdings:
            if ticker not in positions:
                positions[ticker] = {
                    'currency': currency,
                    'trades': [],
                    'last_trade_price': None,
                    'last_trade_date': None
                }
            direction = str(direction).lower()
            if direction == 'buy':
                quantity_delta = float(quantity or 0.0)
            elif direction == 'sell':
                quantity_delta = -float(quantity or 0.0)
            else:
                continue
            positions[ticker]['trades'].append((trade_date, quantity_delta))
            if trade_price is not None:
                positions[ticker]['last_trade_price'] = float(trade_price)
                positions[ticker]['last_trade_date'] = trade_date

        total_value = 0.0
        valued_position_count = 0
        for ticker, position in positions.items():
            quantity = 0.0
            splits = db.session.query(
                StockPrices.date,
                StockPrices.splits
            ).filter(
                StockPrices.ticker == ticker,
                StockPrices.date <= as_of_ts.to_pydatetime(),
                StockPrices.splits.isnot(None),
                StockPrices.splits != 0
            ).order_by(StockPrices.date).all()
            for trade_date, quantity_delta in position['trades']:
                adjusted_quantity = quantity_delta
                for split_date, split in splits:
                    if trade_date <= split_date:
                        adjusted_quantity *= float(split)
                quantity += adjusted_quantity
            if not quantity:
                continue

            price_row = db.session.query(
                StockPrices.date,
                StockPrices.close
            ).filter(
                StockPrices.ticker == ticker,
                StockPrices.date <= as_of_ts.to_pydatetime()
            ).order_by(StockPrices.date.desc()).first()
            if price_row is None or price_row.close is None:
                price = position['last_trade_price']
                valuation_date = position['last_trade_date']
                if price is None or valuation_date is None:
                    logger.warning(
                        'No market or transaction price for %s on or before %s; valuing at zero',
                        ticker,
                        as_of_ts.date()
                    )
                    valued_position_count += 1
                    continue
                logger.warning(
                    'No stored market price for %s on or before %s; using transaction price from %s',
                    ticker,
                    as_of_ts.date(),
                    valuation_date
                )
            else:
                price = float(price_row.close)
                valuation_date = price_row.date

            fx = 1.0
            currency = position['currency']
            ticker_type = data.split_ticker(ticker)[1]
            if currency != self.default_currency and ticker_type not in ('CASH', 'LOAN'):
                fx_ticker = f'{currency}{self.default_currency}=X.FX'
                fx_row = db.session.query(StockPrices.close).filter(
                    StockPrices.ticker == fx_ticker,
                    StockPrices.date <= valuation_date
                ).order_by(StockPrices.date.desc()).first()
                if fx_row is not None and fx_row.close is not None:
                    fx = float(fx_row.close)

            total_value += quantity * price * fx
            valued_position_count += 1

        return total_value if valued_position_count else None

    def _get_nav_snapshot(
        self,
        as_of_date: datetime,
        nav_history: Optional[pd.DataFrame] = None
    ) -> Optional[Dict[str, float]]:
        """
        Retrieve NAV per unit, total units, and portfolio value for a given date.

        Prefers cached month-end values, otherwise falls back to targeted NAV history.
        """
        if as_of_date is None:
            return None

        as_of_ts = pd.Timestamp(as_of_date)
        month_end_ts = as_of_ts.to_period('M').to_timestamp('M')

        if as_of_ts.normalize() == month_end_ts:
            cached_row = PortfolioMonthlyNav.query.filter_by(
                user_id=self.id,
                month_end=month_end_ts.date()
            ).first()
            if cached_row and not cached_row.needs_refresh:
                return {
                    'nav_per_unit': float(cached_row.nav_per_unit or 0.0),
                    'total_units': float(cached_row.total_units or 0.0),
                    'portfolio_value': float(cached_row.portfolio_value or 0.0)
                }

        selected_nav = None
        if nav_history is not None and not nav_history.empty:
            tmp_history = nav_history.copy()
            tmp_history['Date'] = pd.to_datetime(tmp_history['Date'])
            eligible = tmp_history[tmp_history['Date'] <= as_of_ts]
            if not eligible.empty:
                selected_nav = eligible.iloc[-1]

        nav_per_unit: Optional[float] = None
        total_units: Optional[float] = None

        if selected_nav is not None:
            nav_per_unit = float(selected_nav.get('NAV_per_Unit', 0.0) or 0.0)
            total_units = float(selected_nav.get('Cumulative_Units', 0.0) or 0.0)

        if nav_per_unit is None or total_units is None:
            total_units, nav_per_unit = self._get_nav_from_trades(as_of_ts)

        if nav_per_unit is None or total_units is None:
            lookback_start = as_of_ts - pd.DateOffset(days=31)
            recalc_from = lookback_start.to_pydatetime()
            nav_history_local = self.calculate_nav_history(
                recalc_from_date=recalc_from,
                end_date=as_of_ts.to_pydatetime()
            )
            if nav_history_local.empty:
                nav_per_unit = nav_per_unit or 0.0
                total_units = total_units or 0.0
            else:
                nav_history_local['Date'] = pd.to_datetime(nav_history_local['Date'])
                eligible = nav_history_local[nav_history_local['Date'] <= as_of_ts]
                latest_row = eligible.iloc[-1] if not eligible.empty else nav_history_local.iloc[-1]
                nav_per_unit = float(latest_row.get('NAV_per_Unit', 0.0) or 0.0)
                total_units = float(latest_row.get('Cumulative_Units', 0.0) or 0.0)

        portfolio_value = self._portfolio_value_on(as_of_ts.to_pydatetime())
        if portfolio_value is None:
            portfolio_value = nav_per_unit * total_units if nav_per_unit and total_units else 0.0

        if nav_per_unit == 0 and total_units and portfolio_value:
            try:
                nav_per_unit = portfolio_value / total_units if total_units else 0.0
            except ZeroDivisionError:
                nav_per_unit = 0.0

        return {
            'nav_per_unit': float(nav_per_unit or 0.0),
            'total_units': float(total_units or 0.0),
            'portfolio_value': float(portfolio_value or 0.0)
        }

    def _get_nav_from_trades(
        self,
        as_of_ts: pd.Timestamp,
        trades_df: Optional[pd.DataFrame] = None
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Retrieve total units and NAV per unit from the latest trade on or before as_of_ts.
        """
        if trades_df is not None and not trades_df.empty:
            df = trades_df[trades_df['Date'] <= as_of_ts].copy()
            if not df.empty:
                pf_shares_col = next((c for c in ('pf_shares', 'Pf_shares') if c in df.columns), None)
                pf_price_col = next((c for c in ('pf_price', 'Pf_price') if c in df.columns), None)
                if pf_shares_col and pf_price_col:
                    df = df.sort_values(['Date'])
                    last_row = df.iloc[-1]
                    shares_val = last_row.get(pf_shares_col)
                    price_val = last_row.get(pf_price_col)
                    if pd.notna(shares_val) and pd.notna(price_val):
                        try:
                            total_units = float(shares_val or 0.0)
                        except (TypeError, ValueError):
                            total_units = None
                        try:
                            nav_per_unit = float(price_val or 0.0)
                        except (TypeError, ValueError):
                            nav_per_unit = None
                        if total_units is not None and nav_per_unit is not None:
                            logger.debug(
                                'Using cached trade NAV for snapshot on %s (Date %s: units=%s, nav=%s)',
                                as_of_ts.date(),
                                last_row.get('Date'),
                                total_units,
                                nav_per_unit
                            )
                            return total_units, nav_per_unit

        trade_row = Trades.query.filter(
            Trades.user_id == self.id,
            Trades.date <= as_of_ts.to_pydatetime(),
            Trades.pf_price.isnot(None),
            Trades.pf_shares.isnot(None)
        ).order_by(Trades.date.desc(), Trades.id.desc()).first()

        if trade_row is None:
            return None, None

        try:
            total_units = float(trade_row.pf_shares or 0.0)
        except (TypeError, ValueError):
            total_units = None

        try:
            nav_per_unit = float(trade_row.pf_price or 0.0)
        except (TypeError, ValueError):
            nav_per_unit = None

        if total_units is not None and nav_per_unit is not None:
            logger.debug(
                'Using stored trade NAV for snapshot on %s (trade %s: units=%s, nav=%s)',
                as_of_ts.date(),
                trade_row.id,
                total_units,
                nav_per_unit
            )

        return total_units, nav_per_unit

    def calculate_nav_history(self, recalc_from_date: datetime = None, end_date: datetime = None) -> pd.DataFrame:
        """
        Calculate NAV history for the user's portfolio.

        Args:
            recalc_from_date: If provided, only recalculate from this date onwards.
                             If None, calculate from inception.

        Returns:
            DataFrame with columns: trade_id, Date, Portfolio_Value, NAV_per_Unit,
                                  Units_Change, Cumulative_Units, Trade_CF
        """
        logger.info(f'Starting NAV calculation for user {self.id} (end_date={end_date})')

        trades_df = self.get_trades()
        if trades_df.empty:
            logger.warning('No trades found for user')
            return pd.DataFrame()

        # Handle index properly to avoid RangeIndex issues
        if trades_df.index.name == 'id':
            trades_df = trades_df.reset_index()
            trade_id_col = 'id'
        else:
            trade_id_col = 'id' if 'id' in trades_df.columns else 'Id' if 'Id' in trades_df.columns else None
        if trade_id_col is None:
            logger.error('Trades data missing id column')
            raise KeyError('Trades data missing id column')

        pf_shares_col = 'pf_shares' if 'pf_shares' in trades_df.columns else 'Pf_shares' if 'Pf_shares' in trades_df.columns else None
        pf_price_col = 'pf_price' if 'pf_price' in trades_df.columns else 'Pf_price' if 'Pf_price' in trades_df.columns else None
        if pf_shares_col is None or pf_price_col is None:
            logger.error('Trades data missing pf_shares or pf_price columns')
            raise KeyError('Trades data missing pf_shares or pf_price columns')

        cf_col_trades = next((col for col in ('CF', 'Cf', 'cf') if col in trades_df.columns), None)

        trades_df['Date'] = pd.to_datetime(trades_df['Date'])
        trades_df['Direction'] = trades_df['Direction'].astype(str)

        if end_date is not None:
            trades_df = trades_df[trades_df['Date'] <= pd.Timestamp(end_date)]
            if trades_df.empty:
                logger.warning(f'No trades found on or before {end_date}')
                return pd.DataFrame()

        trades_df = trades_df.sort_values(['Date', trade_id_col]).reset_index(drop=True)

        start_ts = pd.Timestamp(recalc_from_date) if recalc_from_date is not None else None
        if start_ts is None:
            start_idx = 0
            logger.info('Calculating NAV from inception')
        else:
            start_mask = trades_df['Date'] >= start_ts
            if not start_mask.any():
                logger.info(f'No trades to recalculate on or after {start_ts}')
                return pd.DataFrame()
            start_idx = start_mask[start_mask].index[0] if start_mask.any() else 0
            logger.info(f'Partial recalculation from {start_ts}')

        if start_idx >= len(trades_df):
            logger.info('No trades to process after applying start/end date filters')
            return pd.DataFrame()

        trades_subset = trades_df.iloc[start_idx:].copy()
        if trades_subset.empty:
            logger.info('No trades to process after applying start/end date filters')
            return pd.DataFrame()

        logger.info(
            f"NAV calculation range: {trades_subset.iloc[0]['Date'].date()} to {trades_subset.iloc[-1]['Date'].date()} "
            f"({len(trades_subset)} trades)"
        )

        # Establish starting cumulative units and NAV from the last confirmed trade before the recalculation point
        cumulative_units = 0.0
        current_nav_per_unit = 1.0
        baseline_idx = start_idx - 1
        while baseline_idx >= 0:
            baseline_row = trades_df.iloc[baseline_idx]
            prev_units = baseline_row.get(pf_shares_col)
            prev_nav = baseline_row.get(pf_price_col)
            if pd.notna(prev_units) and pd.notna(prev_nav):
                cumulative_units = float(prev_units or 0.0)
                try:
                    current_nav_per_unit = float(prev_nav)
                except (TypeError, ValueError):
                    current_nav_per_unit = 1.0
                logger.debug(
                    'Reusing stored NAV baseline from trade %s (%s units @ %s)',
                    baseline_row[trade_id_col],
                    cumulative_units,
                    current_nav_per_unit
                )
                break
            baseline_idx -= 1

        nav_results = []
        grouped = trades_subset.groupby('Date', sort=True)
        for trade_date, same_date_trades in grouped:
            same_date_trades = same_date_trades.sort_values(trade_id_col)

            nav_for_contrib = current_nav_per_unit if current_nav_per_unit and current_nav_per_unit > 0 else 1.0
            daily_records = []

            for _, same_date_trade in same_date_trades.iterrows():
                direction = str(same_date_trade['Direction']).lower()
                fx = float(same_date_trade.get('Fx', 1.0) or 1.0)
                quantity = float(same_date_trade.get('Quantity', 0.0) or 0.0)
                price = float(same_date_trade.get('Price', 0.0) or 0.0)
                fees = float(same_date_trade.get('Fees', 0.0) or 0.0)

                gross_value = quantity * price * fx
                total_fees = fees * fx

                if direction == 'buy':
                    trade_cf = -(gross_value + total_fees)
                elif direction == 'sell':
                    trade_cf = gross_value - total_fees
                elif direction == 'div':
                    trade_cf = same_date_trade.get(cf_col_trades, 0.0) if cf_col_trades else gross_value
                else:
                    cf_raw = same_date_trade.get(cf_col_trades, 0.0) if cf_col_trades else 0.0
                    trade_cf = float(cf_raw) if pd.notna(cf_raw) else 0.0

                if direction == 'buy':
                    units_change = (-trade_cf) / nav_for_contrib if nav_for_contrib > 0 else 0.0
                elif direction == 'sell':
                    units_change = -trade_cf / nav_for_contrib if nav_for_contrib > 0 else 0.0
                else:
                    units_change = 0.0

                cumulative_units += units_change
                daily_records.append({
                    'trade_id': same_date_trade[trade_id_col],
                    'Date': trade_date,
                    'Units_Change': units_change,
                    'Cumulative_Units': cumulative_units,
                    'Trade_CF': trade_cf
                })

            portfolio_value = self._portfolio_value_on(pd.Timestamp(trade_date).to_pydatetime())
            if portfolio_value is None:
                logger.warning(f'Could not get portfolio value for {trade_date}')
                portfolio_value = nav_for_contrib * cumulative_units if cumulative_units > 0 else 0.0

            if cumulative_units > 0:
                current_nav_per_unit = portfolio_value / cumulative_units
            else:
                current_nav_per_unit = 0.0

            for record in daily_records:
                record.update({
                    'Portfolio_Value': portfolio_value,
                    'NAV_per_Unit': current_nav_per_unit
                })
                nav_results.append(record)

        result_df = pd.DataFrame(nav_results)
        logger.info(f'NAV calculation complete. Processed {len(result_df)} trades')
        return result_df

    def _build_dividend_schedule(
        self,
        trades_df: pd.DataFrame,
        month_end_dates: Set[date]
    ) -> Dict[date, float]:
        """
        Build a mapping of month-end date -> dividend cashflow (base currency) using
        price-derived dividend data combined with current holdings.
        """
        if trades_df.empty or not month_end_dates:
            return {}

        tickers = trades_df['Ticker'].dropna().unique().tolist()
        if not tickers:
            return {}

        month_end_ts = [pd.Timestamp(month) for month in month_end_dates]
        last_month_end = max(month_end_ts)
        first_trade_ts = trades_df['Date'].min()
        if pd.isna(first_trade_ts):
            return {}

        dividend_rows = db.session.query(
            StockPrices.ticker,
            StockPrices.date,
            StockPrices.dividends,
            Stocks.currency
        ).join(
            Stocks, StockPrices.ticker == Stocks.ticker
        ).filter(
            StockPrices.ticker.in_(tickers),
            StockPrices.dividends != 0,
            StockPrices.date >= first_trade_ts,
            StockPrices.date <= last_month_end
        ).all()

        if not dividend_rows:
            return {}

        dividends = pd.DataFrame(
            dividend_rows,
            columns=['Ticker', 'Date', 'Dividend', 'Currency']
        )
        dividends['Date'] = pd.to_datetime(dividends['Date'])
        dividends = dividends[dividends['Dividend'] != 0].copy()
        if dividends.empty:
            return {}

        manual_div_mask = trades_df['Direction'].astype(str).str.lower().isin(('div', 'dividend'))
        if manual_div_mask.any():
            manual_pairs = set(
                zip(
                    trades_df.loc[manual_div_mask, 'Ticker'],
                    trades_df.loc[manual_div_mask, 'Date'].dt.normalize()
                )
            )
            dividends = dividends[~dividends.apply(
                lambda row: (row['Ticker'], row['Date'].normalize()) in manual_pairs,
                axis=1
            )]
            if dividends.empty:
                return {}

        trades_copy = trades_df[['Ticker', 'Date', 'Quantity', 'Direction']].copy()
        trades_copy['Direction'] = trades_copy['Direction'].astype(str).str.lower()
        trades_copy['Quantity'] = trades_copy['Quantity'].astype(float).fillna(0.0)
        trades_copy['UnitsDelta'] = 0.0
        trades_copy.loc[trades_copy['Direction'] == 'buy', 'UnitsDelta'] = trades_copy['Quantity']
        trades_copy.loc[trades_copy['Direction'] == 'sell', 'UnitsDelta'] = -trades_copy['Quantity']
        trades_copy = trades_copy.sort_values(['Ticker', 'Date'])

        if trades_copy['UnitsDelta'].abs().sum() == 0:
            return {}

        per_ticker_results = []
        for ticker, ticker_divs in dividends.groupby('Ticker', sort=False):
            ticker_trades = trades_copy[trades_copy['Ticker'] == ticker]
            if ticker_trades.empty:
                continue

            ticker_trades = ticker_trades[['Date', 'UnitsDelta']].sort_values('Date')
            ticker_trades['CumUnits'] = ticker_trades['UnitsDelta'].cumsum()
            baseline_date = ticker_trades['Date'].min() - pd.Timedelta(seconds=1)
            baseline = pd.DataFrame({'Date': [baseline_date], 'CumUnits': [0.0]})
            ticker_units = pd.concat([baseline, ticker_trades[['Date', 'CumUnits']]], ignore_index=True)
            ticker_units = ticker_units.sort_values('Date').reset_index(drop=True)

            unit_dates = ticker_units['Date'].to_numpy()
            unit_values = ticker_units['CumUnits'].to_numpy()
            div_dates = ticker_divs['Date'].sort_values().to_numpy()

            if unit_dates.size == 0 or div_dates.size == 0:
                continue

            search_idx = np.searchsorted(unit_dates, div_dates, side='right') - 1
            valid_mask = search_idx >= 0
            if not valid_mask.any():
                continue

            aligned = ticker_divs.sort_values('Date').reset_index(drop=True).copy()
            aligned['CumUnits'] = 0.0
            aligned.loc[valid_mask, 'CumUnits'] = unit_values[search_idx[valid_mask]]
            aligned = aligned[aligned['CumUnits'] != 0.0]
            if aligned.empty:
                continue
            per_ticker_results.append(aligned)

        if not per_ticker_results:
            return {}

        div_df = pd.concat(per_ticker_results, ignore_index=True)

        fx_df = div_df[['Ticker', 'Currency', 'Date']].copy()
        fx_df = self.get_fx(fx_df)
        div_df = div_df.merge(fx_df[['Ticker', 'Date', 'Fx']], on=['Ticker', 'Date'], how='left')
        div_df['Fx'] = div_df['Fx'].fillna(1.0)

        div_df['CashAmount'] = div_df['Dividend'].astype(float) * div_df['CumUnits'].astype(float) * div_df['Fx']

        div_df['Month_End'] = div_df['Date'].dt.to_period('M').dt.to_timestamp('M').dt.date
        div_df = div_df[div_df['Month_End'].isin(month_end_dates)]
        if div_df.empty:
            return {}

        monthly_totals = div_df.groupby('Month_End')['CashAmount'].sum()
        return {month: float(amount) for month, amount in monthly_totals.items()}

    def _ensure_monthly_nav_cache(self, month_ends: List[pd.Timestamp], trades_df: pd.DataFrame,
                                  baseline_month: Optional[pd.Timestamp] = None) -> Tuple[pd.DataFrame, Optional[float]]:
        """
        Ensure cached monthly NAV rows exist and are fresh for the requested months.
        """
        if not month_ends:
            return pd.DataFrame(), None

        month_end_dates = {
            pd.Timestamp(month_end).to_period('M').to_timestamp('M').date()
            for month_end in month_ends
        }

        baseline_date = None
        if baseline_month is not None:
            baseline_date = pd.Timestamp(baseline_month).to_period('M').to_timestamp('M').date()
            month_end_dates.add(baseline_date)

        if trades_df is None:
            trades_df = self.get_trades()

        if trades_df.empty:
            return pd.DataFrame(), None

        trades_df = trades_df.copy()
        trades_df['Date'] = pd.to_datetime(trades_df['Date'])
        first_trade_ts = trades_df['Date'].min()
        latest_price_ts = self._latest_price_timestamp()

        cf_col = next((col for col in ('CF', 'Cf', 'cf') if col in trades_df.columns), None)

        def _trade_cash_flow(trade_row: pd.Series) -> Tuple[float, float]:
            direction = str(trade_row.get('Direction', '')).lower()
            fx = float(trade_row.get('Fx', 1.0) or 1.0)
            quantity = float(trade_row.get('Quantity', 0.0) or 0.0)
            price = float(trade_row.get('Price', 0.0) or 0.0)
            fees = float(trade_row.get('Fees', 0.0) or 0.0)
            cf_value = float(trade_row.get(cf_col, 0.0) or 0.0) if cf_col else 0.0

            if direction == 'buy':
                return -(quantity * price * fx + fees * fx), 0.0
            if direction == 'sell':
                return quantity * price * fx - fees * fx, 0.0
            if direction in ('div', 'dividend'):
                if cf_col and cf_value:
                    dividend_value = cf_value
                else:
                    dividend_value = quantity * price * fx
                return 0.0, dividend_value
            if cf_col:
                return cf_value, 0.0
            return 0.0, 0.0

        dividend_schedule = self._build_dividend_schedule(trades_df, month_end_dates)

        def _month_flows(month_ts: pd.Timestamp) -> Tuple[float, float]:
            month_start = month_ts.replace(day=1)
            capital_flow = 0.0
            schedule_key = month_ts.to_period('M').to_timestamp('M').date()
            dividend_flow = dividend_schedule.get(schedule_key, 0.0)
            month_mask = (trades_df['Date'] >= month_start) & (trades_df['Date'] <= month_ts)
            if month_mask.any():
                for _, trade_row in trades_df.loc[month_mask].iterrows():
                    cf, div = _trade_cash_flow(trade_row)
                    capital_flow += cf
                    dividend_flow += div
            return capital_flow, dividend_flow

        existing_rows: List[PortfolioMonthlyNav] = []
        today_month_end = datetime.utcnow().replace(day=1).replace(hour=0, minute=0, second=0, microsecond=0)
        today_month_end = pd.Timestamp(today_month_end).to_period('M').to_timestamp('M').date()

        if month_end_dates:
            existing_rows = PortfolioMonthlyNav.query.filter(
                PortfolioMonthlyNav.user_id == self.id,
                PortfolioMonthlyNav.month_end.in_(month_end_dates)
            ).all()

        existing_map = {row.month_end: row for row in existing_rows}
        latest_price_date = latest_price_ts.date() if latest_price_ts else None
        months_to_refresh = sorted(
            month for month in month_end_dates
            if month not in existing_map or existing_map[month].needs_refresh
        )
        months_to_refresh = [month for month in months_to_refresh if month <= today_month_end]

        carry_units = 0.0
        carry_nav = 0.0
        carry_value = 0.0
        if months_to_refresh and existing_map:
            prior_dates = [d for d in existing_map if d < months_to_refresh[0]]
            if prior_dates:
                prev_row = existing_map[max(prior_dates)]
                carry_units = float(prev_row.total_units or 0.0)
                carry_nav = float(prev_row.nav_per_unit or 0.0)
                carry_value = float(prev_row.portfolio_value or 0.0)

        nav_history = None
        computed_rows: Dict = {}
        cache_modified = False

        if months_to_refresh:
            refresh_end_date = max(months_to_refresh)
            recalc_start_date = pd.Timestamp(min(months_to_refresh)).replace(day=1)

            for month_date in months_to_refresh:
                month_ts = pd.Timestamp(month_date)
                portfolio_value = self._portfolio_value_on(month_ts) or 0.0
                capital_flow, dividend_flow = _month_flows(month_ts)

                units_from_trade, nav_from_trade = self._get_nav_from_trades(month_ts, trades_df=trades_df)
                total_units = units_from_trade
                nav_per_unit = nav_from_trade

                if total_units is None or nav_per_unit is None:
                    first_trade_ts = trades_df['Date'].min()
                    if first_trade_ts is not pd.NaT and month_ts < first_trade_ts:
                        total_units = total_units if total_units is not None else 0.0
                        nav_per_unit = nav_per_unit if nav_per_unit is not None else carry_nav
                        nav_per_unit = nav_per_unit if nav_per_unit is not None else 0.0
                    else:
                        if nav_history is None:
                            nav_history = self.calculate_nav_history(
                                recalc_from_date=recalc_start_date,
                                end_date=pd.Timestamp(refresh_end_date)
                            )
                        if nav_history.empty:
                            logger.warning('No NAV history available while refreshing cache (fallback to trade data only)')
                        else:
                            nav_history['Date'] = pd.to_datetime(nav_history['Date'])
                            nav_history = nav_history.sort_values('Date')

                    if nav_history is not None and not nav_history.empty:
                        history_to_date = nav_history[nav_history['Date'] <= month_ts]
                        if not history_to_date.empty:
                            latest_nav = history_to_date.iloc[-1]
                            if total_units is None:
                                total_units = float(latest_nav.get('Cumulative_Units', 0.0) or 0.0)
                            if nav_per_unit is None:
                                nav_per_unit = float(latest_nav.get('NAV_per_Unit', 0.0) or 0.0)

                if total_units is None:
                    total_units = 0.0
                if nav_per_unit is None:
                    nav_per_unit = 0.0

                row = existing_map.get(month_date)
                should_cache = month_date <= today_month_end

                if total_units == 0.0 and carry_units:
                    total_units = carry_units

                # Units only change with external capital flows, so they may be
                # carried forward from the latest trade.  NAV per unit must be
                # revalued at every month end from the stored portfolio value.
                if total_units and portfolio_value:
                    nav_per_unit = portfolio_value / total_units
                elif nav_per_unit == 0.0:
                    if total_units:
                        nav_per_unit = carry_nav
                    elif carry_nav:
                        nav_per_unit = carry_nav

                if portfolio_value == 0.0:
                    if total_units and nav_per_unit:
                        portfolio_value = nav_per_unit * total_units
                    elif carry_value:
                        portfolio_value = carry_value

                if should_cache:
                    if row is None:
                        row = PortfolioMonthlyNav(user_id=self.id, month_end=month_date)
                        existing_map[month_date] = row
                    row.portfolio_value = portfolio_value
                    row.total_units = total_units
                    row.nav_per_unit = nav_per_unit
                    row.capital_flow = capital_flow
                    row.dividend_flow = dividend_flow
                    row.calculated_at = datetime.utcnow()
                    row.needs_refresh = False
                    db.session.add(row)
                    cache_modified = True
                else:
                    if row is not None:
                        row.needs_refresh = True
                        db.session.add(row)
                        cache_modified = True
                computed_rows[month_date] = {
                    'Month_End_Date': month_ts,
                    'Portfolio_Value': float(portfolio_value or 0.0),
                    'Total_Units': float(total_units or 0.0),
                    'NAV_per_Unit': float(nav_per_unit or 0.0),
                    'Capital_Flow': float(capital_flow or 0.0),
                    'Dividend_Flow': float(dividend_flow or 0.0)
                }

                carry_units = float(total_units or 0.0)
                carry_nav = float(nav_per_unit or 0.0)
                carry_value = float(portfolio_value or 0.0)

            if cache_modified:
                db.session.commit()

        future_months = sorted(
            month for month in month_end_dates if month > today_month_end
        )

        if future_months:
            refresh_end_date = pd.Timestamp(future_months[-1])
            recalc_start_date = pd.Timestamp(future_months[0]).replace(day=1)

        for month_date in future_months:
            month_ts = pd.Timestamp(month_date)
            portfolio_value = self._portfolio_value_on(month_ts) or 0.0
            total_units, nav_per_unit = self._get_nav_from_trades(month_ts, trades_df=trades_df)

            if total_units is None or nav_per_unit is None:
                if nav_history is None:
                    nav_history = self.calculate_nav_history(
                        recalc_from_date=recalc_start_date,
                        end_date=refresh_end_date
                    )
                    if nav_history.empty:
                        nav_history = pd.DataFrame()
                    else:
                        nav_history['Date'] = pd.to_datetime(nav_history['Date'])
                        nav_history = nav_history.sort_values('Date')

                if nav_history is not None and not nav_history.empty:
                    history_to_date = nav_history[nav_history['Date'] <= month_ts]
                    if not history_to_date.empty:
                        latest_nav = history_to_date.iloc[-1]
                        if total_units is None:
                            total_units = float(latest_nav.get('Cumulative_Units', 0.0) or 0.0)
                        if nav_per_unit is None:
                            nav_per_unit = float(latest_nav.get('NAV_per_Unit', 0.0) or 0.0)

            if total_units is None:
                total_units = 0.0
            if nav_per_unit is None:
                nav_per_unit = 0.0

            capital_flow, dividend_flow = _month_flows(month_ts)

            if portfolio_value == 0.0 and total_units and nav_per_unit:
                portfolio_value = nav_per_unit * total_units

            if total_units and portfolio_value:
                nav_per_unit = portfolio_value / total_units

            computed_rows[month_date] = {
                'Month_End_Date': month_ts,
                'Portfolio_Value': float(portfolio_value or 0.0),
                'Total_Units': float(total_units or 0.0),
                'NAV_per_Unit': float(nav_per_unit or 0.0),
                'Capital_Flow': float(capital_flow or 0.0),
                'Dividend_Flow': float(dividend_flow or 0.0)
            }

        final_rows = []
        if month_end_dates:
            final_rows = PortfolioMonthlyNav.query.filter(
                PortfolioMonthlyNav.user_id == self.id,
                PortfolioMonthlyNav.month_end.in_(month_end_dates)
            ).all()

        final_map = {row.month_end: row for row in final_rows}
        baseline_nav = None
        if baseline_date is not None:
            baseline_row = final_map.get(baseline_date)
            if baseline_row is not None:
                baseline_nav = float(baseline_row.nav_per_unit or 0.0)

        data_rows = []
        for month_end in month_ends:
            month_date = pd.Timestamp(month_end).to_period('M').to_timestamp('M').date()
            cached_data = computed_rows.get(month_date)
            if cached_data is not None:
                data_rows.append(cached_data)
                continue

            row = final_map.get(month_date)
            if row is None or (month_date > today_month_end and row.needs_refresh):
                continue
            data_rows.append({
                'Month_End_Date': pd.Timestamp(row.month_end),
                'Portfolio_Value': float(row.portfolio_value or 0.0),
                'Total_Units': float(row.total_units or 0.0),
                'NAV_per_Unit': float(row.nav_per_unit or 0.0),
                'Capital_Flow': float(row.capital_flow or 0.0),
                'Dividend_Flow': float(row.dividend_flow or 0.0)
            })

        return pd.DataFrame(data_rows), baseline_nav

    def update_nav_in_trades(
        self,
        force_full_recalc: bool = False,
        end_date: datetime = None,
        recalc_start: datetime = None,
        recalc_end: datetime = None,
        perform_price_refresh: bool = False
    ) -> int:
        """
        Update pf_price and pf_shares columns in trades table with NAV data.

        Args:
            force_full_recalc: If True, recalculate entire history from inception.

        Returns:
            Number of trades updated
        """
        logger.info(
            f'Updating NAV in trades for user {self.id}, '
            f'force_full_recalc={force_full_recalc}, end_date={end_date}, '
            f'recalc_start={recalc_start}, recalc_end={recalc_end}, '
            f'perform_price_refresh={perform_price_refresh}'
        )

        max_trade_dt = db.session.query(func.max(Trades.date)).filter(Trades.user_id == self.id).scalar()
        price_target = recalc_end or end_date or max_trade_dt
        if force_full_recalc:
            # Force full recalculation
            recalc_from_date = recalc_start
            if perform_price_refresh and price_target:
                try:
                    self.update_prices(as_at_date=price_target)
                except Exception as exc:
                    logger.warning(f'Price refresh failed prior to full NAV recalculation: {exc}')
        else:
            # Check for trades missing NAV data
            query = db.session.query(Trades).filter(
                Trades.user_id == self.id,
                (Trades.pf_price.is_(None)) | (Trades.pf_shares.is_(None))
            )
            if end_date is not None:
                query = query.filter(Trades.date <= end_date)

            missing_nav = query.order_by(Trades.date, Trades.id).first()

            if missing_nav is None:
                logger.info('All trades already have NAV data for specified range')
                return 0

            recalc_from_date = missing_nav.date
            if recalc_start and recalc_from_date < recalc_start:
                recalc_from_date = recalc_start
            logger.info(f'Found trades missing NAV data from {recalc_from_date}')

        # Calculate NAV history
        nav_df = self.calculate_nav_history(recalc_from_date, end_date=recalc_end or end_date)

        if nav_df.empty:
            logger.warning('No NAV data to update')
            return 0

        # Update trades with NAV data
        updated_count = 0
        for _, row in nav_df.iterrows():
            trade = Trades.query.get(row['trade_id'])
            if trade and trade.user_id == self.id:
                trade.pf_price = row['NAV_per_Unit']
                trade.pf_shares = row['Cumulative_Units']
                updated_count += 1

        db.session.commit()
        if recalc_from_date is None:
            self.mark_all_monthly_nav_cache_stale()
        else:
            self.mark_monthly_nav_cache_stale(recalc_from_date)
        logger.info(f'Updated {updated_count} trades with NAV data')
        return updated_count

    def nav_monthly_summary(
        self,
        start_date: datetime,
        end_date: datetime,
        force_full_recalc: bool = False,
        recalc_start: datetime = None,
        recalc_end: datetime = None,
        perform_price_refresh: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Generate monthly NAV summary for the specified date range.

        Args:
            start_date: Start date for summary
            end_date: End date for summary

        Returns:
            DataFrame with monthly NAV data
        """
        logger.info(f'Generating monthly NAV summary from {start_date} to {end_date}')

        # Ensure NAV data is current up to end_date
        self.update_nav_in_trades(
            end_date=end_date,
            force_full_recalc=force_full_recalc,
            recalc_start=recalc_start,
            recalc_end=recalc_end,
            perform_price_refresh=perform_price_refresh
        )

        # Generate month-end dates inclusive of partial months
        month_ends = pd.period_range(start_date, end_date, freq='M').to_timestamp('M').tolist()
        if not month_ends:
            month_ends = [pd.Timestamp(end_date)]

        trades_df = self.get_trades()
        if trades_df.empty:
            logger.warning('NAV summary: no trades available')
            return pd.DataFrame(), {}

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        trades_df = trades_df.copy()
        trades_df['Date'] = pd.to_datetime(trades_df['Date'])
        inception_date = trades_df['Date'].min()

        baseline_month = None
        if inception_date is not None and start_ts > inception_date:
            baseline_month = start_ts - pd.offsets.MonthEnd(1)

        cache_df, baseline_cached_nav = self._ensure_monthly_nav_cache(month_ends, trades_df, baseline_month)
        if cache_df.empty:
            logger.warning('NAV summary: no cached NAV data available for requested period')
            return pd.DataFrame(), {}

        cache_df = cache_df.sort_values('Month_End_Date')

        baseline_nav = None
        if inception_date is not None and start_ts <= inception_date:
            baseline_nav = 1.0
        elif baseline_cached_nav and baseline_cached_nav > 0:
            baseline_nav = baseline_cached_nav

        monthly_rows = []
        prev_nav = None
        total_invested = 0.0
        total_withdrawn = 0.0
        total_dividends = 0.0

        for row in cache_df.itertuples(index=False):
            month_end = pd.Timestamp(row.Month_End_Date)
            if month_end < start_ts or month_end > end_ts:
                continue

            nav_per_unit = float(row.NAV_per_Unit or 0.0)
            total_units = float(row.Total_Units or 0.0)
            portfolio_value = float(row.Portfolio_Value or 0.0)
            capital_flow = float(row.Capital_Flow or 0.0)
            dividend_flow = float(row.Dividend_Flow or 0.0)

            if baseline_nav is None and nav_per_unit > 0:
                baseline_nav = nav_per_unit

            monthly_return = (nav_per_unit / prev_nav) - 1 if (prev_nav and prev_nav > 0) else 0.0
            inception_return = (nav_per_unit / baseline_nav) - 1 if (baseline_nav and baseline_nav > 0) else 0.0

            if capital_flow < 0:
                total_invested += -capital_flow
            elif capital_flow > 0:
                total_withdrawn += capital_flow
            if dividend_flow:
                total_dividends += dividend_flow

            monthly_rows.append({
                'Month_End_Date': month_end,
                'NAV_per_Unit': nav_per_unit,
                'Total_Units': total_units,
                'Portfolio_Value': portfolio_value,
                'Monthly_Return_%': monthly_return * 100,
                'Inception_Return_%': inception_return * 100,
                'Capital_Flow': capital_flow,
                'Dividend_Flow': dividend_flow,
                'Net_Cash_Flow': capital_flow + dividend_flow
            })

            if nav_per_unit > 0:
                prev_nav = nav_per_unit

            logger.info(
                f'NAV summary: completed month {month_end.date()} | NAV/unit={nav_per_unit:.4f}, '
                f'total_units={total_units:.2f}, portfolio_value={portfolio_value:.2f}'
            )

        result_df = pd.DataFrame(monthly_rows).sort_values('Month_End_Date').reset_index(drop=True)
        logger.info(f'Generated monthly NAV summary with {len(result_df)} months')

        totals: Dict[str, float] = {}
        if not result_df.empty:
            starting_value = float(result_df.iloc[0]['Portfolio_Value'] or 0.0)
            ending_value = float(result_df.iloc[-1]['Portfolio_Value'] or 0.0)
            net_new_capital = total_invested - total_withdrawn
            value_change = ending_value - starting_value
            investment_gain = value_change - net_new_capital + total_dividends
            net_cash_after_dividends = net_new_capital - total_dividends

            totals = {
                'capital_invested': total_invested,
                'capital_withdrawn': total_withdrawn,
                'net_new_capital': net_new_capital,
                'dividends_received': total_dividends,
                'net_cash_after_dividends': net_cash_after_dividends,
                'starting_value': starting_value,
                'ending_value': ending_value,
                'portfolio_value_change': value_change,
                'investment_gain': investment_gain
            }
        else:
            totals = {
                'capital_invested': 0.0,
                'capital_withdrawn': 0.0,
                'net_new_capital': 0.0,
                'dividends_received': 0.0,
                'net_cash_after_dividends': 0.0,
                'starting_value': 0.0,
                'ending_value': 0.0,
                'portfolio_value_change': 0.0,
                'investment_gain': 0.0
            }

        return result_df, totals

    def nav_performance_metrics(self, start_date: datetime = None, end_date: datetime = None) -> dict:
        """
        Calculate NAV performance metrics for a given period.

        Returns:
            Dictionary with performance metrics
        """
        logger.info(f'Calculating NAV performance metrics for period {start_date} to {end_date}')

        if end_date is None:
            end_date = datetime.now()

        # Ensure NAV data is current
        self.update_nav_in_trades(end_date=end_date)

        trades_df = self.get_trades()
        if trades_df.empty:
            logger.warning('No trades available for NAV metrics')
            return {}

        trades_df = trades_df[trades_df['Date'] <= end_date]
        if trades_df.empty:
            logger.warning('No trades on or before end date for NAV metrics')
            return {}

        trades_df = trades_df.copy()
        trades_df['Date'] = pd.to_datetime(trades_df['Date'])
        inception_date_ts = pd.Timestamp(trades_df['Date'].min())
        inception_date_dt = inception_date_ts.to_pydatetime()

        period_start_dt = inception_date_dt
        if start_date:
            period_start_dt = max(start_date, inception_date_dt)

        period_start_ts = pd.Timestamp(period_start_dt)
        end_ts = pd.Timestamp(end_date)
        end_dt = end_ts.to_pydatetime()
        days_in_period = max((end_ts - period_start_ts).days, 0)

        month_ends_to_cache = []
        if period_start_ts.normalize() == period_start_ts.to_period('M').to_timestamp('M'):
            month_ends_to_cache.append(period_start_ts)
        if end_ts.normalize() == end_ts.to_period('M').to_timestamp('M'):
            month_ends_to_cache.append(end_ts)

        if month_ends_to_cache:
            self._ensure_monthly_nav_cache(month_ends_to_cache, trades_df=trades_df)

        nav_history_window = None
        start_snapshot = self._get_nav_snapshot(period_start_dt)
        end_snapshot = self._get_nav_snapshot(end_dt)

        if start_snapshot is None or end_snapshot is None:
            lookback_start = period_start_ts - pd.DateOffset(days=31)
            recalc_from = None
            if lookback_start.to_pydatetime() > inception_date_dt:
                recalc_from = lookback_start.to_pydatetime()
            nav_history_window = self.calculate_nav_history(
                recalc_from_date=recalc_from,
                end_date=end_ts.to_pydatetime()
            )
            if start_snapshot is None:
                start_snapshot = self._get_nav_snapshot(period_start_dt, nav_history=nav_history_window)
            if end_snapshot is None:
                end_snapshot = self._get_nav_snapshot(end_dt, nav_history=nav_history_window)

        if end_snapshot is None:
            logger.warning('Unable to determine NAV snapshot for end date')
            return {}

        if start_snapshot is None:
            start_snapshot = {'nav_per_unit': 1.0, 'total_units': 0.0, 'portfolio_value': 0.0}

        current_nav_per_unit = float(end_snapshot.get('nav_per_unit', 0.0) or 0.0)
        current_total_units = float(end_snapshot.get('total_units', 0.0) or 0.0)
        current_portfolio_value = float(end_snapshot.get('portfolio_value', 0.0) or 0.0)

        start_nav_per_unit = float(start_snapshot.get('nav_per_unit', 0.0) or 0.0)
        if start_nav_per_unit <= 0:
            start_nav_per_unit = 1.0

        if start_nav_per_unit > 0:
            nav_ratio = current_nav_per_unit / start_nav_per_unit
            total_return_pct = (nav_ratio - 1.0) * 100
        else:
            nav_ratio = None
            total_return_pct = 0.0

        if nav_ratio and nav_ratio > 0 and days_in_period > 0:
            exponent = 365.25 / days_in_period
            annualized_return_pct = (math.exp(math.log(nav_ratio) * exponent) - 1) * 100
        else:
            if days_in_period == 0:
                logger.warning('Cannot compute annualized return due to zero duration')
            else:
                logger.warning('Cannot compute annualized return due to non-positive NAV ratio')
            annualized_return_pct = 0.0

        logger.info(
            'NAV metrics summary: '
            f'start_nav={start_nav_per_unit:.4f}, end_nav={current_nav_per_unit:.4f}, '
            f'nav_ratio={nav_ratio if nav_ratio else None}, total_return_pct={total_return_pct:.2f}, '
            f'annualized_return_pct={annualized_return_pct:.2f}, days_in_period={days_in_period}'
        )

        # Calculate IRR using existing method
        try:
            hist_pos = self.hist_positions(
                start_date=period_start_dt,
                as_at_date=end_date,
                splits=[],
                divs=[],
                include_dividends=True,
                calculate_gains=True
            )
            curr_p = self.current_prices(
                tickers=self.get_tickers(),
                as_at_date=datetime.now(),
                last_change=False
            )
            irr_df = self.calc_IRR(hist_pos[['Date', 'Ticker', 'CF', 'CumQuan']].copy(),
                                   curr_p[['Date', 'Ticker', 'Close']].copy())
            irr_pct = float(irr_df[irr_df['Ticker'] == 'Total']['IRR'].iloc[0]) * 100 if not irr_df.empty else 0.0
        except Exception as e:
            logger.warning(f'Could not calculate IRR: {e}')
            irr_pct = 0.0

        price_warning = None
        price_data_as_at = None
        try:
            portfolio_df, _ = self.info_date(as_at_date=end_date, hide_zero_pos=True)
            if not portfolio_df.empty and 'Date' in portfolio_df.columns:
                latest_price_ts = pd.to_datetime(portfolio_df['Date']).max()
                if pd.notna(latest_price_ts):
                    price_data_as_at = latest_price_ts.to_pydatetime()
                    if latest_price_ts.date() < end_ts.date():
                        price_warning = (
                            f'Latest price data is as at {latest_price_ts.date()}, please refresh prices.'
                        )
        except Exception as exc:
            logger.warning(f'Unable to verify price recency: {exc}')

        return {
            'current_nav_per_unit': current_nav_per_unit,
            'total_return_pct': total_return_pct,
            'annualized_return_pct': annualized_return_pct,
            'irr_pct': irr_pct,
            'current_total_units': current_total_units,
            'current_portfolio_value': current_portfolio_value,
            'inception_date': period_start_dt,
            'days_in_period': days_in_period,
            'price_warning': price_warning,
            'price_data_as_at': price_data_as_at
        }


class Trades(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey(
        'user.id', name='fk_trades_user_id'))
    date = db.Column(db.DateTime, index=True)
    ticker = db.Column(db.String(20), db.ForeignKey(
        'stocks.ticker', name='fk_trades_ticker'), nullable=False)
    quantity = db.Column(db.Numeric(20, 10), index=True)
    price = db.Column(db.Numeric(20, 10), index=True)
    fees = db.Column(db.Numeric(20, 10), index=True)
    direction = db.Column(db.String(10), index=True)
    pf_price = db.Column(db.Numeric(20, 10), index=True)
    pf_shares = db.Column(db.Numeric(20, 10), index=True)
    # fx = db.Column(db.Numeric(20, 10), index=True)

    def __repr__(self):
        return f'<{self.id}: {self.direction} trade on {self.date} for {self.quantity} of {self.ticker} at {self.price}>'


class PortfolioMonthlyNav(db.Model):
    __tablename__ = 'portfolio_monthly_nav'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', name='fk_pmnav_user_id'), nullable=False, index=True)
    month_end = db.Column(db.Date, nullable=False)
    portfolio_value = db.Column(db.Numeric(20, 10))
    total_units = db.Column(db.Numeric(20, 10))
    nav_per_unit = db.Column(db.Numeric(20, 10))
    capital_flow = db.Column(db.Numeric(20, 10))
    dividend_flow = db.Column(db.Numeric(20, 10))
    calculated_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    needs_refresh = db.Column(db.Boolean, nullable=False, default=True, index=True)

    __table_args__ = (
        UniqueConstraint('user_id', 'month_end', name='uq_portfolio_monthly_nav_user_month'),
    )

    def __repr__(self):
        return f'<MonthlyNAV user={self.user_id} month_end={self.month_end} nav={self.nav_per_unit}>'


class Stocks(db.Model):
    ticker = db.Column(db.String(20), primary_key=True)
    name = db.Column(db.String(60), index=True)
    currency = db.Column(db.String(10), index=True)
    last_updated = db.Column(db.DateTime(), index=True)

    def __repr__(self):
        return f'<{self.ticker}: {self.name} and quoted in {self.currency} and last updated on {self.last_updated}>'

    def update_name(self, name: str = None):
        if name is None:
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
    ticker = db.Column(db.String(20), db.ForeignKey(
        'stocks.ticker', name='fk_prices_ticker'), nullable=False)
    date = db.Column(db.DateTime, index=True)
    open = db.Column(db.Numeric(40, 20), index=True)
    high = db.Column(db.Numeric(40, 20), index=True)
    low = db.Column(db.Numeric(40, 20), index=True)
    close = db.Column(db.Numeric(40, 20), index=True)
    volume = db.Column(db.Numeric(40, 20), index=True)
    adjclose = db.Column(db.Numeric(40, 20), index=True)
    dividends = db.Column(db.Numeric(40, 20), index=True)
    splits = db.Column(db.Numeric(20, 10), index=True)

    __table_args__ = (UniqueConstraint(
        'ticker', 'date', name='unique_ticker_date'), )

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
        """Calculate the previous close price for this ticker"""
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
