from datetime import datetime
from flask_login import UserMixin
import logging
import pandas as pd
import traceback

from werkzeug.security import generate_password_hash, check_password_hash

from app import db, login
from utils import data


logger = logging.getLogger('pt_logger')


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
        # return '<User {}>'.format(self.username)
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

    def add_trades(self, df: pd.DataFrame, append: bool = True):
        if append:
            # If append is true, get existing trades and append passed df to existing trades
            exist_df = self.get_trades()
            df.rename(str.capitalize, axis=1, inplace=True)
            if not exist_df.empty:
                df = pd.concat([exist_df, df], ignore_index=True, join='inner')

        # Update user_id to be current id
        df['User_id'] = self.id
        logger.info(df)

        # Checks if tickers already in db and, if not, insert into DB
        df_tickers = df['Ticker'].unique()
        current_tickers = Stocks.current_tickers()
        for ticker in df_tickers:
            if ticker not in current_tickers:
                stock = Stocks(ticker=ticker.upper())
                stock.update_name()
                stock.update_currency(
                    pf_currency=self.default_currency)
                stock.update_last_updated(None)
                db.session.add(stock)
                db.session.commit()

        # Remove existing trades and add all trades to DB. Rollback changes if any errors
        try:
            Trades.query.filter_by(user_id=self.id).delete()
            db.session.commit()
            df.to_sql('trades', db.engine,
                      if_exists='append', index=False)
        except Exception as e:
            # db.session.rollback()
            logger.debug(
                f'-------------- Exception {traceback.print_exc()} --------------')

    def drop_trades(self):
        Trades.query.filter_by(user_id=self.id).delete()
        db.session.commit()

    def update_last_accessed(self, date):
        self.last_accessed = date
        db.session.commit()

    def get_stock_info(self):
        df = pd.read_sql(Stocks.query.filter(Stocks.ticker.in_(
            [i.ticker for i in self.trades.all()])).statement, db.engine)
        df.rename(str.capitalize, axis=1, inplace=True)
        return df


class Trades(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey(
        'user.id', name='fk_trades_user_id'))
    date = db.Column(db.DateTime, index=True)
    # ticker = db.Column(db.String(20), index=True, nullable=False)
    ticker = db.Column(db.String(20), db.ForeignKey(
        'stocks.ticker', name='fk_trades_ticker'), nullable=False)
    quantity = db.Column(db.Numeric(20, 10), index=True)
    price = db.Column(db.Numeric(20, 10), index=True)
    fees = db.Column(db.Numeric(20, 10), index=True)
    direction = db.Column(db.String(10), index=True)
    pf_price = db.Column(db.Numeric(20, 10), index=True)
    pf_shares = db.Column(db.Numeric(20, 10), index=True)
    fx = db.Column(db.Numeric(20, 10), index=True)

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

    def update_currency(self, pf_currency):
        self.currency = data.get_currency(self.ticker, pf_currency)
        db.session.commit()

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
