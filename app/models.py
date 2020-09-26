from flask_login import UserMixin
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash

from app import db, login


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(120), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    trades = db.relationship('Trades', backref='user', lazy='dynamic')

    def __repr__(self):
        return '<User {}>'.format(self.username)

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
            if not exist_df.empty:
                df = pd.concat([exist_df, df], ignore_index=True)

        # Replaces current trades with new df
        self.drop_trades()
        df.rename(str.lower, axis=1, inplace=True)
        df['user_id'] = self.id
        df.to_sql('trades', db.engine, if_exists='append', index=False)

    def drop_trades(self):
        Trades.query.filter_by(user_id=self.id).delete()
        db.session.commit()


class Trades(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    date = db.Column(db.DateTime, index=True)
    ticker = db.Column(db.String(10), index=True)
    quantity = db.Column(db.Float, index=True)
    price = db.Column(db.Float, index=True)
    fees = db.Column(db.Float, index=True)
    direction = db.Column(db.String(10), index=True)

    def __repr__(self):
        return f'<{self.id}: {self.direction} trade on {self.date} for {self.quantity} of {self.ticker} at {self.price}>'
