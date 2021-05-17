from datetime import datetime, timedelta, timezone
import logging
from os import path

from flask import Flask, flash, render_template, request, redirect, url_for, abort
from flask.globals import current_app
from flask.helpers import make_response
from flask_login import current_user, login_user, logout_user
from flask_login.utils import login_required
import pandas as pd
import plotly.io as pio
from werkzeug.urls import url_parse

from app import app, db
from app.forms import LoginForm, RegistrationForm
from app.models import Stocks, User, Trades
from app.portfolio import Portfolio
from utils import web_utils

logger = logging.getLogger('pt_logger')

# app = Flask(__name__)
DATA_FILE = 'data/pf_data.pkl'


@app.route('/')
@app.route('/index')
@login_required
def index():
    current_user.update_last_accessed(datetime.utcnow())
    return update_pf()
    # return render_template('home.jinja2', title="Portfolio Tracker: Home")


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('index')
        return redirect(next_page)
    return render_template('login.jinja2', title='Sign In', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you have been registered! Please login.', 'info')
        return redirect(url_for('login'))
    return render_template('register.jinja2', title='Register', form=form)


@app.route('/update', methods=['POST'])
@login_required
def update_pf():
    as_at_date = get_date(request.form.get(
        'date'), request.form.get('time_offset'))

    hide_zero = not(bool(request.form.get('hide_zero'))) or False
    no_update = not(bool(request.form.get('no_update'))) or False
    currency = request.form.get('currency') or 'AUD'

    pf_trades = current_user.get_trades()
    if pf_trades.empty:
        flash('Portfolio is empty. Please add some trades', 'error')
        return render_template('home.jinja2', title="Overview")
    stocks_df = current_user.get_stock_info()
    pf = Portfolio(trades=pf_trades, currency=currency,
                   filename=DATA_FILE, stocks_df=stocks_df)
    start = datetime.now()
    df = pf.info_date(as_at_date=as_at_date,
                      hide_zero_pos=hide_zero, no_update=no_update)
    logger.info(f'info_date took {(datetime.now()-start)} to run')

    start = datetime.now()
    df['Date'] = df['Date'].dt.strftime('%d-%m-%y')
    df_html = web_utils.pandas_table_styler(
        df, neg_cols=['%LastChange', '$LastChange', '%UnRlGain', 'RlGain', 'UnRlGain', 'TotalGain'], left_align_cols=['Ticker', 'Name'], ticker_links=True, uuid='portfolio')
    df_html = web_utils.add_footer(df_html)
    as_at_date = str(as_at_date.date())
    df_html = web_utils.update_links(df_html, currency, as_at_date)
    logger.info(f'render HTML took {(datetime.now()-start)} to run')
    return render_template('home.jinja2', tables=df_html, title="Overview")


@ app.route('/load', methods=['GET', 'POST'])
@ login_required
def load_trades_csv():
    if request.method == 'POST':
        pf_file = request.files['file']
        # checks if file is in allowed extensions
        if pf_file.filename != '':
            file_ext = path.splitext(pf_file.filename)[1]
            if file_ext not in current_app.config['UPLOAD_EXTENSIONS']:
                flash("Uploaded file is not CSV. Please upload CSV file. ", "error")
                return redirect(url_for('add_trades'))

        # checks if file can be loaded into dataframe
        try:
            trade_df = pd.read_csv(pf_file, parse_dates=[
                'Date'], dayfirst=True, thousands=',')
            current_user.add_trades(trade_df)
            flash("Loaded successfully", "info")
        except Exception as e:
            flash("An error occured, try again!", "error")
    return redirect(url_for('index'))


@ app.route('/save', methods=['GET', 'POST'])
@ login_required
def save_pf():
    pf_trades = current_user.get_trades()
    if pf_trades.empty:
        flash('No trades to export / save. Please add trades and try again', 'error')
        return render_template('home.jinja2', title='Overview')
    resp = make_response(pf_trades.to_csv(index=False))
    resp.headers.set("Content-Disposition",
                     "attachment", filename="trades.csv")
    return resp


@ app.route('/add_trades', methods=['GET', 'POST'])
@ login_required
def add_trades():
    if request.method == 'POST':
        trades_df = web_utils.resp_to_trades_df(request)
        for t in trades_df['Ticker'].unique():
            if Stocks.check_stock_exists(t) == None:
                stock = Stocks(ticker=t)
                stock.update_name()
                stock.update_currency(
                    pf_currency=current_user.default_currency)
                stock.update_last_updated(None)
                db.session.add(stock)
                db.session.commit()
        current_user.add_trades(trades_df)
        flash('Trades added successfully', 'info')
        return render_template('home.jinja2', title='Overview')
    return render_template('add_trades.jinja2', title='Add Trades')


@ app.route('/view_trades', methods=['GET', 'POST'])
@ login_required
def view_trades():
    if request.method == 'GET':
        df = current_user.get_trades()
        if df.empty:
            flash('No portfolio trades founds. Please load file or add trades', 'error')
            return render_template('home.jinja2', title='Overview')

        # format date to allow render in date input field
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        # add delete trade column
        df['Delete?'] = 0

        df_html = (df.reset_index(drop=True).style
                   .format({c: web_utils.html_input(c) for c in df.columns})
                   .set_uuid('view_trades')
                   .hide_index()
                   .render()
                   )
        return render_template('view_trades.jinja2', tables=df_html, title='View Trades')
    else:
        # gets updated trade data
        trades_df = web_utils.resp_to_trades_df(request)

        # checks and deletes relevant rows
        trades_df['Delete?'] = request.form.getlist('Delete?')
        trades_df = trades_df[trades_df['Delete?'] == '0']
        trades_df.drop(['Delete?'], axis=1, inplace=True)
        trades_df.reset_index(inplace=True, drop=True)
        current_user.add_trades(trades_df, append=False)

        flash('Successfully updated trades', 'info')
        return redirect(url_for('view_trades', title='View Trades'))


@ app.route('/stock/<ticker>')
@ login_required
def stock(ticker: str):
    currency = request.args.get('currency')
    as_at_date = get_date(request.args.get(
        'date'), request.form.get('time_offset'))
    logger.info(f'Loading trades from db')
    pf_trades = current_user.get_trades()
    start_date = pf_trades[pf_trades['Ticker'] == ticker]['Date'].min()
    trades = pf_trades[pf_trades['Ticker'] == ticker].to_html()
    stocks_df = current_user.get_stock_info()
    pf = Portfolio(trades=pf_trades, currency=currency,
                   filename=DATA_FILE, stocks_df=stocks_df)
    hist_pos, divs, splits = pf.price_history(start_date=start_date,
                                              ticker=ticker, as_at_date=as_at_date, period='D', no_update=False)
    position_fig = web_utils.create_fig(hist_pos, 'Date', ['CurrVal', 'TotalGain'], [
        'RlGain', 'UnRlGain', 'Dividends', 'Quantity'], 600)
    position = pio.to_html(
        position_fig, include_plotlyjs='cdn', full_html=False)

    return render_template('stock_dynamic.jinja2', title=f'Overview for {ticker}', stock_name=ticker, postition_df=position, divs=divs.to_html(), splits=splits.to_html(), trades=trades)


@app.route('/pfactions', methods=['GET', 'POST'])
@login_required
def pfactions():
    if request.form["action"] == "Export to CSV":
        return exportpf()
    else:
        return update_pf()


@ app.route('/exportpf', methods=['GET', 'POST'])
@ login_required
def exportpf():
    as_at_date = get_date(request.form.get(
        'date'), request.form.get('time_offset'))
    hide_zero = not(bool(request.form.get('hide_zero'))) or False
    no_update = not(bool(request.form.get('no_update'))) or False
    currency = request.form.get('currency') or 'AUD'

    pf_trades = current_user.get_trades()
    stocks_df = current_user.get_stock_info()
    pf = Portfolio(trades=pf_trades, currency=currency,
                   filename=DATA_FILE, stocks_df=stocks_df)
    df = pf.info_date(as_at_date=as_at_date,
                      hide_zero_pos=hide_zero, no_update=no_update)
    resp = make_response(df.to_csv(index=False))
    resp.headers.set("Content-Disposition",
                     "attachment", filename="pf_position.csv")
    return resp


@app.route('/tax', methods=['GET', 'POST'])
@login_required
def tax():
    title = 'Tax Summary'
    if request.method == 'POST':
        if (request.form.get(
                'start_date') == '') or (request.form.get(
                'end_date') == ''):
            flash('Please insert dates and submit query', 'info')
            return render_template('tax.jinja2', title=title)
        if request.form["action"] == "Export to CSV":
            return exportpftax(title)
        else:
            return taxoutput(title)
    else:
        flash('Please insert dates and submit query', 'info')
        return render_template('tax.jinja2', title=title)


def exportpftax(title: str):
    df = get_tax_df(title)
    resp = make_response(df.to_csv(index=False))
    resp.headers.set("Content-Disposition",
                     "attachment", filename="pf_position.csv")
    return resp


def taxoutput(title: str):
    df = get_tax_df(title)
    df['Date'] = df['Date'].dt.strftime('%d-%m-%y')
    df_html = web_utils.pandas_table_styler(
        df, neg_cols=['RlGain'], left_align_cols=['Ticker', 'Name'], ticker_links=False, uuid='taxsummary')
    df_html = web_utils.add_footer(df_html)
    return render_template('tax.jinja2', tables=df_html, title=title)


def get_tax_df(title: str):
    start_date = get_date(request.form.get(
        'start_date'), None)
    end_date = get_date(request.form.get(
        'end_date'), None)

    hide_zero = False
    no_update = True
    currency = 'AUD'

    pf_trades = current_user.get_trades()
    if pf_trades.empty:
        flash('Portfolio is empty. Please add some trades', 'error')
        return render_template('home.jinja2', title="Overview")

    stocks_df = current_user.get_stock_info()
    pf = Portfolio(trades=pf_trades, currency=currency,
                   filename=DATA_FILE, stocks_df=stocks_df)
    df = pf.info_date(start_date=start_date, as_at_date=end_date,
                      hide_zero_pos=hide_zero, no_update=no_update)

    # restrict output to stocks where there was a tax event in the period (i.e. dividends or capital gains)
    df = df[(df['RlGain'] != 0) | (df['Dividends'] != 0)].copy()
    df = df[['Ticker', 'Name', 'CurrVal', 'Dividends', 'RlGain', 'Date', 'Type']]

    return df


def get_date(date: str, offset: str):
    """
    Takes date and offset as string and if None or left blank, gets time_offset and converts to today's date from UTC to date in local timezone

    Args:
        date (str): date in string format
        offset (str): timezone offset in string format
    """
    as_at_date = None
    if date == '' or date == None:
        if offset == None:
            tz = timezone(timedelta(minutes=0))
        else:
            tz = timezone(timedelta(minutes=-int(offset)))
        as_at_date = pd.to_datetime(
            'today', utc=True).tz_convert(tz).tz_localize(None)
        logger.info(f'Localised datetime is: {as_at_date}')
    else:
        as_at_date = datetime.strptime(date, "%Y-%m-%d")
    return as_at_date


if __name__ == '__main__':
    app.run(debug=False, use_reloader=True)
