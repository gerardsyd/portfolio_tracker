from datetime import datetime
import logging
from os import path

from flask import Flask, flash, render_template, request, redirect, url_for
from flask.helpers import make_response
import pandas as pd
import plotly.io as pio

from app import app
from app.forms import LoginForm
from app.portfolio import Portfolio
from utils import web_utils

logger = logging.getLogger('pt_logger')

# app = Flask(__name__)
TRADES_FILE = 'data/pf_trades.pkl'
DATA_FILE = 'data/pf_data.pkl'
NAMES_FILE = 'data/pf_names.pkl'


@app.route('/')
@app.route('/index')
def index():
    return updatepf()
    # return render_template('home.jinja2', title="Portfolio Tracker: Home")


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        flash(
            f'Login requested for user {form.username.data}, remember_me={form.remember_me.data}')
        return redirect(url_for('index'))
    return render_template('login.jinja2', title='Sign In', form=form)


@app.route('/update', methods=['POST'])
def updatepf():
    # get as at date from drop down. If left blank, set none (defaults to today per portfolio info function)
    as_at_date = None if request.form.get(
        'date') == '' else request.form.get('date')
    hide_zero = not(bool(request.form.get('hide_zero'))) or False
    no_update = not(bool(request.form.get('no_update'))) or False
    currency = request.form.get('currency') or 'AUD'

    start = datetime.now()
    if path.isfile(TRADES_FILE):
        logger.info(f'{TRADES_FILE} exists, loading')
        pf_trades = pd.read_pickle(TRADES_FILE)
        pf = Portfolio(trades=pf_trades, currency=currency,
                       filename=DATA_FILE, names_filename=NAMES_FILE)
    else:
        pf = Portfolio(filename=DATA_FILE,
                       names_filename=NAMES_FILE, currency=currency)
    logger.info(f'file loading took {(datetime.now()-start)} to run')
    start = datetime.now()
    df = pf.info_date(as_at_date, hide_zero_pos=hide_zero, no_update=no_update)
    logger.info(f'info_date took {(datetime.now()-start)} to run')
    start = datetime.now()
    df['Date'] = df['Date'].dt.strftime('%d-%m-%y')
    if as_at_date == None:
        as_at_date = datetime.strftime(pd.to_datetime('today'), '%Y-%m-%d')
    df_html = web_utils.pandas_table_styler(
        df, neg_cols=['%LastChange', '$LastChange', '%UnRlGain', 'RlGain', 'UnRlGain', 'TotalGain'], left_align_cols=['Ticker', 'Name'], ticker_links=True, uuid='portfolio')
    df_html = web_utils.add_footer(df_html)
    df_html = web_utils.update_links(df_html, currency, as_at_date)
    logger.info(f'render HTML took {(datetime.now()-start)} to run')
    start = datetime.now()
    pf.trades_df.to_pickle(TRADES_FILE)
    logger.info(f'trades_df.to_pickle took {(datetime.now()-start)} to run')
    return render_template('home.jinja2', tables=df_html, title="Overview")


@app.route('/load', methods=['GET', 'POST'])
def loadpf():
    if request.method == 'POST':
        try:
            trade_df = pd.read_csv(request.files.get(
                'pf_file'), parse_dates=['Date'], dayfirst=True, thousands=',')
            trade_df.to_pickle(TRADES_FILE)
            message = "Loaded successfully"
        except:
            message = "An error occured, try again!"
        return render_template('home.jinja2', message=message, title='Overview')


@app.route('/save', methods=['GET', 'POST'])
def savepf():
    if path.isfile(TRADES_FILE):
        logger.info(f'{TRADES_FILE} exists, loading')
        pf_trades = pd.read_pickle(TRADES_FILE)
    else:
        return render_template('home.jinja2', message='File not found', title='Overview')
    resp = make_response(pf_trades.to_csv(index=False))
    resp.headers.set("Content-Disposition",
                     "attachment", filename="trades.csv")
    return resp


@app.route('/add_trades', methods=['GET', 'POST'])
def add_trades():
    if request.method == 'POST':
        trades_df = web_utils.resp_to_trades_df(request)
        if path.isfile(TRADES_FILE):
            logger.info(f'{TRADES_FILE} exists, loading')
            pf_trades = pd.read_pickle(TRADES_FILE)
            pf = Portfolio(pf_trades, DATA_FILE)
            pf.add_trades(trades_df)
        else:
            pf = Portfolio(trades_df, DATA_FILE)
        pf.trades_df.reset_index(inplace=True, drop=True)
        pf.trades_df.to_pickle(TRADES_FILE)
        return render_template('home.jinja2', message='Trades added successfully', title='Overview')
    return render_template('add_trades.jinja2', title='Add Trades')


@app.route('/view_trades', methods=['GET', 'POST'])
def view_trades():
    if request.method == 'GET':
        if path.isfile(TRADES_FILE):
            logger.info(f'{TRADES_FILE} exists, loading')
            df = pd.read_pickle(TRADES_FILE)
        else:
            return render_template('home.jinja2', message='No portfolio trades founds. Please load file or add trades', title='Overview')

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
        return render_template('view_trades.jinja2', tables=df_html, message=request.args.get('message'), title='View Trades')
    else:
        # gets updated trade data
        trades_df = web_utils.resp_to_trades_df(request)

        # checks and deletes relevant rows
        trades_df['Delete?'] = request.form.getlist('Delete?')
        trades_df = trades_df[trades_df['Delete?'] == '0']
        trades_df.drop(['Delete?'], axis=1, inplace=True)
        trades_df.reset_index(inplace=True, drop=True)
        trades_df.to_pickle(TRADES_FILE)
        return redirect(url_for('view_trades', message='Successfully updated trades', title='View Trades'))


@app.route('/stock/<ticker>')
def stock(ticker):
    currency = request.args.get('currency')
    as_at_date = datetime.strptime(request.args.get('date'), '%Y-%m-%d')
    logger.info(f'{TRADES_FILE} exists, loading')
    pf_trades = pd.read_pickle(TRADES_FILE)
    trades = pf_trades[pf_trades['Ticker'] == ticker].to_html()
    pf = Portfolio(trades=pf_trades, currency=currency,
                   filename=DATA_FILE, names_filename=NAMES_FILE)
    hist_pos, divs, splits = pf.price_history(
        ticker=ticker, as_at_date=as_at_date, period='D', no_update=False)

    position_fig = web_utils.create_fig(hist_pos, 'Date', ['CurrVal', 'TotalGain'], [
                                        'RlGain', 'UnRlGain', 'Dividends', 'Quantity'], 600)
    position = pio.to_html(
        position_fig, include_plotlyjs='cdn', full_html=False)

    # position = web_utils.pandas_table_styler(hist_pos, neg_cols=[
    #                                          '%LastChange', 'RlGain', 'UnRlGain', 'TotalGain'], left_align_cols=['Ticker'], ticker_links=False, uuid='stock')
    return render_template('stock_dynamic.jinja2', title=f'Overview for {ticker}', stock_name=ticker, postition_df=position, divs=divs.to_html(), splits=splits.to_html(), trades=trades)


if __name__ == '__main__':
    app.run(debug=False, use_reloader=True)
