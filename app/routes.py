from datetime import datetime, timedelta, timezone
import logging
from os import path
import traceback

from flask import flash, render_template, request, redirect, url_for
from flask.globals import current_app
from flask.helpers import make_response
from flask_login import current_user, login_user, logout_user
from flask_login.utils import login_required
import pandas as pd
import plotly.io as pio
from werkzeug.urls import url_parse

from app import app, db
from app.forms import LoginForm, RegistrationForm, UpdateDetailsForm
from app.models import User
from utils import web_utils

logger = logging.getLogger('pt_logger')


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

    hide_zero = not (bool(request.form.get('hide_zero'))) or False
    no_update = not (bool(request.form.get('no_update'))) or False
    currency = request.form.get('currency') or 'AUD'

    pf_trades = current_user.get_trades()
    if pf_trades.empty:
        flash('Portfolio is empty. Please add some trades', 'error')
        return render_template('home.jinja2', title="Overview")
    start = datetime.now()
    if not no_update:
        current_user.update_prices(as_at_date=as_at_date)
    df = current_user.info_date(as_at_date=as_at_date, hide_zero_pos=hide_zero)
    logger.info(f'info_date took {(datetime.now()-start)} to run')

    start = datetime.now()
    # removes any IRR with very large numbers and sets to None for display purposes
    df.loc[df['IRR'] > 10 ^ 6, 'IRR'] = None

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
            trade_df['Ticker'] = trade_df['Ticker'].str.upper()
            current_user.add_trades(trade_df)
            flash("Loaded successfully", "info")
        except Exception as e:
            logger.debug(
                f'------------- An error {traceback.print_exc()} occurred ----------------')
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
        current_user.add_trades(trades_df)
        flash('Trades added successfully', 'info')
        return render_template('home.jinja2', title='Overview')
    return render_template('add_trades.jinja2', title='Add Trades')


@ app.route('/view_trades', methods=['GET', 'POST'])
@ login_required
def view_trades():
    if request.method == 'GET':
        df = current_user.get_trades()
        df.drop(columns=['Pf_price', 'Pf_shares'], inplace=True)
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
                   .hide(axis='index')
                   .to_html()
                   )
        return render_template('view_trades.jinja2', tables=df_html, title='View Trades')
    else:
        # gets updated trade data
        trades_df = web_utils.resp_to_trades_df(request)

        # checks and deletes relevant rows
        trades_df = trades_df[trades_df['Delete?'] == '0']
        trades_df.drop(['Delete?'], axis=1, inplace=True)
        trades_df.reset_index(inplace=True, drop=True)
        current_user.add_trades(trades_df, append=False)

        flash('Successfully updated trades', 'info')
        return redirect(url_for('view_trades', title='View Trades'))


@ app.route('/stock/<ticker>')
@ login_required
def stock(ticker: str):
    # currency = request.args.get('currency')
    as_at_date = get_date(request.args.get('date'), request.form.get('time_offset'))
    logger.info('Loading trades from db')
    pf_trades = current_user.get_trades()
    start_date = pf_trades[pf_trades['Ticker'] == ticker]['Date'].min()
    trades = pf_trades[pf_trades['Ticker'] == ticker].to_html()
    hist_pos, divs, splits = current_user.price_history(start_date=start_date,
                                                        ticker=ticker, as_at_date=as_at_date, period='D')
    position_fig = web_utils.create_fig(hist_pos, 'Date', ['CurrVal', 'TotalGain'], [
        'RlGain', 'UnRlGain', 'CumDiv', 'Quantity'], 600)
    position = pio.to_html(
        position_fig, include_plotlyjs='cdn', full_html=False)

    return render_template('stock_dynamic.jinja2', title=f'Overview for {ticker}', stock_name=ticker, postition_df=position, divs=divs.to_html(), splits=splits.to_html(), trades=trades)


@app.route('/pfactions', methods=['GET', 'POST'])
@login_required
def pfactions():
    if "action" in request.form and request.form["action"] == "Export to CSV":
        return exportpf()
    else:
        return update_pf()


@ app.route('/exportpf', methods=['GET', 'POST'])
@ login_required
def exportpf():
    as_at_date = get_date(request.form.get(
        'date'), request.form.get('time_offset'))
    hide_zero = not (bool(request.form.get('hide_zero'))) or False
    # currency = request.form.get('currency') or 'AUD'

    df = current_user.info_date(as_at_date=as_at_date, hide_zero_pos=hide_zero)
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
        if "action" in request.form and request.form["action"] == "Export to CSV":
            return exportpftax(title)
        else:
            return taxoutput(title)
    else:
        flash('Please insert dates and submit query', 'info')
        return render_template('tax.jinja2', title=title)


@app.route('/user/<username>', methods=['GET', 'POST'])
@login_required
def profile(username):
    form = UpdateDetailsForm()
    if form.validate_on_submit():
        # update database
        if current_user.check_password(form.existing_password.data):
            if form.password.data is not None:
                current_user.set_password(form.password.data)
            current_user.default_currency = form.currency.data
            db.session.commit()
            flash('Your changes have been saved!', 'info')
            return redirect(url_for('profile', username=current_user.username))
        else:
            flash('Existing password is incorrect. Please try again!', 'error')
    elif request.method == 'GET':
        form.email.data = current_user.email
        form.currency.data = current_user.default_currency
    return render_template('profile.jinja2', username=username, title="Profile", form=form)


def exportpftax(title: str):
    df = get_tax_df(title)
    resp = make_response(df.to_csv(index=False))
    resp.headers.set("Content-Disposition",
                     "attachment", filename="pf_position.csv")
    return resp


def taxoutput(title: str):
    df = get_tax_df(title)
    if df.empty:
        df_html = "<p><div class='alert alert-primary' role='alert'> No dividends or capital gains in period</div>"
    else:
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
    # currency = 'AUD'

    pf_trades = current_user.get_trades()
    if pf_trades.empty:
        flash('Portfolio is empty. Please add some trades', 'error')
        return render_template('home.jinja2', title="Overview")

    df = current_user.info_date(
        start_date=start_date, as_at_date=end_date, hide_zero_pos=hide_zero)

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
    if date == '' or date is None:
        if offset is None:
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
