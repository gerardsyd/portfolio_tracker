from collections import Counter
from datetime import datetime, timedelta, timezone
from io import BytesIO
import logging
from os import path
import traceback

from flask import flash, jsonify, render_template, request, redirect, url_for, Response
from flask.globals import current_app
from flask.helpers import make_response
from flask_login import current_user, login_user, logout_user
from flask_login.utils import login_required
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from werkzeug.urls import url_parse

from app import app, db
from app.forms import LoginForm, RegistrationForm, UpdateDetailsForm
from app.models import User, Stocks
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
    if not current_app.config.get('REGISTRATION_ENABLED', False):
        flash('Registration is disabled', 'error')
        return redirect(url_for('login'))
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
        return render_template('home.jinja2', title="Overview", last_updated=None)
    start = datetime.now()
    if not no_update:
        current_user.update_prices(as_at_date=as_at_date)
    df, _ = current_user.info_date(
        as_at_date=as_at_date, hide_zero_pos=hide_zero)
    raw_df = df.copy()
    logger.info(f'info_date took {(datetime.now()-start)} to run')

    start = datetime.now()
    # removes any IRR with very large numbers and sets to None for display purposes
    df.loc[df['IRR'] > 10 ^ 6, 'IRR'] = None

    df['Date'] = df['Date'].dt.strftime('%d-%m-%y')
    df_html = web_utils.pandas_table_styler(
        df, neg_cols=['%LastChange', '$LastChange', 'IRR', '%UnRlGain', 'RlGain', 'UnRlGain', 'TotalGain'], left_align_cols=['Ticker', 'Name'], ticker_links=True, uuid='portfolio')
    df_html = web_utils.add_footer(df_html)
    as_at_date = str(as_at_date.date())
    df_html = web_utils.update_links(df_html, currency, as_at_date)
    logger.info(f'render HTML took {(datetime.now()-start)} to run')
    holdings_df = raw_df[raw_df['Ticker'].notna() & (raw_df['Ticker'] != 'Total')]
    holdings_df = holdings_df[~holdings_df['Ticker'].astype(str).str.contains(r'\.FX$', na=False)]
    date_series = pd.to_datetime(holdings_df['Date'], errors='coerce').dropna()
    if not date_series.empty:
        counts = Counter(date_series.dt.date)
        most_common_date, _ = max(counts.items(), key=lambda x: (x[1], x[0]))
        last_updated_display = most_common_date.strftime('%d %b %Y')
    else:
        last_updated_display = 'Unknown'
    return render_template('home.jinja2', tables=df_html, title="Overview", last_updated=last_updated_display)


@app.route('/load', methods=['GET', 'POST'])
@login_required
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
        except Exception:
            logger.debug(
                f'------------- An error {traceback.print_exc()} occurred ----------------')
            flash("An error occured, try again!", "error")
    return redirect(url_for('index'))


@app.route('/save', methods=['GET', 'POST'])
@login_required
def save_pf():
    pf_trades = current_user.get_trades()
    if pf_trades.empty:
        flash('No trades to export / save. Please add trades and try again', 'error')
        return render_template('home.jinja2', title='Overview')
    resp = make_response(pf_trades.to_csv(index=False))
    resp.headers.set("Content-Disposition",
                     "attachment", filename="trades.csv")
    return resp


@app.route('/add_trades', methods=['GET', 'POST'])
@login_required
def add_trades():
    if request.method == 'POST':
        trades_df = web_utils.resp_to_trades_df(request)
        current_user.add_trades(trades_df)
        flash('Trades added successfully', 'info')
        return render_template('home.jinja2', title='Overview')
    return render_template('add_trades.jinja2', title='Add Trades')


@app.route('/view_trades', methods=['GET', 'POST'])
@login_required
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


@app.route('/stock/<ticker>')
@login_required
def stock(ticker: str):
    # currency = request.args.get('currency')
    stock = Stocks.query.filter_by(ticker=ticker).first()
    name = stock.name
    as_at_date = get_date(request.args.get(
        'date'), request.form.get('time_offset'))
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

    return render_template('stock_dynamic.jinja2', title=f'Overview for {name}', stock_name=name, postition_df=position, divs=divs.to_html(), splits=splits.to_html(), trades=trades, ticker=ticker)


@app.route('/portfolio/monthly', methods=['GET', 'POST'])
@login_required
def monthly_pf():
    title = 'Monthly Portfolio Summary'
    if request.method == 'GET':
        return render_template('monthly.jinja2', title=title, view_mode='summary')
    elif request.method == 'POST':
        view_mode = request.form.get('view_mode', 'summary')
        detail = view_mode == 'detail'

        if (request.form.get('start_date') == '') or (request.form.get('end_date') == ''):
            flash('Please insert dates and submit query', 'info')
            return render_template('monthly.jinja2', title=title, view_mode=view_mode)
        start_date = get_date(request.form.get('start_date'), None)
        end_date = get_date(request.form.get('end_date'), None)
        exclude_crypto = request.form.get('exclude_crypto') == '1'
        exclude_loans = request.form.get('exclude_loans') == '1'

        logger.info(
            f'{start_date=}, {end_date=}, {exclude_loans=}, {exclude_crypto=}')

        monthly_summ = current_user.monthly_summary(
            start_date=start_date, end_date=end_date, exclude_crypto=exclude_crypto, exclude_loans=exclude_loans, detail=detail)
        monthly_summ.columns = pd.to_datetime(
            monthly_summ.columns).strftime('%b %y')
        if detail:
            monthly_summ['Total'] = monthly_summ.sum(axis=1)
        else:
            rows_to_sum = ['Dividends', 'Net Investment', 'Investment Returns']
            rows_available = [row for row in rows_to_sum if row in monthly_summ.index]
            if rows_available:
                monthly_summ.loc[rows_available,
                                 'Total'] = monthly_summ.loc[rows_available].sum(axis=1)

        export_df = monthly_summ.copy()
        sheet_name = 'Detail' if detail else 'Summary'
        if "action" in request.form and request.form["action"] == "Export to File":
            return exportxls(filename='monthly_summary.xlsx', export_index=True, df1=export_df, df1_name=sheet_name)
        else:
            if monthly_summ.empty:
                df_html = "<p><div class='alert alert-primary' role='alert'> No dividends or capital gains in period</div>"
            else:
                # convert columns to string and format dates, reset index and rename the first column to be blank
                monthly_summ.reset_index(inplace=True)
                index_label = 'Ticker' if detail else ' '
                monthly_summ.rename(columns={'index': index_label}, inplace=True)

                neg_cols = [col for col in monthly_summ.columns if col != index_label]
                if detail and 'Total' in monthly_summ[index_label].values:
                    total_row_index = monthly_summ.index[monthly_summ[index_label] == 'Total'].tolist()
                    rows_to_bold = total_row_index
                else:
                    rows_to_bold = [0, 3]
                df_html = web_utils.pandas_table_styler(
                    monthly_summ, neg_cols=neg_cols, left_align_cols=[index_label], ticker_links=detail, uuid='monthlysumm', rows_to_bold=rows_to_bold)
            return render_template('monthly.jinja2', tables=[df_html], title=title, view_mode=view_mode)


@app.route('/portfolio/nav', methods=['GET', 'POST'])
@login_required
def nav_pf():
    title = 'NAV Tracking'
    if request.method == 'GET':
        return render_template('nav.jinja2', title=title, nav_chart=None, nav_total_chart=None, nav_totals=None)
    elif request.method == 'POST':
        # Check if this is a force recalculation
        if request.form.get('action') == 'force_recalc':
            end_date = None
            end_date_input = request.form.get('end_date')
            if end_date_input:
                end_date = get_date(end_date_input, None)
            logger.info(f'Force recalculating NAV history (end_date={end_date})')
            updated_count = current_user.update_nav_in_trades(force_full_recalc=True, end_date=end_date)
            flash(f'Recalculated NAV for {updated_count} trades', 'info')
            return render_template('nav.jinja2', title=title, nav_chart=None, nav_total_chart=None, nav_totals=None)

        # Get date range
        if (request.form.get('start_date') == '') or (request.form.get('end_date') == ''):
            flash('Please insert dates and submit query', 'info')
            return render_template('nav.jinja2', title=title, nav_chart=None, nav_total_chart=None, nav_totals=None)

        start_date = get_date(request.form.get('start_date'), None)
        end_date = get_date(request.form.get('end_date'), None)

        logger.info(f'Generating NAV summary from {start_date} to {end_date}')

        # Get NAV monthly summary
        try:
            nav_summary_df, nav_totals = current_user.nav_monthly_summary(
                start_date,
                end_date,
                force_full_recalc=(request.form.get('action') == 'force_recalc'),
                recalc_start=start_date,
                recalc_end=end_date,
                perform_price_refresh=(request.form.get('action') == 'force_recalc')
            )
            nav_metrics = current_user.nav_performance_metrics(start_date=start_date, end_date=end_date)
        except Exception as e:
            logger.exception('Error generating NAV data')
            flash(f'Error generating NAV data: {str(e)}', 'error')
            return render_template('nav.jinja2', title=title, nav_chart=None, nav_total_chart=None, nav_totals=None)

        if nav_summary_df.empty:
            df_html = "<p><div class='alert alert-primary' role='alert'> No NAV data available for the selected period</div>"
            return render_template('nav.jinja2', tables=[df_html], title=title, metrics=nav_metrics, nav_chart=None, nav_total_chart=None, nav_totals=nav_totals)

        chart_df = nav_summary_df.sort_values('Month_End_Date')[['Month_End_Date', 'NAV_per_Unit', 'Portfolio_Value']].copy()
        nav_chart = None
        nav_total_chart = None
        if not chart_df.empty:
            fig_per_unit = go.Figure()
            fig_per_unit.add_trace(go.Scatter(
                x=chart_df['Month_End_Date'],
                y=chart_df['NAV_per_Unit'],
                mode='lines+markers',
                name='NAV per Unit'
            ))
            fig_per_unit.update_layout(
                template='plotly_white',
                margin=dict(t=40, r=20, b=40, l=60),
                height=400,
                title='NAV per Unit Over Time',
                xaxis_title='Month End',
                yaxis_title='NAV per Unit'
            )
            nav_chart = pio.to_html(fig_per_unit, include_plotlyjs='cdn', full_html=False)

            fig_total = go.Figure()
            fig_total.add_trace(go.Scatter(
                x=chart_df['Month_End_Date'],
                y=chart_df['Portfolio_Value'],
                mode='lines+markers',
                name='Total NAV'
            ))
            fig_total.update_layout(
                template='plotly_white',
                margin=dict(t=40, r=20, b=40, l=60),
                height=400,
                title='Total NAV Over Time',
                xaxis_title='Month End',
                yaxis_title='Portfolio Value'
            )
            nav_total_chart = pio.to_html(fig_total, include_plotlyjs='cdn', full_html=False)

        # Format the data for display
        nav_summary_df = nav_summary_df.copy()
        nav_summary_df['Month_End_Date'] = nav_summary_df['Month_End_Date'].dt.strftime('%Y-%m-%d')
        nav_summary_df = nav_summary_df.round({
            'NAV_per_Unit': 2,
            'Total_Units': 2,
            'Portfolio_Value': 2,
            'Monthly_Return_%': 2,
            'Inception_Return_%': 2,
            'Capital_Flow': 2,
            'Dividend_Flow': 2,
            'Net_Cash_Flow': 2
        })

        # Create HTML table
        neg_cols = ['Monthly_Return_%', 'Inception_Return_%', 'Capital_Flow', 'Net_Cash_Flow']
        df_html = web_utils.pandas_table_styler(
            nav_summary_df,
            neg_cols=neg_cols,
            left_align_cols=['Month_End_Date'],
            ticker_links=False,
            uuid='navsummary'
        )

        return render_template('nav.jinja2', tables=[df_html], title=title, metrics=nav_metrics, nav_chart=nav_chart, nav_total_chart=nav_total_chart, nav_totals=nav_totals)


@app.route('/update_stock_name', methods=['POST'])
@login_required
def update_stock_name():
    new_name = request.form.get('new_name')
    ticker = request.form.get('ticker')
    stock = Stocks.query.filter_by(ticker=ticker).first()
    logger.info(stock)
    stock.update_name(new_name)
    return jsonify({'status': 'success'}), 200


@app.route('/pfactions', methods=['GET', 'POST'])
@login_required
def pfactions():
    if "action" in request.form and request.form["action"] == "Export to CSV":
        return exportpf()
    else:
        return update_pf()


@app.route('/exportpf', methods=['GET', 'POST'])
@login_required
def exportpf():
    as_at_date = get_date(request.form.get(
        'date'), request.form.get('time_offset'))
    hide_zero = not (bool(request.form.get('hide_zero'))) or False
    # currency = request.form.get('currency') or 'AUD'

    df, _ = current_user.info_date(
        as_at_date=as_at_date, hide_zero_pos=hide_zero)
    resp = make_response(df.to_csv(index=False))
    resp.headers.set("Content-Disposition",
                     "attachment", filename="pf_position.csv")
    return resp


@app.route('/tax', methods=['GET', 'POST'])
@login_required
def tax():
    title = 'Tax Summary'
    if request.method == 'POST':
        if (request.form.get('start_date') == '') or (request.form.get('end_date') == ''):
            flash('Please insert dates and submit query', 'info')
            return render_template('tax.jinja2', title=title)
        if "action" in request.form and request.form["action"] == "Export to File":
            start_date = get_date(request.form.get('start_date'), None)
            end_date = get_date(request.form.get('end_date'), None)
            return exportpftax(title, start_date, end_date)
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


def exportpftax(title: str, start_date: str, end_date: str):
    df, trades_df = get_tax_df(title, start_date, end_date)
    return exportxls(filename='tax_trades.xlsx', export_index=False, df1=df, df1_name='Summary', df2=trades_df, df2_name='Trades')


def exportxls(filename: str, export_index: bool, df1: pd.DataFrame, df1_name: str, df2: pd.DataFrame = None, df2_name: str = None) -> Response:
    """
    Export excel file with each df as a sheet as a Response object

    Args:
        filename (str): name of the file
        export_index (bool): True if index to be exported, false otherwise
        df1 (pd.DataFrame): dataframe 1 to export into sheet
        df1_name (str): sheet name for dataframe 1
        df2 (pd.DataFrame, optional): dataframe 2 to export into sheet. Defaults to None.
        df2_name (str, optional): sheet name for dataframe 1. Defaults to None.

    Returns:
        Response: Excel sheet in Response object
    """

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    df1.to_excel(writer, sheet_name=df1_name, index=export_index)
    if df2 is not None:
        df2.to_excel(writer, sheet_name=df2_name, index=export_index)

    # Close the Pandas Excel writer and output the Excel file.
    writer.close()
    output.seek(0)

    resp = make_response(output.getvalue())
    resp.headers.set("Content-Disposition",
                     "attachment", filename=filename)
    return resp


def taxoutput(title: str):
    start_date = get_date(request.form.get(
        'start_date'), None)
    end_date = get_date(request.form.get(
        'end_date'), None)
    df, trades_df = get_tax_df(title, start_date, end_date)
    if df.empty:
        df_html = "<p><div class='alert alert-primary' role='alert'> No dividends or capital gains in period</div>"
        trades_df_html = "<p>"
    else:
        df['Date'] = df['Date'].dt.strftime('%d-%m-%y')
        df_html = web_utils.pandas_table_styler(
            df, neg_cols=['RlGain'], left_align_cols=['Ticker', 'Name'], ticker_links=False, uuid='taxsummary')
        df_html = web_utils.add_footer(df_html)

        trades_df['Date'] = trades_df['Date'].dt.strftime('%d-%m-%y')
        trades_df_html = web_utils.pandas_table_styler(
            trades_df, neg_cols=['CashFlow'], left_align_cols=['Ticker'], ticker_links=False, uuid='taxtrades')
        # trades_df_html = web_utils.add_footer(df_html)
    logger.debug(df_html)
    logger.debug(trades_df_html)
    return render_template('tax.jinja2', tables=[df_html, trades_df_html], title=title)


def get_tax_df(title: str, start_date: str, end_date: str):
    logger.info(f'{start_date=}, {end_date=}')
    hide_zero = False
    # currency = 'AUD'

    pf_trades = current_user.get_trades()
    if pf_trades.empty:
        flash('Portfolio is empty. Please add some trades', 'error')
        return render_template('home.jinja2', title=title)

    df, trades = current_user.info_date(
        start_date=start_date, as_at_date=end_date, hide_zero_pos=hide_zero, limit_divs_by_date=True)
    trades = trades[['Date', 'Ticker', 'Quantity',
                     'Price', 'Fees', 'CF', 'Direction']]

    # restrict output to stocks where there was a tax event in the period (i.e. dividends or capital gains)
    df = df[(df['RlGain'] != 0) | (df['Dividends'] != 0)].copy()
    df = df[['Ticker', 'Name', 'CurrVal', 'Dividends', 'RlGain', 'Date', 'Type']]

    logger.info(df)

    trades_df = pd.DataFrame()
    for index, row in df.iterrows():
        if row['Ticker'] != 'Total':
            ticker_trades = trades[trades['Ticker'] == row['Ticker']]
            # logger.info(ticker_trades)
            if row['Dividends'] != 0:
                ticker_trades_divs = ticker_trades[(ticker_trades['Date'] >= start_date) & (
                    ticker_trades['Date'] <= end_date) & (ticker_trades['Direction'] == 'Div')]
                trades_df = pd.concat(
                    [trades_df, ticker_trades_divs])
            if row['RlGain'] != 0:
                ticker_trades_gains = ticker_trades[(ticker_trades['Date'] <= end_date) & (
                    ticker_trades['Direction'].isin(['Buy', 'Sell']))]
                trades_df = pd.concat(
                    [trades_df, ticker_trades_gains])
    logger.info(trades_df)
    trades_df.rename(columns={'CF': 'CashFlow'}, inplace=True)

    return df, trades_df


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
        as_at_date = pd.Timestamp.now(
            tz=timezone.utc).tz_convert(tz).tz_localize(None)
        logger.info(f'Localised datetime is: {as_at_date}')
    else:
        as_at_date = datetime.strptime(date, "%Y-%m-%d")
    return as_at_date


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
