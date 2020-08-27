from datetime import datetime
import logging
from os import path

import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from flask.helpers import make_response

from portfolio import Portfolio

logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S',
                    level=logging.INFO, filename=r'logs\logs.log')
logger = logging.getLogger('pt_logger')

app = Flask(__name__)
TRADES_FILE = 'data/pf_trades.pkl'
DATA_FILE = 'data/pf_data.pkl'
NAMES_FILE = 'data/pf_names.pkl'
PF_FORMAT_DICT = {'Quantity': "{:,.2f}", '%LastChange': '{:,.2%}', 'IRR': '{:,.2%}', '%UnRlGain': '{:,.2%}', '%PF': '{:,.2%}', '%CostPF': '{:,.2%}',
                  'LastPrice': '{:,.2f}', 'CurrVal': '{:,.2f}', 'AvgCost': '{:,.2f}', 'Cost': '{:,.2f}', 'RlGain': '{:,.2f}', 'UnRlGain': '{:,.2f}',
                  'Dividends': '{:,.2f}', 'TotalGain': '{:,.2f}', '$LastChange': '{:,.2f}'}
TD_TYPE_DICT = {'Date': 'date', 'Ticker': 'text', 'Quantity': 'number',
                'Price': 'number', 'Fees': 'number', 'Direction': 'text'}


@app.route('/')
def home():
    return updatepf()
    # return render_template('home.jinja2', title="Portfolio Tracker: Home")


@app.route('/update', methods=['POST'])
def updatepf():
    # get as at date from drop down. If left blank, set none (defaults to today per portfolio info function)
    as_at_date = None if request.form.get(
        'up_date') == '' else request.form.get('up_date')
    hide_zero = bool(request.form.get('hide_zero')) or True
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
    df_html = (df.style
               .applymap(neg_red, subset=['%LastChange', '$LastChange', '%UnRlGain', 'RlGain', 'UnRlGain', 'TotalGain'])
               .format(PF_FORMAT_DICT, na_rep="--")
               .set_properties(**{'text-align': 'left'}, subset=['Ticker', 'Name'])
               .set_properties(**{'font-weight': 'bold'}, subset=df.index[-1])
               .hide_index()
               .set_uuid('portfolio')
               .set_table_attributes('class="hover stripe row-border order-column display compact" style="width:100%"')
               .render()
               )
    df_html = add_footer(df_html)
    logger.info(f'render HTML took {(datetime.now()-start)} to run')
    start = datetime.now()
    pf.trades_df.to_pickle(TRADES_FILE)
    logger.info(f'trades_df.to_pickle took {(datetime.now()-start)} to run')
    return render_template('home.jinja2', tables=df_html, title="Portfolio Tracker: Portfolio")


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
        return render_template('home.jinja2', message=message, title="Portfolio Tracker: Home")


@app.route('/save', methods=['GET', 'POST'])
def savepf():
    if path.isfile(TRADES_FILE):
        logger.info(f'{TRADES_FILE} exists, loading')
        pf_trades = pd.read_pickle(TRADES_FILE)
    else:
        return render_template('home.jinja2', message='File not found', title="Portfolio Tracker: Home")
    resp = make_response(pf_trades.to_csv(index=False))
    resp.headers.set("Content-Disposition",
                     "attachment", filename="trades.csv")
    return resp


@app.route('/add_trades', methods=['GET', 'POST'])
def add_trades():
    if request.method == 'POST':
        trades_df = resp_to_trades_df(request)
        if path.isfile(TRADES_FILE):
            logger.info(f'{TRADES_FILE} exists, loading')
            pf_trades = pd.read_pickle(TRADES_FILE)
            pf = Portfolio(pf_trades, DATA_FILE)
            pf.add_trades(trades_df)
        else:
            pf = Portfolio(trades_df, DATA_FILE)
        pf.trades_df.to_pickle(TRADES_FILE)
        return render_template('home.jinja2', message='Trades added successfully')
    return render_template('add_trades.jinja2')


@app.route('/view_trades', methods=['GET', 'POST'])
def view_trades():
    if request.method == 'GET':
        if path.isfile(TRADES_FILE):
            logger.info(f'{TRADES_FILE} exists, loading')
            df = pd.read_pickle(TRADES_FILE)
        else:
            return render_template('home.jinja2', message='No portfolio trades founds. Please load file or add trades', title="Portfolio Tracker: Home")

        # format date to allow render in date input field
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')

        # add delete trade column
        df['Delete?'] = 0

        df_html = (df.style
                   .format({c: html_input(c) for c in df.columns})
                   .set_uuid('view_trades')
                   .hide_index()
                   .render()
                   )
        return render_template('view_trades.jinja2', tables=df_html, message=request.args.get('message'))
    else:
        # gets updated trade data
        trades_df = resp_to_trades_df(request)

        # checks and deletes relevant rows
        trades_df['Delete?'] = request.form.getlist('Delete?')
        trades_df = trades_df[trades_df['Delete?'] == '0']
        trades_df.drop(['Delete?'], axis=1, inplace=True)
        trades_df.to_pickle(TRADES_FILE)
        return redirect(url_for('view_trades', message='Successfully updated trades'))


@app.route('/stock/<ticker>')
def stock(ticker):
    return render_template('stock_dynamic.jinja2', stock_name=ticker)


def add_footer(html: str):
    """
    Adds tfoot tags to total row to allow sorting

    Args:
        html (str): table html string

    Returns:
        [type]: table html string
    """

    loc = html.rfind('<tr>')
    html = html[:loc] + '<tfoot>' + html[loc:]
    loc = html.rfind('</tr>')
    html = html[:loc] + '</tfoot>' + html[loc:]
    return html


def resp_to_trades_df(req: request):
    """
    Converts response from add trades / update trades functions into dataframe

    Args:
        req (request): html request from view / update trades function

    Returns:
        pd.Dataframe: Dataframe in the form of Portfolio.trades_df

    """

    df = pd.DataFrame(columns=Portfolio.TD_COLUMNS)
    data = req.form.listvalues()
    for col in df.columns:
        df[col] = next(data)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    return df


def html_input(col: str):
    """
    Adds html input tags for given column

    Args:
        col (str): column in pandas table

    Returns:
        string: f-string with relevant html tags
    """

    if col == 'Delete?':
        return f'<input type="hidden" name="{col}" value="0"><input type="checkbox" onclick="this.previousSibling.value=1-this.previousSibling.value">'
    else:
        return f'<input type="{TD_TYPE_DICT[col]}" name="{col}" value="{{}}" step=".01"/>'


def neg_red(val: float):
    """
    Takes a value and returns a string with
    the css property color: red for negative
    strings, green otherwise.
    """
    try:
        color = 'red' if val < 0 else 'limegreen'
    except TypeError:
        # catch Na / NaN / strings / empty fields
        color = 'black'
    return 'color: %s' % color


if __name__ == '__main__':
    app.run(debug=False, use_reloader=True)
