from typing import List
import re

from flask import request
import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio

from app.portfolio import Portfolio

TD_TYPE_DICT = {'Date': 'date', 'Ticker': 'text', 'Quantity': 'number',
                'Price': 'number', 'Fees': 'number', 'Direction': 'text'}
PF_FORMAT_DICT = {'Quantity': "{:,.2f}", '%LastChange': '{:,.2%}', 'IRR': '{:,.2%}', '%UnRlGain': '{:,.2%}', '%PF': '{:,.2%}', '%CostPF': '{:,.2%}',
                  'LastPrice': '{:,.2f}', 'CurrVal': '{:,.2f}', 'AvgCost': '{:,.2f}', 'Cost': '{:,.2f}', 'RlGain': '{:,.2f}', 'UnRlGain': '{:,.2f}',
                  'Dividends': '{:,.2f}', 'TotalGain': '{:,.2f}', '$LastChange': '{:,.2f}'}
pio.templates.default = "plotly_white"

logger = logging.getLogger('pt_logger.web_utils')


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
    df.drop(['Pf_price', 'Pf_shares'], inplace=True, axis=1)
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


def stock_link(value: str):
    return f'<a href="/stock/{value}">{value}</a>'


def update_links(html: str, currency: str, date: str):
    pattern = '(\/stock\/[\w\d.]*)'
    link = f'?currency={currency}&date={date}'
    html = re.sub(pattern, r'\1{}'.format(link), html)
    return html


def pandas_table_styler(df: pd.DataFrame, neg_cols: List, left_align_cols: List, ticker_links: bool, uuid: str):
    df_html = (df.style
               .applymap(neg_red, subset=neg_cols)
               .format(PF_FORMAT_DICT, na_rep="--")
               .set_properties(**{'text-align': 'left'}, subset=left_align_cols)
               .set_properties(**{'font-weight': 'bold'}, subset=df.index[-1])
               .hide(axis="index")
               .set_uuid(uuid)
               .set_table_attributes('class="hover stripe row-border order-column display compact"')
               )
    if ticker_links:
        df_html.format(
            stock_link, subset=pd.IndexSlice[df.index[:-1], 'Ticker'])
    return df_html.to_html()


def create_fig(df: pd.DataFrame, x: str, y: List, hover: List, height: int):
    position_fig = px.line(data_frame=df, x=x,
                           y=y, hover_data=hover, height=height)
    position_fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="Since Acquisition")
            ])
        )
    )
    return position_fig


def create_fig2(df: pd.DataFrame, height: int, hover: List):

    layout = go.Layout(
        title='Historial Position',
        xaxis_title='Date', height=height
    )
    hoverdata = np.dstack([df[col] for col in hover])
    print(hoverdata)
    trace_CurrVal = go.Scatter(
        x=df['Date'], y=df['CurrVal'], name="Current Value", customdata=hoverdata, hovertemplate=['%{hover[i]}:%{hoverdata[i]:.2f} <br>' for i in range(0, len(hover))])
    trace_TotalGain = go.Scatter(
        x=df['Date'], y=df['TotalGain'], name="Total Gain")
    position_fig = go.Figure(
        data=[trace_CurrVal, trace_TotalGain], layout=layout)
    position_fig.update_layout(
        hovermode='x unified',
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="Since Acquisition")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )
    position_fig.update_

    return position_fig
