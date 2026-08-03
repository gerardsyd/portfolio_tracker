import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from flask import Blueprint, jsonify, request
from sqlalchemy import func

from app import db
from app.models import User, Trades

logger = logging.getLogger('pt_logger.api')

api_bp = Blueprint('api', __name__, url_prefix='/api')


def require_api_token(f):
    """Decorator: check Bearer token against config."""
    from flask import current_app
    from functools import wraps

    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.headers.get('Authorization', '')
        token = current_app.config.get('API_TOKEN', '')
        if not auth.startswith('Bearer ') or auth.split(' ', 1)[1] != token:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return decorated


def get_user() -> User:
    """Return the first (and only) user — single-user deployment."""
    return User.query.first()


def get_date(date_str: str, time_offset: str = None) -> datetime:
    """Parse date string or return now (AEST-tz-aware)."""
    if date_str:
        return datetime.strptime(date_str, '%Y-%m-%d')
    offset = int(time_offset) if time_offset else 0
    return datetime.utcnow() + timedelta(hours=offset)


# ──────────────────────────────────────────────
#  Existing endpoints (preserved from running container)
# ──────────────────────────────────────────────

@api_bp.route('/portfolio/summary', methods=['GET'])
@require_api_token
def api_summary():
    """Current portfolio summary with holdings, values, gains/losses."""
    user = get_user()
    if not user:
        return jsonify({'error': 'No user found'}), 404

    as_at_date = get_date(request.args.get('date'))
    hide_zero = request.args.get('hide_zero', 'false').lower() == 'true'

    pf_trades = user.get_trades()
    if pf_trades.empty:
        return jsonify({'as_at_date': str(as_at_date.date()), 'summary': []})

    df, _ = user.info_date(as_at_date=as_at_date, hide_zero_pos=hide_zero)
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df = df.where(pd.notna(df), None)

    # Convert IRR to float or None
    if 'IRR' in df.columns:
        df['IRR'] = df['IRR'].apply(lambda x: float(x) if x is not None and x != '' else '')

    summary = df.to_dict(orient='records')
    return jsonify({
        'as_at_date': str(as_at_date.date()),
        'summary': summary
    })


@api_bp.route('/portfolio/update', methods=['POST'])
@require_api_token
def api_update():
    """Update portfolio prices and calculations for a date."""
    user = get_user()
    if not user:
        return jsonify({'error': 'No user found'}), 404

    data = request.get_json(silent=True) or {}
    as_at_date = get_date(data.get('date'))

    pf_trades = user.get_trades()
    if pf_trades.empty:
        return jsonify({'error': 'Portfolio is empty'}), 400

    user.update_prices(as_at_date=as_at_date)
    return jsonify({'status': 'ok', 'as_at_date': str(as_at_date.date())})


@api_bp.route('/portfolio/trades', methods=['GET'])
@require_api_token
def api_trades():
    """Trade history. Supports start_date, end_date query params."""
    user = get_user()
    if not user:
        return jsonify({'error': 'No user found'}), 404

    df = user.get_trades()
    if df.empty:
        return jsonify([])

    start = request.args.get('start_date')
    end = request.args.get('end_date')

    if start:
        df = df[df['Date'] >= pd.to_datetime(start)]
    if end:
        df = df[df['Date'] <= pd.to_datetime(end)]

    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    df = df.where(pd.notna(df), None)
    return jsonify(df.to_dict(orient='records'))


@api_bp.route('/portfolio/add_trades', methods=['POST'])
@require_api_token
def api_add_trades():
    """Add one or more trades to the portfolio.

    Expects JSON body with a 'trades' array, each trade having:
      date, ticker, quantity, price, fees, direction

    Or a single trade object (not wrapped in 'trades').
    """
    user = get_user()
    if not user:
        return jsonify({'error': 'No user found'}), 404

    body = request.get_json(silent=True)
    if not body:
        return jsonify({'error': 'Request body must be JSON'}), 400

    # Accept either {trades: [...]} or a single trade object
    if 'trades' in body:
        trades_list = body['trades']
    else:
        trades_list = [body]

    if not trades_list:
        return jsonify({'error': 'No trades provided'}), 400

    required_cols = ['date', 'ticker', 'quantity', 'price', 'fees', 'direction']
    rows = []
    for i, trade in enumerate(trades_list):
        missing = [c for c in required_cols if c not in trade]
        if missing:
            return jsonify({'error': f'Trade {i} missing fields: {", ".join(missing)}'}), 400

        # Normalise to DB column names
        rows.append({
            'Date': trade['date'],
            'Ticker': trade['ticker'].upper(),
            'Quantity': float(trade['quantity']),
            'Price': float(trade['price']),
            'Fees': float(trade['fees']),
            'Direction': trade['direction'].capitalize(),
        })

    df = pd.DataFrame(rows)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

    # Validate direction
    valid_dirs = ['Buy', 'Sell', 'Div']
    invalid = df[~df['Direction'].isin(valid_dirs)]
    if not invalid.empty:
        bad = invalid['Direction'].unique().tolist()
        return jsonify({'error': f'Invalid direction(s): {bad}. Must be: Buy, Sell, or Div'}), 400

    user.add_trades(df)

    count = len(df)
    logger.info(f'API: added {count} trade(s) for user {user.id}')
    return jsonify({'status': 'ok', 'trades_added': count}), 201


@api_bp.route('/portfolio/monthly', methods=['POST'])
@require_api_token
def api_monthly():
    """Monthly portfolio performance for a period."""
    user = get_user()
    if not user:
        return jsonify({'error': 'No user found'}), 404

    data = request.get_json(silent=True) or {}
    start_date = get_date(data.get('start_date'))
    end_date = get_date(data.get('end_date'))
    exclude_crypto = data.get('exclude_crypto', False)
    exclude_loans = data.get('exclude_loans', False)

    monthly_summ = user.monthly_summary(
        start_date=start_date,
        end_date=end_date,
        exclude_crypto=exclude_crypto,
        exclude_loans=exclude_loans,
        detail=False,
    )

    if monthly_summ.empty:
        return jsonify([])

    result = []
    for col in monthly_summ.columns:
        result.append({
            'period': str(col),
            'value': monthly_summ[col].to_dict()
        })
    return jsonify(result)


@api_bp.route('/portfolio/tax', methods=['POST'])
@require_api_token
def api_tax():
    """Tax-related trade data for a period."""
    user = get_user()
    if not user:
        return jsonify({'error': 'No user found'}), 404

    data = request.get_json(silent=True) or {}
    start_date = get_date(data.get('start_date'))
    end_date = get_date(data.get('end_date'))

    hide_zero = False
    pf_trades = user.get_trades()
    if pf_trades.empty:
        return jsonify({'summary': [], 'trades': []})

    df, trades = user.info_date(
        start_date=start_date, as_at_date=end_date, hide_zero_pos=hide_zero, limit_divs_by_date=True)
    trades = trades[['Date', 'Ticker', 'Quantity', 'Price', 'Fees', 'CF', 'Direction']]

    # restrict to stocks with tax events in period
    df = df[(df['RlGain'] != 0) | (df['Dividends'] != 0)].copy()
    df = df[['Ticker', 'Name', 'CurrVal', 'Dividends', 'RlGain', 'Date', 'Type']]

    trades_out = pd.DataFrame()
    for _, row in df.iterrows():
        if row['Ticker'] != 'Total':
            ticker_trades = trades[trades['Ticker'] == row['Ticker']]
            if row['Dividends'] != 0:
                ticker_trades_divs = ticker_trades[
                    (ticker_trades['Date'] >= start_date) &
                    (ticker_trades['Date'] <= end_date) &
                    (ticker_trades['Direction'] == 'Div')
                ]
                trades_out = pd.concat([trades_out, ticker_trades_divs])
            if row['RlGain'] != 0:
                ticker_trades_gains = ticker_trades[
                    (ticker_trades['Date'] <= end_date) &
                    (ticker_trades['Direction'].isin(['Buy', 'Sell']))
                ]
                trades_out = pd.concat([trades_out, ticker_trades_gains])

    trades_out.rename(columns={'CF': 'CashFlow'}, inplace=True)

    return jsonify({
        'summary': df.to_dict(orient='records') if not df.empty else [],
        'trades': trades_out.to_dict(orient='records') if not trades_out.empty else [],
    })


@api_bp.route('/portfolio/save', methods=['GET'])
@require_api_token
def api_save():
    """Export current portfolio trades as CSV (via API)."""
    user = get_user()
    if not user:
        return jsonify({'error': 'No user found'}), 404

    df = user.get_trades()
    if df.empty:
        return jsonify({'error': 'No trades found'}), 404

    return jsonify(df.to_dict(orient='records'))