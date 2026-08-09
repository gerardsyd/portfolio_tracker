from datetime import datetime
from decimal import Decimal

import pytest

from app.models import PortfolioMonthlyNav, StockPrices, Stocks, Trades, User


def _add_price(ticker, price_date, close):
    return StockPrices(
        ticker=ticker,
        date=price_date,
        open=Decimal(str(close)),
        high=Decimal(str(close)),
        low=Decimal(str(close)),
        close=Decimal(str(close)),
        adjclose=Decimal(str(close)),
        volume=0,
        dividends=0,
        splits=0,
    )


def test_portfolio_value_on_uses_stored_prices_without_info_date(db, monkeypatch):
    user = User(username='nav-user', email='nav@example.com', default_currency='AUD')
    db.session.add_all([
        user,
        Stocks(ticker='ABC.AX', name='ABC', currency='AUD'),
        Stocks(ticker='USD.AUD=X.FX', name='USD/AUD', currency='AUD'),
    ])
    db.session.flush()
    db.session.add_all([
        Trades(user_id=user.id, ticker='ABC.AX', date=datetime(2026, 1, 5),
               quantity=Decimal('10'), price=Decimal('5'), fees=0, direction='Buy'),
        _add_price('ABC.AX', datetime(2026, 1, 30), 7),
        _add_price('ABC.AX', datetime(2026, 2, 27), 8),
    ])
    db.session.commit()

    monkeypatch.setattr(user, 'info_date', lambda **_: pytest.fail('info_date must not be used'))

    assert user._portfolio_value_on(datetime(2026, 2, 28)) == pytest.approx(80.0)


def test_monthly_nav_revalues_per_unit_when_units_carry_forward(db, monkeypatch):
    user = User(username='monthly-nav-user', email='monthly-nav@example.com', default_currency='AUD')
    db.session.add_all([user, Stocks(ticker='ABC.AX', name='ABC', currency='AUD')])
    db.session.flush()
    db.session.add_all([
        Trades(user_id=user.id, ticker='ABC.AX', date=datetime(2026, 1, 5),
               quantity=Decimal('10'), price=Decimal('5'), fees=0, direction='Buy',
               pf_price=Decimal('5'), pf_shares=Decimal('10')),
        _add_price('ABC.AX', datetime(2026, 1, 30), 7),
        _add_price('ABC.AX', datetime(2026, 2, 27), 8),
    ])
    db.session.commit()

    monkeypatch.setattr(user, 'update_nav_in_trades', lambda **_: 0)

    summary, _ = user.nav_monthly_summary(
        start_date=datetime(2026, 1, 1), end_date=datetime(2026, 2, 28)
    )

    february = summary.loc[summary['Month_End_Date'] == datetime(2026, 2, 28)].iloc[0]
    assert february['Total_Units'] == pytest.approx(10.0)
    assert february['Portfolio_Value'] == pytest.approx(80.0)
    assert february['NAV_per_Unit'] == pytest.approx(8.0)
    cached_february = PortfolioMonthlyNav.query.filter_by(
        user_id=user.id,
        month_end=datetime(2026, 2, 28).date()
    ).one()
    assert not cached_february.needs_refresh


def test_portfolio_value_on_uses_latest_stored_fx_rate(db):
    user = User(username='fx-nav-user', email='fx-nav@example.com', default_currency='AUD')
    db.session.add_all([
        user,
        Stocks(ticker='USSTOCK', name='US Stock', currency='USD'),
        Stocks(ticker='USDAUD=X.FX', name='USD/AUD', currency='AUD'),
    ])
    db.session.flush()
    db.session.add_all([
        Trades(user_id=user.id, ticker='USSTOCK', date=datetime(2026, 1, 5),
               quantity=Decimal('10'), price=Decimal('5'), fees=0, direction='Buy'),
        _add_price('USSTOCK', datetime(2026, 1, 30), 5),
        _add_price('USDAUD=X.FX', datetime(2026, 1, 29), 1.50),
        _add_price('USDAUD=X.FX', datetime(2026, 1, 30), 1.60),
    ])
    db.session.commit()

    assert user._portfolio_value_on(datetime(2026, 1, 31)) == pytest.approx(80.0)


def test_portfolio_value_on_falls_back_to_last_transaction_price(db):
    user = User(username='fallback-nav-user', email='fallback-nav@example.com', default_currency='AUD')
    db.session.add_all([user, Stocks(ticker='UNPRICED', name='Unpriced', currency='AUD')])
    db.session.flush()
    db.session.add(
        Trades(user_id=user.id, ticker='UNPRICED', date=datetime(2026, 1, 5),
               quantity=Decimal('10'), price=Decimal('7.50'), fees=0, direction='Buy')
    )
    db.session.commit()

    assert user._portfolio_value_on(datetime(2026, 1, 31)) == pytest.approx(75.0)
