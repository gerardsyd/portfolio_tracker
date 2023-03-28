from app import app, db
from app.models import User, Trades, Stocks


@app.shell_context_processor
def make_shell_context():
    return {'db': db, 'User': User, 'Trades': Trades, 'Stocks': Stocks}
