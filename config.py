import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    SECRET_KEY = (os.environ.get('SECRET_KEY'))
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'data/app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # EXPLAIN_TEMPLATE_LOADING = True
