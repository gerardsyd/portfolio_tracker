import os

basedir = os.path.abspath(os.path.dirname(__file__))


class Config(object):
    SECRET_KEY = (os.environ.get('SECRET_KEY'))
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'data/app.db')
    CHROMEDRIVER_PATH = os.environ.get('CHROMEDRIVER_PATH')
    CHD_LOG_PATH = os.environ.get('CHD_LOG_PATH')
    SQLALCHEMY_POOL_RECYCLE = 100
    SQLALCHEMY_POOL_TIMEOUT = 600
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # MAX_CONTENT_LENGTH = 1024 * 1024
    UPLOAD_EXTENSIONS = ['.csv']
    # EXPLAIN_TEMPLATE_LOADING = True
