import logging
import os
import sys

from flask import Flask
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

from config import Config

# set up logging — create logs dir if it doesn't exist
log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(filename=os.path.join(log_dir, 'logs.log'))
stdout_handler = logging.StreamHandler(sys.stdout)
handlers = [file_handler, stdout_handler]
logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S', level=logging.INFO, handlers=handlers)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
# add engine_options={'echo': True} to see SQL statements / queries
db = SQLAlchemy(app)
migrate = Migrate(app, db)
login = LoginManager(app)
login.login_view = 'login'

# isort:off
from app import routes, models
from app.api_routes import api_bp
# isort:on

app.register_blueprint(api_bp)

# app.run()
