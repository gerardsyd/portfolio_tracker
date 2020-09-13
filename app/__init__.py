import logging

from flask import Flask
from config import Config


logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%d-%m-%Y:%H:%M:%S',
                    level=logging.INFO, filename=r'logs\logs.log')
app = Flask(__name__)
app.config.from_object(Config)

from app import routes

# app.run()
