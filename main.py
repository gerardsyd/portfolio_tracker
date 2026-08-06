"""Entrypoint — also importable as 'main:app' for gunicorn."""
from app import app

if __name__ == "__main__":
    app.run(host='0.0.0.0')
