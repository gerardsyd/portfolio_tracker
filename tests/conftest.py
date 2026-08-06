"""Pytest configuration for PFTrackr tests."""
import os
import tempfile

import pytest


@pytest.fixture(scope="session")
def app():
    """Create a Flask app instance configured for testing."""
    from app import app as flask_app
    flask_app.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "API_TOKEN": "test-token",
        "WTF_CSRF_ENABLED": False,
        "SECRET_KEY": "test-secret",
        "REGISTRATION_ENABLED": False,
    })
    yield flask_app


@pytest.fixture(scope="function")
def client(app):
    """Test client for the application."""
    with app.test_client() as client:
        yield client


@pytest.fixture(scope="function")
def db(app):
    """Set up and tear down the in-memory database."""
    from app import db as _db
    with app.app_context():
        _db.create_all()
        yield _db
        _db.session.remove()
        _db.drop_all()
