"""Pytest configuration for PFTrackr tests."""
import os
import tempfile

import pytest

# Must be set BEFORE importing the app so the SQLAlchemy engine binds to a
# writable temp DB rather than the default data/app.db
_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_db_file.name}")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("API_TOKEN", "test-token")
os.environ.setdefault("REGISTRATION_ENABLED", "False")


@pytest.fixture(scope="session")
def app():
    """Create a Flask app instance configured for testing."""
    from app import app as flask_app
    flask_app.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{_db_file.name}",
        "API_TOKEN": "test-token",
        "WTF_CSRF_ENABLED": False,
        "SECRET_KEY": "test-secret",
        "REGISTRATION_ENABLED": False,
    })
    yield flask_app
    # Clean up temp db after session
    try:
        os.unlink(_db_file.name)
    except OSError:
        pass


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
