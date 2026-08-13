"""Shared pytest setup for a deterministic isolated database."""

import os


# This must run before any test module imports app.db.database. Individual
# modules may still override the value, but all tests start on the test DB.
os.environ.setdefault("DB_PATH", "test_gemi_memory.db")

from app.db import database

database.init_db()
