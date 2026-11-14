"""Database connection and table management using SQLite."""
import os
import shutil
import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Generator
from backend.app.core.config import DATA_DIR, DATABASE_URL, PROJECT_ROOT
from backend.app.database.migrations import run_migrations

logger = logging.getLogger("apex_analytics")


def _resolve_sqlite_path(url: str) -> Path:
    """Resolve sqlite database path from connection URL or fallback to DATA_DIR."""
    # When running on Vercel or read-only serverless environment, copy DB to /tmp
    if os.getenv("VERCEL"):
        tmp_db = Path("/tmp/student_performance.db")
        if not tmp_db.exists():
            bundled_db = DATA_DIR / "student_performance.db"
            if not bundled_db.exists():
                bundled_db = PROJECT_ROOT / "data" / "student_performance.db"
            if bundled_db.exists():
                try:
                    tmp_db.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(bundled_db, tmp_db)
                except Exception as e:
                    logger.warning(f"Could not copy bundled DB to /tmp: {e}")
        return tmp_db

    prefix = "sqlite:///"
    if url.startswith(prefix):
        raw_path = url[len(prefix):]
        p = Path(raw_path)
        if p.is_absolute():
            return p
        return (PROJECT_ROOT / p).resolve()
    return DATA_DIR / "student_performance.db"


DB_PATH = _resolve_sqlite_path(DATABASE_URL)


def init_db() -> None:
    """Initialize SQLite database tables and apply pending schema migrations."""
    try:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        run_migrations(DB_PATH)
    except Exception as e:
        logger.warning(f"Database migration note: {e}")


@contextmanager
def get_db_connection() -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite database connection with row factory, WAL, and foreign keys enabled."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.execute("PRAGMA busy_timeout = 30000;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

