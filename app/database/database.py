"""
Database connection and session management for Polar Navigator AI.
Supports PostgreSQL/PostGIS in production and resilient SQLite/In-Memory fallback on Vercel.
"""

import os
import tempfile
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv
from app.database.models import Base

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

is_serverless = bool(os.getenv("VERCEL") or os.getenv("VERCEL_ENVIRONMENT") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))

if not DATABASE_URL:
    if is_serverless:
        # Use temp directory or in-memory SQLite on serverless platforms
        temp_db = Path(tempfile.gettempdir()) / "polar_navigator.db"
        DATABASE_URL = f"sqlite:///{temp_db.as_posix()}"
    else:
        DATABASE_URL = "sqlite:///./polar_navigator.db"

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

try:
    engine = create_engine(
        DATABASE_URL,
        connect_args=connect_args,
        echo=False
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
except Exception:
    # Fail-safe in-memory fallback
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initializes all database tables with error resilience."""
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        # Fail safe on read-only environments
        pass


def is_db_connected() -> bool:
    """Checks whether the database connection is live."""
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def get_db():
    """Dependency for getting DB session in FastAPI."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
