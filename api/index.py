"""
Vercel Serverless Entrypoint for Polar Navigator AI
Exposes the FastAPI application instance for Vercel Python Runtime.
"""

import sys
import os
from pathlib import Path

# Ensure project root is in Python sys.path so 'app' packages import cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Set serverless environment indicator
os.environ.setdefault("VERCEL_ENVIRONMENT", "true")
os.environ.setdefault("DATA_MODE", "demo")

# Import the configured FastAPI application from the core application layer
from app.main import app
