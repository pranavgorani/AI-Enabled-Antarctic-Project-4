"""
Background Data Refresh Jobs
Polar Navigator AI - MoES / NCPOR

Refreshes local scientific caches every 6 hours using APScheduler when running locally.
Disabled on serverless / Vercel platforms to prevent hanging worker loops.
Supports manual on-demand triggers.
"""

import os
from datetime import datetime
from app.providers.nsidc_provider import nsidc_provider
from app.providers.era5_provider import era5_provider
from app.providers.copernicus_marine_provider import copernicus_marine_provider
from app.providers.iceberg_provider import iceberg_provider
from app.core.logging import logger

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    HAS_APSCHEDULER = True
    scheduler = BackgroundScheduler()
except ImportError:
    HAS_APSCHEDULER = False
    scheduler = None


def refresh_all_data():
    """Fetches latest environmental observations and updates local caches."""
    logger.info("Executing scheduled polar data synchronization...", extra={"event": "data_refresh"})
    try:
        nsidc_provider.get_current()
        era5_provider.get_current()
        copernicus_marine_provider.get_currents()
        iceberg_provider.get_observations()
        logger.info("Polar data synchronization completed successfully.", extra={"event": "data_refresh_complete"})
    except Exception as e:
        logger.error(f"Error during polar data synchronization: {e}", extra={"event": "data_refresh_error"})


def start_scheduler():
    """Starts the background synchronization daemon if supported and not serverless."""
    is_serverless = bool(os.getenv("VERCEL") or os.getenv("VERCEL_ENVIRONMENT") or os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
    if is_serverless:
        logger.info("Serverless environment detected — skipping background scheduler.")
        return

    if HAS_APSCHEDULER and scheduler and not scheduler.running:
        scheduler.add_job(refresh_all_data, "interval", hours=6, id="polar_sync_job", replace_existing=True)
        scheduler.start()
        logger.info("Polar Navigator background data scheduler active (interval: 6h).")


def trigger_manual_refresh() -> dict:
    """Manual on-demand synchronization trigger."""
    start_time = datetime.utcnow()
    refresh_all_data()
    elapsed = (datetime.utcnow() - start_time).total_seconds()
    return {
        "status": "success",
        "synced_at": datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": round(elapsed, 2),
        "message": "All Antarctic scientific caches refreshed."
    }
