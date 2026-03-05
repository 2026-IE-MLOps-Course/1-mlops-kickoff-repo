"""
Module: Data Loader
-------------------
Role: Ingest raw data from sources (CSV, SQL, API).
"""
import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Acquires data from external systems or local files.
    Why: Provides a single entry point for data ingestion.
    """
    logger.info(f"Cargando datos crudos desde: {raw_data_path}")
    df = pd.read_csv(raw_data_path)
    return df
