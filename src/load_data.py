# src/load_data.py
"""
Data ingestion — load raw data from disk.
Never mutate the source file. Return an unmodified DataFrame.
"""

import logging
from pathlib import Path

import pandas as pd

from src.utils import load_csv

logger = logging.getLogger(__name__)


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """Load raw CSV data from disk with fail-fast guards."""
    logger.info("Loading raw data from: %s", raw_data_path)

    raw_data_path = Path(raw_data_path)

    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {raw_data_path}. "
            "Please place the dataset in the configured path."
        )

    if raw_data_path.is_dir():
        raise IsADirectoryError(
            f"Expected a file but got a directory: {raw_data_path}"
        )

    df = load_csv(raw_data_path)

    if df.empty:
        raise ValueError(
            f"Loaded DataFrame from {raw_data_path} has zero rows. "
            "Check the upstream data export."
        )

    logger.info(
        "Data loaded successfully — shape: %d rows x %d columns",
        df.shape[0], df.shape[1],
    )
    return df
