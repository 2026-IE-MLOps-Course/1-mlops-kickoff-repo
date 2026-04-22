# src/utils.py
"""
Centralised I/O plumbing — read files, write files, serialise/deserialise model artifacts.
Contains zero business logic.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(filepath: Path) -> pd.DataFrame:
    """Load a CSV file and return a DataFrame."""
    logger.info("Loading CSV from %s", filepath)

    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1")
    except pd.errors.ParserError as exc:
        raise pd.errors.ParserError(
            f"Failed to parse CSV at {filepath}. "
            "Check the file format and separator."
        ) from exc

    logger.info("CSV loaded — shape: %s", df.shape)
    return df


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """Save a DataFrame to CSV with index=False."""
    logger.info("Saving CSV to %s", filepath)

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)

    logger.info("CSV saved successfully.")


def save_model(model, filepath: Path) -> None:
    """Persist a fitted sklearn Pipeline to disk."""
    logger.info("Saving model to %s", filepath)

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)

    logger.info("Model saved successfully.")


def load_model(filepath: Path):
    """Load a serialised sklearn Pipeline from disk."""
    logger.info("Loading model from %s", filepath)

    filepath = Path(filepath)
    if not filepath.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {filepath}. Run training first."
        )

    model = joblib.load(filepath)
    logger.info("Model loaded successfully.")
    return model
