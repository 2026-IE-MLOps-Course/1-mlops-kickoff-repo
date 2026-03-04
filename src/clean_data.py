"""
Module: Data Cleaning

Role:
    Perform deterministic and reproducible cleaning of the raw dataset.

Responsibilities:
    - Load raw data safely
    - Standardize column formats
    - Convert data types (e.g., TotalCharges to numeric)
    - Remove duplicates
    - Handle missing values
    - Validate minimal schema expectations
    - Save cleaned dataset

Designed to be:
    - Importable inside main.py
    - Executable from CLI
    - Fully testable
"""

from __future__ import annotations

import logging
import os
from typing import Tuple

import pandas as pd


# ------------------------------------------------------------------------------
# Logging configuration
# ------------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


# ------------------------------------------------------------------------------
# Expected schema (adjust if needed)
# ------------------------------------------------------------------------------

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


# ------------------------------------------------------------------------------
# I/O FUNCTIONS
# ------------------------------------------------------------------------------

def load_raw(path: str) -> pd.DataFrame:
    """
    Load raw CSV data with basic validation.

    Parameters
    ----------
    path : str
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.

    Raises
    ------
    FileNotFoundError
        If file does not exist.
    ValueError
        If loaded DataFrame is empty.
    """
    logger.info("Loading raw data from: %s", path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError("Loaded DataFrame is empty.")

    logger.info("Raw dataset shape: %s", df.shape)
    return df


# ------------------------------------------------------------------------------
# CLEANING STEPS
# ------------------------------------------------------------------------------

def _strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove leading/trailing whitespace from all string columns.
    """
    object_columns = df.select_dtypes(include="object").columns

    for col in object_columns:
        df[col] = df[col].str.strip()

    logger.info("Whitespace stripped from object columns.")
    return df


def _convert_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert 'TotalCharges' column to numeric.
    Invalid values are coerced into NaN.
    """
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"], errors="coerce"
        )
        logger.info("TotalCharges converted to numeric.")

    return df


def _convert_tenure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure 'tenure' column is numeric.
    """
    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")

    return df



def clean_data(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Main cleaning pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataset.
    drop_na : bool
        If True, drop rows containing missing values.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    logger.info("Starting cleaning pipeline.")
    df = df.copy()

    # 1. Strip whitespace
    df = _strip_whitespace(df)

    # 2. Convert numeric columns
    df = _convert_total_charges(df)
    df = _convert_tenure(df)


    # 4. Handle missing values
    if drop_na:
        before = df.shape[0]
        df = df.dropna().reset_index(drop=True)
        after = df.shape[0]

        logger.info(
            "Rows dropped due to missing values: %s",
            before - after,
        )

    logger.info("Cleaning finished. Final shape: %s", df.shape)
    return df


# ------------------------------------------------------------------------------
# VALIDATION
# ------------------------------------------------------------------------------

def validate_df(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate minimal dataset requirements.

    Returns
    -------
    (bool, str)
        Validation result and message.
    """
    if df.empty:
        return False, "DataFrame is empty."

    if "customerID" not in df.columns:
        return False, "Missing required column: customerID."

    missing_columns = [
        col for col in EXPECTED_COLUMNS if col not in df.columns
    ]

    if missing_columns:
        logger.warning("Missing expected columns: %s", missing_columns)
        return True, "Warning: some expected columns are missing."

    return True, "Validation passed."


# ------------------------------------------------------------------------------
# SAVE FUNCTION
# ------------------------------------------------------------------------------

def save_df(df: pd.DataFrame, path: str) -> None:
    """
    Save cleaned dataset to CSV.

    Creates directories if they do not exist.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

    logger.info("Clean dataset saved to: %s", path)


# ------------------------------------------------------------------------------
# CLI ENTRY POINT
# ------------------------------------------------------------------------------

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean Telco Customer Churn dataset."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw CSV file."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to save cleaned CSV file."
    )

    parser.add_argument(
        "--no-drop-na",
        action="store_true",
        help="Keep missing values instead of dropping them."
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    df_raw = load_raw(args.input)
    df_clean = clean_data(df_raw, drop_na=not args.no_drop_na)

    valid, message = validate_df(df_clean)

    if not valid:
        logger.error("Validation failed: %s", message)
        raise RuntimeError(message)

    save_df(df_clean, args.output)


if __name__ == "__main__":
    main()
