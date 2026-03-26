"""
Module: validate.py
-------------------
Role: Quality gate to ensure data schema, types, and ranges are compliant before training.
Input: pandas.DataFrame.
Output: True if valid, or raises DataValidationError.
"""

import logging
import pandas as pd

# Logging configuration for industrial traceability [cite: 50]
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataValidationError(Exception):
    """Custom exception for data validation errors[cite: 54]."""
    pass


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validates the schema and integrity of the Telco Churn data.

    Why: Acts as a "fail-fast" mechanism to prevent corrupted data from 
    reaching the training phase and generating unreliable models[cite: 21, 50].

    Inputs:
    - df: DataFrame to be validated.
    - required_columns: List of mandatory columns.

    Outputs:
    - True if validation passes, otherwise raises DataValidationError.
    """
    logger.info("Executing DataFrame validation...")

    # 1. Empty DataFrame check [cite: 50]
    if df.empty:
        logger.error("Validation failed: The DataFrame is completely empty.")
        raise DataValidationError("The input DataFrame contains no data.")

    # 2. Schema verification (mandatory columns) [cite: 21, 50]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(
            f"Validation failed: Mandatory columns are missing: {missing_cols}")
        raise DataValidationError(f"Missing columns: {missing_cols}")

    # --------------------------------------------------------
    # PROJECT-SPECIFIC LOGIC (Based on New Final.ipynb)
    # --------------------------------------------------------
    try:
        # 3. Critical data type validation
        # 'MonthlyCharges' must be numeric for proper scaling
        if not pd.api.types.is_numeric_dtype(df['MonthlyCharges']):
            raise DataValidationError(
                "The 'MonthlyCharges' column must be numeric.")

        # 'tenure' must be numeric and have no negative values
        if not pd.api.types.is_numeric_dtype(df['tenure']):
            raise DataValidationError("The 'tenure' column must be numeric.")
        if (df['tenure'] < 0).any():
            raise DataValidationError(
                "Negative values detected in the 'tenure' column.")

        # 'SeniorCitizen' must be numeric (typically 0 or 1)
        if not pd.api.types.is_numeric_dtype(df['SeniorCitizen']):
            raise DataValidationError(
                "The 'SeniorCitizen' column must be numeric.")

        # 4. Target column check (Churn)
        if df['Churn'].isnull().any():
            logger.warning(
                "Null values detected in the target column 'Churn'.")

        logger.info("Validation completed successfully. Data is compliant.")
        return True

    except DataValidationError as e:
        logger.error(f"Validation Error: {e}")
        raise e
    except Exception as e:
        logger.error(f"An unexpected error occurred during validation: {e}")
        raise e
