"""
Module: Data Validation
-----------------------
Role: Check data quality (schema, types, ranges) before training.
Input: pandas.DataFrame.
Output: Boolean (True if valid) or raises Error.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "Churn",
    "AccountWeeks",
    "ContractRenewal",
    "DataPlan",
    "DataUsage",
    "CustServCalls",
    "DayMins",
    "DayCalls",
    "MonthlyCharge",
    "OverageFee",
    "RoamMins",
]

EXPECTED_TYPES = {
    "Churn": "int64",
    "AccountWeeks": "int64",
    "ContractRenewal": "int64",
    "DataPlan": "int64",
    "DataUsage": "float64",
    "CustServCalls": "int64",
    "DayMins": "float64",
    "DayCalls": "int64",
    "MonthlyCharge": "float64",
    "OverageFee": "float64",
    "RoamMins": "float64",
}

NON_NEGATIVE_COLUMNS = [
    "AccountWeeks",
    "DataUsage",
    "CustServCalls",
    "DayMins",
    "DayCalls",
    "MonthlyCharge",
    "OverageFee",
    "RoamMins",
]


def validate_schema(df: pd.DataFrame) -> None:
    """Check that all required columns are present in the DataFrame.

    Args:
        df: Input DataFrame to validate.

    Raises:
        ValueError: If the DataFrame is empty or any required column is missing.
    """
    if df.empty:
        raise ValueError("Dataset is empty.")

    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def validate_types(df: pd.DataFrame) -> None:
    """Check that each column has the expected data type.

    Args:
        df: Input DataFrame to validate.

    Raises:
        ValueError: If any column has an unexpected dtype.
    """
    for col, expected in EXPECTED_TYPES.items():
        actual = str(df[col].dtype)
        if actual != expected:
            raise ValueError(
                f"Column '{col}' has dtype '{actual}', expected '{expected}'."
            )


def validate_missing(df: pd.DataFrame) -> None:
    """Check that the DataFrame contains no missing values.

    Args:
        df: Input DataFrame to validate.

    Raises:
        ValueError: If any column contains null values.
    """
    for col in df.columns:
        if df[col].isnull().any():
            raise ValueError(f"Dataset contains missing values in column '{col}'.")


def validate_duplicates(df: pd.DataFrame) -> None:
    """Check that the DataFrame contains no duplicated rows.

    Args:
        df: Input DataFrame to validate.

    Raises:
        ValueError: If duplicated rows are found.
    """
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        raise ValueError(f"Dataset contains {n_duplicates} duplicated row(s).")


def validate_ranges(df: pd.DataFrame) -> None:
    """Check that numeric columns contain only non-negative values.

    Args:
        df: Input DataFrame to validate.

    Raises:
        ValueError: If any non-negative column contains a negative value.
    """
    for col in NON_NEGATIVE_COLUMNS:
        if (df[col] < 0).any():
            raise ValueError(
                f"Column '{col}' contains negative values, expected >= 0."
            )


def validate_target(df: pd.DataFrame) -> None:
    """Check that the target column 'Churn' contains only values 0 or 1.

    Args:
        df: Input DataFrame to validate.

    Raises:
        ValueError: If 'Churn' contains values other than 0 or 1.
    """
    invalid = df["Churn"][~df["Churn"].isin([0, 1])]
    if not invalid.empty:
        raise ValueError(
            f"Column 'Churn' contains invalid values: {invalid.unique().tolist()}. "
            "Expected only 0 or 1."
        )


def validate_dataframe(df: pd.DataFrame) -> bool:
    """Run all validation checks on the DataFrame.

    Calls validate_schema, validate_types, validate_missing,
    validate_duplicates, validate_ranges, and validate_target in order.

    Args:
        df: Input DataFrame to validate.

    Returns:
        True if all checks pass.

    Raises:
        ValueError: If any validation check fails.
    """
    validate_schema(df)
    validate_types(df)
    validate_missing(df)
    validate_duplicates(df)
    validate_ranges(df)
    validate_target(df)

    logger.info("Dataset validation passed.")
    return True
