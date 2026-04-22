# src/validate.py
"""
Data validation gate — schema, types, ranges.
Does NOT fix data (that is clean_data's job).
"""

import logging
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate the cleaned DataFrame against required columns and domain rules.
    Returns True if valid, raises ValueError otherwise.
    """
    logger.info("Starting data validation …")

    if df.empty:
        raise ValueError(
            "Validation received an empty DataFrame. "
            "Check the upstream data loading and cleaning steps."
        )

    issues = []

    # Check required columns exist
    present = set(df.columns)
    for col in required_columns:
        if col not in present:
            issues.append(f"Required column missing: '{col}'")

    # Numeric range checks for travel domain
    range_checks = {
        "duration_days": {"min": 1, "max": 365},
        "traveler_age": {"min": 1, "max": 120},
        "accommodation_cost": {"min": 0, "max": 100000},
        "transportation_cost": {"min": 0, "max": 100000},
    }
    for col, bounds in range_checks.items():
        if col not in df.columns:
            continue
        col_min = bounds.get("min")
        col_max = bounds.get("max")
        if col_min is not None and (df[col] < col_min).any():
            n_bad = int((df[col] < col_min).sum())
            issues.append(f"'{col}': {n_bad} values below minimum ({col_min})")
        if col_max is not None and (df[col] > col_max).any():
            n_bad = int((df[col] > col_max).sum())
            issues.append(f"'{col}': {n_bad} values above maximum ({col_max})")

    # Check target column for nulls
    if "total_cost" in df.columns and df["total_cost"].isna().any():
        n_na = int(df["total_cost"].isna().sum())
        issues.append(f"Target 'total_cost' has {n_na} null values")

    if issues:
        msg = f"Data validation found {len(issues)} issue(s):\n"
        for issue in issues:
            msg += f"  - {issue}\n"
        logger.error("FAILED — %s", msg)
        raise ValueError(msg)

    logger.info("Validation passed — no issues detected.")
    return True
