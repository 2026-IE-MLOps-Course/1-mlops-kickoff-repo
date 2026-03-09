"""
Module: Data Validation
-----------------------
Role: Check data quality (schema, types, ranges) before training.
Input: pandas.DataFrame.
Output: Boolean (True if valid) or raises Error.
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Garbage In, Garbage Out (GIGO).
  A dedicated validation gate catches schema violations, missing columns,
  impossible values, and empty data BEFORE expensive compute begins.
- Responsibility (separation of concerns): Check data quality — schema,
  types, ranges, completeness. Does NOT fix data (that is clean_data's job).
- Pipeline contract (inputs and outputs):
  Input  — df (pd.DataFrame) and required_columns (list of column names).
  Output — bool: True if valid, raises ValueError if invalid.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Inputs:
    - df (pd.DataFrame): The cleaned DataFrame to validate.
    - required_columns (list): Column names that must be present.
    Outputs:
    - bool: True if the DataFrame passes all validation checks.
    Why this contract matters for reliable ML delivery:
    - Fail-fast validation prevents silent pipeline failures. Catching a
      missing column or negative value here is far cheaper than debugging
      a cryptic Scikit-Learn error deep inside training.
    """
    print("[validate] Starting data validation …")  # TODO: replace with logging later

    # Guard against empty DataFrame
    if df.empty:
        raise ValueError(
            "Validation received an empty DataFrame. "
            "Check the upstream data loading and cleaning steps."
        )

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Every dataset has unique domain rules (e.g., ages cannot be negative,
    #       days supply must be positive, classification targets must be binary)
    # Examples:
    # 1. assert "OD" in df.columns, "Target column 'OD' is missing"
    # 2. assert (df["rx_ds"] >= 0).all(), "Negative days supply detected"
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Student implementation for VoyageIQ travel dataset:
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
            issues.append(
                f"'{col}': {n_bad} values below minimum ({col_min})"
            )
        if col_max is not None and (df[col] > col_max).any():
            n_bad = int((df[col] > col_max).sum())
            issues.append(
                f"'{col}': {n_bad} values above maximum ({col_max})"
            )

    # Check target column for nulls
    if "total_cost" in df.columns and df["total_cost"].isna().any():
        n_na = int(df["total_cost"].isna().sum())
        issues.append(f"Target 'total_cost' has {n_na} null values")

    if issues:
        msg = f"Data validation found {len(issues)} issue(s):\n"
        for issue in issues:
            msg += f"  - {issue}\n"
        print(f"[validate] FAILED — {msg}")  # TODO: replace with logging later
        raise ValueError(msg)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    print("[validate] Validation passed — no issues detected.")  # TODO: replace with logging later
    return True