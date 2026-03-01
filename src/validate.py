"""
Module: Data Validation
-----------------------
Role: Check data quality (schema, types, ranges) before training.
Input: pandas.DataFrame.
Output: Boolean (True if valid) or raises Error.
"""
"""
Educational Goal:
- Why this module exists in an MLOps system: Validation catches obvious data contract breaks early (cheap) before training (expensive).
- Responsibility (separation of concerns): Fail fast on empty data and missing required columns; keep checks minimal and readable.
- Pipeline contract (inputs and outputs): Input df + required column list; output True if valid, otherwise raise.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Inputs:
    - df: DataFrame to validate.
    - required_columns: List of columns that must exist in df.
    Outputs:
    - is_valid: True if valid; raises ValueError otherwise.
    Why this contract matters for reliable ML delivery:
    - Simple, explicit validation prevents silent training on wrong schemas and reduces costly downstream debugging.
    """
    print("[validate.validate_dataframe] Validating dataframe (fail fast for empty/missing columns)")  # TODO: replace with logging later

    if df is None or df.empty:
        raise ValueError("Validation failed: DataFrame is empty. Check data ingestion and cleaning steps.")

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Validation failed: Missing required columns: {missing}. Present columns: {list(df.columns)}")

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add simple dataset-specific checks (range checks, label sanity, duplicates)
    # Why: Each business domain has different "obvious wrong" conditions
    # Examples:
    # 1. Ensure target is not all null: df[target].notna().mean() > 0.99
    # 2. Ensure no negative values in a count column
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return True