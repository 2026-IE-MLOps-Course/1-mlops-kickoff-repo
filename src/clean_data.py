"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature engineering.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Cleaning is dataset-specific and frequently changes; isolating it reduces regression risk.
- Responsibility (separation of concerns): Transform raw DataFrame into a clean, model-ready tabular dataset (still pre-feature-engineering).
- Pipeline contract (inputs and outputs): Input raw df + target column name; output cleaned df with target preserved.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Inputs:
    - df_raw: Raw DataFrame.
    - target_column: Name of the target column to preserve.
    Outputs:
    - df_clean: Cleaned DataFrame (baseline is identity copy).
    Why this contract matters for reliable ML delivery:
    - Separating cleaning from training prevents hidden notebook mutations and makes behavior reproducible across runs.
    """
    print("[clean_data.clean_dataframe] Cleaning raw dataframe (baseline: identity copy)")  # TODO: replace with logging later

    df_clean = df_raw.copy()

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Cleaning depends on source quirks (missing values, outliers, deduplication, label fixes)
    # Examples:
    # 1. Drop rows with invalid target values, enforce types
    # 2. Normalize text categories, impute missing numerical values
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    # Minimal guardrail: ensure target exists if specified (do not mutate; validate.py handles required cols)
    if target_column not in df_clean.columns:
        print(
            f"[clean_data.clean_dataframe] Warning: target_column '{target_column}' not found in dataframe columns"
        )  # TODO: replace with logging later

    return df_clean