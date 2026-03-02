"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature engineering.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""

import pandas as pd


def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Clean raw data into a stable, model-ready tabular dataset.

    Key expectations for this project:
      - Standardize column naming
      - Rename 'rx ds' -> 'rx_ds' if present
      - Preserve target column
      - Remove duplicates
      - Enforce binary 0/1 in flag columns when possible
    """
    if df_raw is None:
        raise ValueError("df_raw cannot be None")
    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError("df_raw must be a pandas DataFrame")
    if df_raw.empty:
        raise ValueError("df_raw is empty")
    if not target_column or not isinstance(target_column, str):
        raise ValueError("target_column must be a non-empty string")

    df = df_raw.copy()

    # 1) Standardize column names (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]

    # 2) Specific contract: rx ds -> rx_ds
    if "rx ds" in df.columns and "rx_ds" not in df.columns:
        df = df.rename(columns={"rx ds": "rx_ds"})

    # 3) Target must exist
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # 4) Drop exact duplicates
    df = df.drop_duplicates()

    # 5) Coerce target to int (assumes target is already 0/1-like)
    df[target_column] = df[target_column].astype(int)

    # 6) Coerce binary-ish flags to 0/1 when safe
    exclude = {target_column, "rx_ds", "ID"}
    candidates = [c for c in df.columns if c not in exclude]

    for c in candidates:
        s = df[c].dropna()
        if not s.empty and s.isin([0, 1, True, False]).all():
            df[c] = df[c].astype(int)

    return df