"""
Educational Goal:
- Why this module exists in an MLOps system: Raw data is messy. A dedicated
  cleaning module stabilises the data early so the pipeline behaves predictably,
  reproducibly, and safely in production.
- Responsibility (separation of concerns): Standardise column names, remove
  duplicates/missing values, create the target variable. Never perform
  statistical feature engineering (that belongs in features.py after splitting).
- Pipeline contract (inputs and outputs):
  Input  — df_raw (pd.DataFrame) from load_data, target_column (str).
  Output — pd.DataFrame with deterministic schema, ready for validation.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Inputs:
    - df_raw (pd.DataFrame): The raw DataFrame loaded by load_raw_data.
    - target_column (str): Name of the target column expected after cleaning.
    Outputs:
    - pd.DataFrame: Cleaned DataFrame with standardised column names,
      no duplicates, no missing values, and the target column present.
    Why this contract matters for reliable ML delivery:
    - Deterministic cleaning guarantees the same schema given the same input.
      Separating cleaning from feature engineering prevents data leakage —
      statistical transforms like pd.qcut() must happen AFTER the train/test split.
    """
    print(f"[clean_data] Cleaning started — initial rows: {len(df_raw)}")  # TODO: replace with logging later

    df = df_raw.copy()
    initial_rows = len(df)

    # 1. Standardise column names — prevents invisible spaces from breaking code
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    print(f"[clean_data] Column names standardised: {list(df.columns)}")  # TODO: replace with logging later

    # 2. Drop columns that should not enter the pipeline (idempotent)
    drop_cols = ["trip_id", "traveler_name"]
    df = df.drop(columns=drop_cols, errors="ignore")
    print(f"[clean_data] Dropped columns (if present): {drop_cols}")  # TODO: replace with logging later

    # 3. Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print(f"[clean_data] Duplicates removed: {before - len(df)}")  # TODO: replace with logging later

    # 4. Drop rows with missing values
    before = len(df)
    df = df.dropna()
    print(f"[clean_data] Rows dropped (NaN): {before - len(df)}")  # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Every dataset has unique cleaning needs (currency parsing, date
    #       standardisation, domain-specific column creation, etc.)
    # Examples:
    # 1. df["rx_ds"] = df["rx ds"]  # standardise a column name
    # 2. df[target_column] = (df["col_a"] + df["col_b"])  # create a derived target
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Student implementation for VoyageIQ travel dataset:
    # Parse currency columns that may contain symbols like $, USD, commas
    for cost_col in ["accommodation_cost", "transportation_cost"]:
        if cost_col in df.columns:
            df[cost_col] = (
                df[cost_col]
                .astype(str)
                .str.strip()
                .str.replace("$", "", regex=False)
                .str.replace(",", "", regex=False)
                .str.replace("USD", "", regex=False)
                .str.replace("EUR", "", regex=False)
                .str.strip()
            )
            df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce")

    # Drop rows where cost parsing failed
    before = len(df)
    df = df.dropna(subset=["accommodation_cost", "transportation_cost"])
    if before - len(df) > 0:
        print(  # TODO: replace with logging later
            f"[clean_data] Rows dropped (unparseable cost values): {before - len(df)}"
        )

    # Create the target column: total_cost = accommodation + transportation
    if "accommodation_cost" in df.columns and "transportation_cost" in df.columns:
        df[target_column] = df["accommodation_cost"] + df["transportation_cost"]
        print(f"[clean_data] Target column '{target_column}' created.")  # TODO: replace with logging later
    else:
        raise KeyError(
            f"Cannot create target '{target_column}': "
            "'accommodation_cost' and/or 'transportation_cost' columns are missing."
        )

    # Extract destination country from "City, Country" format
    if "destination" in df.columns:
        parts = df["destination"].str.split(",", n=1, expand=True)
        df["destination_city"] = parts[0].str.strip() if 0 in parts.columns else "Unknown"
        df["destination_country"] = (
            parts[1].str.strip() if 1 in parts.columns else "Unknown"
        )

    # Extract date-derived features (row-wise, deterministic, no leakage)
    if "start_date" in df.columns:
        dt = pd.to_datetime(df["start_date"], errors="coerce", dayfirst=False)
        df["travel_month"] = dt.dt.month.fillna(0).astype(int)
        df["day_of_week"] = dt.dt.dayofweek.fillna(0).astype(int)

    # Ensure numeric types for key columns
    for num_col in ["traveler_age", "duration_days"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    # 5. Verify the target column exists after all cleaning
    if target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' is missing after cleaning. "
            "Check your cleaning logic."
        )

    # 6. Reset index to prevent misalignment downstream
    df = df.reset_index(drop=True)

    final_rows = len(df)
    print(  # TODO: replace with logging later
        f"[clean_data] Cleaning complete — final rows: {final_rows} "
        f"(dropped: {initial_rows - final_rows})"
    )
    return df
