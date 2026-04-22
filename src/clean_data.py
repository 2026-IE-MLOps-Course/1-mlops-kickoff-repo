# src/clean_data.py
"""
Data cleaning / stabilisation.
Supports both training mode (target_column provided) and inference mode (target_column=None).
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def clean_dataframe(df_raw: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
    """
    Clean raw data for both training and inference.

    Training mode (target_column provided): creates target, drops NaN rows.
    Inference mode (target_column=None): skips target creation, keeps rows.
    """
    logger.info("Cleaning started — initial rows: %d", len(df_raw))

    df = df_raw.copy()
    initial_rows = len(df)

    # 1. Standardise column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    logger.info("Column names standardised: %s", list(df.columns))

    # 2. Drop ID columns
    drop_cols = ["trip_id", "traveler_name"]
    df = df.drop(columns=drop_cols, errors="ignore")
    logger.info("Dropped columns (if present): %s", drop_cols)

    # 3. Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    logger.info("Duplicates removed: %d", before - len(df))

    # 4. Drop rows with missing values
    before = len(df)
    df = df.dropna()
    logger.info("Rows dropped (NaN): %d", before - len(df))

    # 5. Parse currency columns
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
    cost_cols_present = [c for c in ["accommodation_cost", "transportation_cost"] if c in df.columns]
    if cost_cols_present:
        df = df.dropna(subset=cost_cols_present)
        if before - len(df) > 0:
            logger.info("Rows dropped (unparseable cost values): %d", before - len(df))

    # 6. Create target column (training mode only)
    if target_column is not None:
        if "accommodation_cost" in df.columns and "transportation_cost" in df.columns:
            df[target_column] = df["accommodation_cost"] + df["transportation_cost"]
            logger.info("Target column '%s' created.", target_column)
        else:
            raise KeyError(
                f"Cannot create target '{target_column}': "
                "'accommodation_cost' and/or 'transportation_cost' columns are missing."
            )

    # 7. Extract destination country
    if "destination" in df.columns:
        parts = df["destination"].str.split(",", n=1, expand=True)
        df["destination_city"] = parts[0].str.strip() if 0 in parts.columns else "Unknown"
        df["destination_country"] = (
            parts[1].str.strip() if 1 in parts.columns else "Unknown"
        )

    # 8. Extract date-derived features
    if "start_date" in df.columns:
        dt = pd.to_datetime(df["start_date"], errors="coerce", dayfirst=False)
        df["travel_month"] = dt.dt.month.fillna(0).astype(int)
        df["day_of_week"] = dt.dt.dayofweek.fillna(0).astype(int)

    # 9. Ensure numeric types
    for num_col in ["traveler_age", "duration_days"]:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    # 10. Verify target exists (training mode only)
    if target_column is not None and target_column not in df.columns:
        raise KeyError(
            f"Target column '{target_column}' is missing after cleaning. "
            "Check your cleaning logic."
        )

    df = df.reset_index(drop=True)

    final_rows = len(df)
    logger.info(
        "Cleaning complete — final rows: %d (dropped: %d)",
        final_rows, initial_rows - final_rows,
    )
    return df
