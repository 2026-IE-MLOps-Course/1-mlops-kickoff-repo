"""
Module: Feature Engineering
-----------------------------
Role: Define the transformation "recipe" (binning, encoding, scaling) to be bundled with the model.
Input: Configuration (lists of column names).
Output: scikit-learn ColumnTransformer object.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger("voyageiq.features")


# ------------------------------------------------------------------ #
#  Derived-feature helpers (row-wise, deterministic, no leakage)     #
# ------------------------------------------------------------------ #

def _extract_destination_country(df: pd.DataFrame) -> pd.DataFrame:
    """Split 'destination' into 'destination_city' and 'destination_country'.

    Args:
        df: DataFrame that may contain a 'destination' column.

    Returns:
        pd.DataFrame: DataFrame with derived destination columns added.
    """
    if "destination" in df.columns:
        parts = df["destination"].str.split(",", n=1, expand=True)
        df = df.copy()
        df["destination_city"] = parts[0].str.strip() if 0 in parts.columns else "Unknown"
        df["destination_country"] = (
            parts[1].str.strip() if 1 in parts.columns else "Unknown"
        )
    return df


def _extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive travel_month and day_of_week from 'start_date'.

    Args:
        df: DataFrame that may contain a 'start_date' column.

    Returns:
        pd.DataFrame: DataFrame with date-derived columns added.
    """
    if "start_date" in df.columns:
        df = df.copy()
        dt = pd.to_datetime(df["start_date"], errors="coerce", dayfirst=False)
        df["travel_month"] = dt.dt.month.fillna(0).astype(int)
        df["day_of_week"] = dt.dt.dayofweek.fillna(0).astype(int)
    return df


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create all derived columns required by the pipeline.

    This function is called *before* the train/val/test split so that
    every split has access to the same column set.  No statistics are
    learned here — only deterministic row-wise transformations.

    Args:
        df: Cleaned and validated DataFrame.
        config: Full pipeline configuration dictionary.

    Returns:
        pd.DataFrame: DataFrame enriched with derived features.
    """
    logger.info("Engineering derived features …")

    df = _extract_destination_country(df)
    df = _extract_date_features(df)

    # Ensure numeric types for age and duration
    if "traveler_age" in df.columns:
        df["traveler_age"] = pd.to_numeric(
            df["traveler_age"], errors="coerce"
        )
    if "duration_days" in df.columns:
        df["duration_days"] = pd.to_numeric(
            df["duration_days"], errors="coerce"
        )

    logger.info(
        "Feature engineering complete — columns: %s", list(df.columns)
    )
    return df


def build_preprocessor(config: dict) -> ColumnTransformer:
    """Return an *unfitted* scikit-learn ColumnTransformer recipe.

    The recipe is assembled from the feature lists in config.yaml.
    Fitting is intentionally deferred to the training module so that
    statistics are learned exclusively on the training split.

    Args:
        config: Full pipeline configuration dictionary.

    Returns:
        ColumnTransformer: Unfitted preprocessing pipeline.

    Raises:
        ValueError: If both feature lists are empty.
    """
    feat_cfg = config.get("features", {})

    numeric_features = feat_cfg.get("numeric_features", [])
    categorical_features = feat_cfg.get("categorical_features", [])
    date_features = feat_cfg.get("date_derived_features", [])

    # Date-derived features are treated as numeric
    all_numeric = numeric_features + date_features

    if not all_numeric and not categorical_features:
        raise ValueError(
            "Both numeric and categorical feature lists are empty. "
            "Check config.yaml → features section."
        )

    num_imputer = feat_cfg.get("numeric_imputer_strategy", "median")
    cat_imputer = feat_cfg.get("categorical_imputer_strategy", "most_frequent")

    # Numeric sub-pipeline: impute → scale
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=num_imputer)),
        ("scaler", StandardScaler()),
    ])

    # Categorical sub-pipeline: impute → one-hot encode
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy=cat_imputer)),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if all_numeric:
        transformers.append(("num", numeric_pipeline, all_numeric))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Block unauthorised raw columns
    )

    logger.info(
        "Preprocessor recipe built — numeric: %s, categorical: %s",
        all_numeric,
        categorical_features,
    )
    return preprocessor
