# src/features.py
"""
Feature engineering recipe — returns an UNFITTED ColumnTransformer.
Fitting is deferred to train.py to prevent data leakage.
"""

import logging
from typing import List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    n_bins: int = 3,
):
    """Build and return an UNFITTED ColumnTransformer preprocessing recipe."""
    logger.info("Building feature preprocessor recipe …")

    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []

    if not quantile_bin_cols and not categorical_onehot_cols and not numeric_passthrough_cols:
        raise ValueError(
            "All feature lists are empty. "
            "Check SETTINGS → features section."
        )

    transformers = []

    if quantile_bin_cols:
        bin_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("binner", KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy="quantile",
            )),
        ])
        transformers.append(("quantile_bin", bin_pipeline, quantile_bin_cols))

    if numeric_passthrough_cols:
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num_passthrough", num_pipeline, numeric_passthrough_cols))

    if categorical_onehot_cols:
        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ])
        transformers.append(("cat_onehot", cat_pipeline, categorical_onehot_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    logger.info(
        "Preprocessor recipe built — quantile_bin: %s, passthrough: %s, categorical: %s",
        quantile_bin_cols, numeric_passthrough_cols, categorical_onehot_cols,
    )
    return preprocessor
