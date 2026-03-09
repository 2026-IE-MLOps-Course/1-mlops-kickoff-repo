"""
Educational Goal:
- Why this module exists in an MLOps system: Feature engineering rules must be
  separated from both data cleaning and model training. This module defines the
  preprocessing "recipe" (a ColumnTransformer) that is returned UNFITTED.
  Fitting is deferred to train.py to prevent data leakage.
- Responsibility (separation of concerns): Define transformation rules —
  which columns get binned, which get one-hot encoded, which pass through.
  Never call .fit(). Never accept raw DataFrames.
- Pipeline contract (inputs and outputs):
  Input  — Lists of column names and configuration integers.
  Output — An unfitted sklearn ColumnTransformer object.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from typing import Optional, List

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler


def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    n_bins: int = 3,
):
    """
    Inputs:
    - quantile_bin_cols (Optional[List[str]]): Columns to discretise via
      quantile binning using KBinsDiscretizer.
    - categorical_onehot_cols (Optional[List[str]]): Columns to encode
      using OneHotEncoder.
    - numeric_passthrough_cols (Optional[List[str]]): Numeric columns to
      impute and scale without binning.
    - n_bins (int): Number of quantile bins for KBinsDiscretizer.
    Outputs:
    - sklearn ColumnTransformer: An UNFITTED preprocessing recipe.
    Why this contract matters for reliable ML delivery:
    - Returning an unfitted recipe separates "define rules" from "execute rules".
      The recipe is fitted exclusively on the training split inside train.py,
      which structurally prevents data leakage by design.
    """
    print("[features] Building feature preprocessor recipe …")  # TODO: replace with logging later

    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []

    if not quantile_bin_cols and not categorical_onehot_cols and not numeric_passthrough_cols:
        raise ValueError(
            "All feature lists are empty. "
            "Check SETTINGS → features section."
        )

    transformers = []

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Different datasets require different preprocessing strategies
    #       (e.g., log transforms, polynomial features, custom encoders)
    # Examples:
    # 1. Add a FunctionTransformer for log1p scaling
    # 2. Chain SimpleImputer → KBinsDiscretizer → OrdinalEncoder for binned cols
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Student implementation for VoyageIQ travel dataset:

    # Quantile binning sub-pipeline: impute → bin
    if quantile_bin_cols:
        bin_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("binner", KBinsDiscretizer(
                n_bins=n_bins,
                encode="ordinal",
                strategy="quantile",
            )),
        ])
        transformers.append(("quantile_bin", bin_pipeline, quantile_bin_cols))

    # Numeric passthrough sub-pipeline: impute → scale
    if numeric_passthrough_cols:
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num_passthrough", num_pipeline, numeric_passthrough_cols))

    # Categorical sub-pipeline: impute → one-hot encode
    if categorical_onehot_cols:
        # Compatibility: try sparse_output first (sklearn >= 1.2),
        # fall back to sparse for older versions
        try:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )
        except TypeError:
            encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse=False,
            )
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", encoder),
        ])
        transformers.append(("cat_onehot", cat_pipeline, categorical_onehot_cols))
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # Block unauthorised raw columns
    )

    print(  # TODO: replace with logging later
        f"[features] Preprocessor recipe built — "
        f"quantile_bin: {quantile_bin_cols}, "
        f"passthrough: {numeric_passthrough_cols}, "
        f"categorical: {categorical_onehot_cols}"
    )
    return preprocessor
