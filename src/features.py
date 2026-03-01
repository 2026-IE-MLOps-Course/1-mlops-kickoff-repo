"""
Educational Goal:
- Why this module exists in an MLOps system: Feature preprocessing must be consistent between training and inference to avoid skew.
- Responsibility (separation of concerns): Define the feature transformation recipe (not fitting it) using ColumnTransformer.
- Pipeline contract (inputs and outputs): Input is column-name configuration; output is an unfitted preprocessor object.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from typing import List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder


def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    n_bins: int = 3,
):
    """
    Inputs:
    - quantile_bin_cols: Numeric columns to bin into quantiles (leakage-safe when fitted only on train split).
    - categorical_onehot_cols: Categorical columns to one-hot encode.
    - numeric_passthrough_cols: Numeric columns to pass through unchanged.
    - n_bins: Number of quantile bins for KBinsDiscretizer.
    Outputs:
    - preprocessor: Unfitted ColumnTransformer.
    Why this contract matters for reliable ML delivery:
    - A stable, unfitted recipe can be composed into a Pipeline and fitted only on training data to prevent leakage.
    """
    print("[features.get_feature_preprocessor] Building feature preprocessor recipe (unfitted)")  # TODO: replace with logging later

    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []

    transformers = []

    if quantile_bin_cols:
        transformers.append(
            (
                "quantile_bin",
                KBinsDiscretizer(n_bins=n_bins, encode="onehot-dense", strategy="quantile"),
                quantile_bin_cols,
            )
        )

    if categorical_onehot_cols:
        # scikit-learn version compatibility:
        # - newer versions use sparse_output
        # - older versions use sparse
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
        transformers.append(("categorical_onehot", ohe, categorical_onehot_cols))

    if numeric_passthrough_cols:
        transformers.append(("numeric_passthrough", "passthrough", numeric_passthrough_cols))

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Feature engineering and encoding choices depend on model family and business constraints
    # Examples:
    # 1. Add text vectorization, date part extraction, or target encoding (careful!)
    # 2. Add scaling, interaction features, or custom transformations
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor