"""
Module: Feature Engineering
---------------------------
Role: Define the transformation "recipe" (binning, encoding, scaling) to be bundled with the model.
Input: Configuration (lists of column names).
Output: scikit-learn ColumnTransformer object.
"""


"""
Educational Goal:
- Why this module exists in an MLOps system: Encapsulates feature transformations so they can be bundled with the model.
- Responsibility (separation of concerns): Defining the exact mathematical transformations applied to input features.
- Pipeline contract (inputs and outputs): Returns an unfitted scikit-learn ColumnTransformer. Does NOT accept data.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from typing import Optional, List


def get_feature_preprocessor(quantile_bin_cols: Optional[List[str]] = None,
                             categorical_onehot_cols: Optional[List[str]] = None,
                             numeric_passthrough_cols: Optional[List[str]] = None,
                             n_bins: int = 3):
    """
    Inputs:
    - quantile_bin_cols: List of numeric column names to discretize.
    - categorical_onehot_cols: List of categorical column names to one-hot encode.
    - numeric_passthrough_cols: List of numeric column names to pass through unchanged.
    - n_bins: Integer number of bins for the discretizer.
    Outputs:
    - An unfitted scikit-learn ColumnTransformer object.
    Why this contract matters for reliable ML delivery:
    - Returning a recipe (transformer) rather than transformed data prevents data leakage and ensures train/infer use identical logic.
    """
    print("Executing get_feature_preprocessor to build transformation recipe")  # TODO: replace with logging later

    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []

    # Handle older scikit-learn versions gracefully
    try:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

    transformers = []

    if quantile_bin_cols:
        transformers.append(("q_bin", KBinsDiscretizer(
            n_bins=n_bins, encode='ordinal', strategy='quantile'), quantile_bin_cols))

    if categorical_onehot_cols:
        transformers.append(("cat_ohe", ohe, categorical_onehot_cols))

    if numeric_passthrough_cols:
        transformers.append(
            ("num_pass", "passthrough", numeric_passthrough_cols))

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Different domains require different feature engineering (e.g., text TF-IDF, standard scaling, polynomial features).
    # Examples:
    # 1. transformers.append(("scaler", StandardScaler(), numeric_cols))
    # 2. transformers.append(("tfidf", TfidfVectorizer(), text_col))
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return ColumnTransformer(transformers=transformers, remainder="drop")
