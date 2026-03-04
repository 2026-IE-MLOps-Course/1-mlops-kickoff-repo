"""
Module: Model Training
----------------------
Role: Bundle preprocessing and algorithms into a single Pipeline and fit on training data.
Input: pandas.DataFrame (Processed) + ColumnTransformer (Recipe).
Output: Serialized scikit-learn Pipeline in `models/`.
"""


"""
Educational Goal:
- Why this module exists in an MLOps system: Isolates the training loop, enabling hyperparameter tuning runs without rerunning data prep.
- Responsibility (separation of concerns): Fitting the combined Pipeline (preprocessor + model) to the training split.
- Pipeline contract (inputs and outputs): Takes training data and an unfitted preprocessor, returns a fitted Pipeline.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
import pandas as pd


def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor, problem_type: str):
    """
    Inputs:
    - X_train: Training features DataFrame.
    - y_train: Training target Series.
    - preprocessor: The unfitted ColumnTransformer recipe.
    - problem_type: "classification" or "regression".
    Outputs:
    - A fully fitted scikit-learn Pipeline.
    Why this contract matters for reliable ML delivery:
    - Bundling the preprocessor and estimator into one Pipeline guarantees that inference data undergoes the exact same transformations as training data.
    """
    print("Executing train_model")  # TODO: replace with logging later

    if problem_type == "classification":
        estimator = LogisticRegression(max_iter=500)
    else:
        estimator = Ridge()

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: You will want to experiment with different algorithms (RandomForest, XGBoost) and hyperparameters.
    # Examples:
    # 1. estimator = RandomForestClassifier(n_estimators=100, max_depth=5)
    # 2. estimator = xgb.XGBRegressor(learning_rate=0.01)
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator)
    ])

    # TODO: replace with logging later
    print("Fitting pipeline on training data...")
    pipeline.fit(X_train, y_train)

    return pipeline
