"""
Module: Model Training
----------------------
Role: Split data, train model, and save the artifact.
Input: pandas.DataFrame (Processed).
Output: Serialized model file (e.g., .pkl) in `models/`.
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Training should be deterministic and parameterized so we can reproduce models in CI/CD.
- Responsibility (separation of concerns): Build and fit a single scikit-learn Pipeline (preprocess + model).
- Pipeline contract (inputs and outputs): Input train split + preprocessor + problem type; output fitted Pipeline.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline


def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor, problem_type: str):
    """
    Inputs:
    - X_train: Training features (raw columns, not preprocessed).
    - y_train: Training target series.
    - preprocessor: Unfitted ColumnTransformer defining feature transformations.
    - problem_type: "regression" or "classification".
    Outputs:
    - model: Fitted scikit-learn Pipeline.
    Why this contract matters for reliable ML delivery:
    - A single Pipeline artifact prevents training/serving skew and keeps leakage controls enforceable.
    """
    print(f"[train.train_model] Training model as problem_type='{problem_type}' using Pipeline")  # TODO: replace with logging later

    if problem_type not in ["regression", "classification"]:
        raise ValueError("problem_type must be either 'regression' or 'classification'")

    if problem_type == "regression":
        estimator = Ridge()
    else:
        estimator = LogisticRegression(max_iter=500)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Swap the baseline estimator or add hyperparameters
    # Why: Model choice depends on your business objective, constraints, and evaluation metric
    # Examples:
    # 1. RandomForestRegressor / GradientBoosting / XGBoost (if allowed)
    # 2. LogisticRegression(C=0.5, class_weight="balanced")
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )

    model.fit(X_train, y_train)
    return model