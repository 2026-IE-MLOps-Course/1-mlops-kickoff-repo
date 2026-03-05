"""
Module: Model Training
----------------------
Role: Bundle preprocessing and algorithms into a single Pipeline and fit on training data.
Input: pandas.DataFrame (Processed) + ColumnTransformer (Recipe).
Output: Serialized scikit-learn Pipeline in `models/`.
"""
"""
Educational Goal:
- Why this module exists in an MLOps system: Training logic is centralised here
  so that .fit() is called exclusively on the training split. Bundling the
  preprocessor and model into a single Pipeline artifact prevents training-serving
  skew and guarantees identical transforms at inference time.
- Responsibility (separation of concerns): Accept an unfitted preprocessor,
  select an algorithm by problem_type, build a Pipeline, fit it on training data,
  and return the fitted Pipeline. No data loading, no evaluation.
- Pipeline contract (inputs and outputs):
  Input  — X_train (DataFrame), y_train (Series), preprocessor (ColumnTransformer),
           problem_type (str: "regression" or "classification").
  Output — Fitted sklearn Pipeline.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    problem_type: str,
):
    """
    Inputs:
    - X_train (pd.DataFrame): Training feature matrix.
    - y_train (pd.Series): Training target vector.
    - preprocessor: Unfitted ColumnTransformer from features.py.
    - problem_type (str): "regression" or "classification".
    Outputs:
    - sklearn.pipeline.Pipeline: A fitted pipeline (preprocessor + model).
    Why this contract matters for reliable ML delivery:
    - The Pipeline bundles feature rules and model weights into a single
      deployable artifact. Calling .fit() only on the training split
      structurally prevents feature math from leaking across splits.
    """
    print(f"[train] Training model — problem_type: {problem_type}")  # TODO: replace with logging later

    # Fail-fast guardrails
    if X_train.empty or y_train.empty:
        raise ValueError("Training data is empty — cannot train a model.")
    if len(X_train) != len(y_train):
        raise ValueError(
            f"Row mismatch: X_train has {len(X_train)} rows but "
            f"y_train has {len(y_train)} rows."
        )

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: The choice of algorithm is a core modelling decision. Students should
    #       experiment with different estimators (RF, GBM, XGBoost, etc.)
    # Examples:
    # 1. from sklearn.ensemble import RandomForestRegressor
    #    estimator = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42)
    # 2. from sklearn.ensemble import GradientBoostingClassifier
    #    estimator = GradientBoostingClassifier(n_estimators=100, random_state=42)
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Student implementation for VoyageIQ travel dataset:
    # Using RandomForest for regression (better for non-linear trip cost patterns)
    # and LogisticRegression as classification baseline
    if problem_type == "classification":
        estimator = LogisticRegression(max_iter=500, solver="liblinear")
    elif problem_type == "regression":
        # Baseline: Ridge()
        # Student override: RandomForestRegressor for better travel cost modelling
        from sklearn.ensemble import RandomForestRegressor
        estimator = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
        )
    else:
        raise ValueError(
            f"Unknown problem_type '{problem_type}'. "
            "Use 'regression' or 'classification'."
        )
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", estimator),
    ])

    print(f"[train] Fitting pipeline on {len(X_train)} training samples …")  # TODO: replace with logging later
    pipeline.fit(X_train, y_train)
    print("[train] Training complete.")  # TODO: replace with logging later

    return pipeline