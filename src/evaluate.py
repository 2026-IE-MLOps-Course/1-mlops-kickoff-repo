"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Evaluation provides an objective gate for model quality before promotion.
- Responsibility (separation of concerns): Compute a single scalar metric based on problem type (regression vs classification).
- Pipeline contract (inputs and outputs): Input fitted Pipeline + test split; output float metric (RMSE or weighted F1).

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import math

import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str) -> float:
    """
    Inputs:
    - model: Fitted scikit-learn Pipeline (preprocess + estimator).
    - X_test: Held-out features.
    - y_test: Held-out target.
    - problem_type: "regression" or "classification".
    Outputs:
    - metric_value: float (RMSE for regression, weighted F1 for classification).
    Why this contract matters for reliable ML delivery:
    - A consistent numeric score enables automation (CI gates, model registry thresholds) and reduces subjective decisions.
    """
    print(f"[evaluate.evaluate_model] Evaluating model as problem_type='{problem_type}'")  # TODO: replace with logging later

    y_pred = model.predict(X_test)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Customize metrics (MAE, ROC-AUC, precision/recall) and add business-relevant slicing
    # Why: Different business goals require different tradeoffs; one metric rarely captures full performance
    # Examples:
    # 1. Use MAE for robustness to outliers
    # 2. Compute F1 for a specific positive class if binary classification
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    if problem_type == "regression":
        mse = mean_squared_error(y_test, y_pred)
        rmse = math.sqrt(mse)
        return float(rmse)

    if problem_type == "classification":
        score = f1_score(y_test, y_pred, average="weighted")
        return float(score)

    raise ValueError("problem_type must be either 'regression' or 'classification'")