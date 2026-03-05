"""
Educational Goal:
- Why this module exists in an MLOps system: Evaluation logic must be isolated
  from training to prevent accidental model updates during scoring. Scoring
  happens strictly on untouched data (validation or test split).
- Responsibility (separation of concerns): Compute a single primary metric
  for the given problem_type. Return a float for automated experiment tracking.
- Pipeline contract (inputs and outputs):
  Input  — model (fitted Pipeline), X_test (DataFrame), y_test (Series),
           problem_type (str).
  Output — float: RMSE for regression, F1 (weighted) for classification.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str,
) -> float:
    """
    Inputs:
    - model: Fitted sklearn Pipeline (must have .predict()).
    - X_test (pd.DataFrame): Feature DataFrame for the split being evaluated.
    - y_test (pd.Series): True target values for the same split.
    - problem_type (str): "regression" or "classification".
    Outputs:
    - float: A single metric value — RMSE for regression, F1 for classification.
    Why this contract matters for reliable ML delivery:
    - Returning a standardised single float enables automated experiment tracking
      and objective deployment decisions. The same function works for both
      regression and classification, routed by the problem_type parameter.
    """
    print(f"[evaluate] Evaluating model — problem_type: {problem_type}")  # TODO: replace with logging later

    # Enforce duck-typing contract
    if not hasattr(model, "predict"):
        raise TypeError(
            "The provided artifact does not have a .predict() method. "
            "Ensure a valid sklearn Pipeline was saved."
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            f"Shape mismatch: X_test has {len(X_test)} rows, "
            f"y_test has {len(y_test)} rows."
        )

    y_pred = model.predict(X_test)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Different business contexts demand different metrics (e.g., recall
    #       for healthcare, MAE for pricing, AUC for ranking)
    # Examples:
    # 1. from sklearn.metrics import recall_score
    #    metric = recall_score(y_test, y_pred)
    # 2. from sklearn.metrics import mean_absolute_error
    #    metric = mean_absolute_error(y_test, y_pred)
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Student implementation for VoyageIQ travel dataset:
    if problem_type == "classification":
        metric = float(f1_score(y_test, y_pred, average="weighted"))
        metric_name = "F1 (weighted)"
    elif problem_type == "regression":
        metric = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        metric_name = "RMSE"
    else:
        raise ValueError(
            f"Unknown problem_type '{problem_type}'. "
            "Use 'regression' or 'classification'."
        )

    # Additional metrics printed for observability (students can extend)
    if problem_type == "regression":
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        print(  # TODO: replace with logging later
            f"[evaluate] {metric_name}: {metric:.4f} | "
            f"MAE: {mae:.4f} | R2: {r2:.4f}"
        )
    else:
        print(f"[evaluate] {metric_name}: {metric:.4f}")  # TODO: replace with logging later
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return metric
