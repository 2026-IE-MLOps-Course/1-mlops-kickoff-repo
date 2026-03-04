"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Ensures objective, automated assessment of model performance before deployment.
- Responsibility (separation of concerns): Scoring the fitted model on held-out test data.
- Pipeline contract (inputs and outputs): Takes a fitted model and test data, returns a single float metric.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import numpy as np
from sklearn.metrics import mean_squared_error, f1_score
import pandas as pd


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str) -> float:
    """
    Inputs:
    - model: The fitted scikit-learn Pipeline.
    - X_test: Test features DataFrame.
    - y_test: Test target Series.
    - problem_type: "classification" or "regression".
    Outputs:
    - A float representing the evaluation metric (RMSE or F1).
    Why this contract matters for reliable ML delivery:
    - Returning a strict float allows ML tracking tools (like MLflow) to easily log and compare experiment performance.
    """
    print("Executing evaluate_model")  # TODO: replace with logging later

    predictions = model.predict(X_test)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Different business problems require different optimization metrics (e.g., Precision, Recall, MAE).
    # Examples:
    # 1. return float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
    # 2. return float(mean_absolute_error(y_test, predictions))
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    if problem_type == "classification":
        metric = float(f1_score(y_test, predictions, average="weighted"))
        # TODO: replace with logging later
        print(f"Evaluation complete. F1 Score (weighted): {metric:.4f}")
        return metric
    else:
        metric = float(np.sqrt(mean_squared_error(y_test, predictions)))
        # TODO: replace with logging later
        print(f"Evaluation complete. RMSE: {metric:.4f}")
        return metric
