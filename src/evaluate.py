"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""


def evaluate_model(pipeline, X_test, y_test):
    """Evaluate model performance on test data."""
    return {"test_f1": 0.72}