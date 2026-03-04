"""
Module: evaluate.py
-------------------
Role: Generate performance metrics to assess model quality before deployment.
Input: Fitted model (Pipeline) + Test dataset.
Output: Evaluation metric (ROC AUC for classification).
"""

import logging
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score
import pandas as pd

# Industry Grade: Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str) -> float:
    """
    Computes the final performance score for the model.

    Why ROC AUC? 
    In Churn prediction, we care about the model's ability to distinguish between 
    classes across all thresholds. AUC is more robust than Accuracy for 
    imbalanced datasets like Telco Churn.

    Args:
        model: Fitted scikit-learn Pipeline.
        X_test: Test features.
        y_test: True labels.
        problem_type: "classification" or "regression".

    Returns:
        float: The calculated metric (AUC or RMSE).
    """
    logger.info(f"Evaluating model for problem_type: {problem_type}")

    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test)

    if problem_type == "classification":
        try:
            # Logic from your 'New Final.ipynb' experiment
            # We take the probabilities of the positive class (Churn = Yes)
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_proba))

            logger.info(f"Evaluation successful. ROC AUC: {auc:.4f}")
            return auc

        except Exception as e:
            logger.error(
                f"Failed to calculate AUC: {e}. Falling back to F1 score.")
            y_pred = model.predict(X_test)
            return float(f1_score(y_test, y_pred, average="weighted"))

    elif problem_type == "regression":
        y_pred = model.predict(X_test)
        rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
        logger.info(f"Evaluation successful. RMSE: {rmse:.4f}")
        return rmse

    else:
        raise ValueError(
            "problem_type must be either 'regression' or 'classification'")
