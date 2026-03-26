"""
Module: Evaluation.
------------------
Role: Generate metrics and plots for model performance.
"""
import logging
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str) -> float:
    """
    Scores the fitted model on held-out test data.
    Why: Ensures objective assessment of model performance before deployment.
    """
    logger.info(f"Evaluando modelo para el problema tipo: {problem_type}")

    if not isinstance(y_test, pd.Series):
        y_test = pd.Series(y_test)

    if problem_type == "classification":
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, y_proba))
            logger.info(f"Evaluación exitosa. ROC AUC: {auc:.4f}")
            return auc
        except Exception as e:
            logger.error(f"Fallo al calcular AUC: {e}. Usando F1 score.")
            y_pred = model.predict(X_test)
            return float(f1_score(y_test, y_pred, average="weighted"))

    elif problem_type == "regression":
        y_pred = model.predict(X_test)
        rmse = float(mean_squared_error(y_test, y_pred) ** 0.5)
        logger.info(f"Evaluación exitosa. RMSE: {rmse:.4f}")
        return rmse

    raise ValueError("problem_type debe ser 'regression' o 'classification'")
