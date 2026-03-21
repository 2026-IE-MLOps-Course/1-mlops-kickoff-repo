# src/evaluate.py
"""
Evaluation — compute metrics on held-out data.
Returns a dictionary of metrics for W&B logging.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    problem_type: str,
) -> Dict[str, float]:
    """
    Evaluate a fitted model and return a dictionary of metrics.

    Returns dict like {"rmse": 123.4, "mae": 99.1, "r2": 0.85}
    """
    logger.info("Evaluating model — problem_type: %s", problem_type)

    if not hasattr(model, "predict"):
        raise TypeError(
            "The provided artifact does not have a .predict() method."
        )

    if len(X_eval) != len(y_eval):
        raise ValueError(
            f"Shape mismatch: X_eval has {len(X_eval)} rows, "
            f"y_eval has {len(y_eval)} rows."
        )

    y_pred = model.predict(X_eval)

    if problem_type == "classification":
        metric_val = float(f1_score(y_eval, y_pred, average="weighted"))
        metrics = {"f1_weighted": metric_val}
    elif problem_type == "regression":
        rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
        mae = float(mean_absolute_error(y_eval, y_pred))
        r2 = float(r2_score(y_eval, y_pred))
        metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    else:
        raise ValueError(
            f"Unknown problem_type '{problem_type}'. "
            "Use 'regression' or 'classification'."
        )

    logger.info("Metrics: %s", metrics)
    return metrics
