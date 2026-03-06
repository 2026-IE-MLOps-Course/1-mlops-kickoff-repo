"""
Module: Evaluation
--------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""

import json
import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger("voyageiq.evaluate")


def _mean_absolute_percentage_error(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Compute MAPE, guarding against zero-division.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        float: MAPE as a percentage.
    """
    mask = y_true != 0
    if mask.sum() == 0:
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


_METRIC_FUNCTIONS = {
    "rmse": lambda y, p: float(np.sqrt(mean_squared_error(y, p))),
    "mae": lambda y, p: float(mean_absolute_error(y, p)),
    "r2": lambda y, p: float(r2_score(y, p)),
    "mape": _mean_absolute_percentage_error,
}


def evaluate_model(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    config: dict,
    split_name: str = "validation",
) -> dict:
    """Score the pipeline on the given data split and persist metrics.

    Args:
        pipeline: Fitted sklearn Pipeline (must have ``.predict()``).
        X: Feature DataFrame for the split being evaluated.
        y: True target values for the same split.
        config: Full pipeline configuration dictionary.
        split_name: Label used in logs and the report (e.g. 'validation', 'test').

    Returns:
        dict: Mapping of metric name → float value.

    Raises:
        TypeError: If the pipeline lacks a predict method.
        ValueError: If X and y have different row counts.
    """
    # Enforce duck-typing contract
    if not hasattr(pipeline, "predict"):
        raise TypeError(
            "The provided artifact does not have a .predict() method. "
            "Ensure a valid sklearn Pipeline was saved."
        )

    if len(X) != len(y):
        raise ValueError(
            f"Shape mismatch: X has {len(X)} rows, y has {len(y)} rows."
        )

    y_pred = pipeline.predict(X)
    y_true = np.array(y)

    eval_cfg = config.get("evaluation", {})
    metric_names = eval_cfg.get("metrics", ["rmse", "mae", "r2"])

    results: dict = {}
    for name in metric_names:
        func = _METRIC_FUNCTIONS.get(name)
        if func is None:
            logger.warning("Unknown metric '%s' — skipping.", name)
            continue
        results[name] = round(func(y_true, y_pred), 4)

    # Log results
    logger.info(
        "Evaluation on %s split — %s",
        split_name,
        ", ".join(f"{k}: {v}" for k, v in results.items()),
    )

    # Persist to JSON — accumulate results across splits
    metrics_path = config.get("paths", {}).get(
        "metrics_output", "reports/metrics.json"
    )
    metrics_dir = os.path.dirname(metrics_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    # Load existing report if present, so we keep all splits
    all_metrics = {}
    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as fh:
                all_metrics = json.load(fh)
        except (json.JSONDecodeError, OSError):
            all_metrics = {}

    all_metrics[split_name] = results

    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(all_metrics, fh, indent=2)
    logger.info("Metrics saved to %s", metrics_path)

    return results
