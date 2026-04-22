# src/train.py
"""
Model training — bundle preprocessing and algorithm into a single Pipeline.
"""

import logging

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    problem_type: str,
):
    """Train and return a fitted sklearn Pipeline."""
    logger.info("Training model — problem_type: %s", problem_type)

    if X_train.empty or y_train.empty:
        raise ValueError("Training data is empty — cannot train a model.")
    if len(X_train) != len(y_train):
        raise ValueError(
            f"Row mismatch: X_train has {len(X_train)} rows but "
            f"y_train has {len(y_train)} rows."
        )

    if problem_type == "classification":
        estimator = LogisticRegression(max_iter=500, solver="liblinear")
    elif problem_type == "regression":
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

    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", estimator),
    ])

    logger.info("Fitting pipeline on %d training samples …", len(X_train))
    pipeline.fit(X_train, y_train)
    logger.info("Training complete.")

    return pipeline
