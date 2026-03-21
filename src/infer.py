# src/infer.py
"""
Inference — accept a fitted model and new data, return predictions.
No training, no evaluation, no file I/O.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def run_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Run predictions on new data.
    Returns a DataFrame with a single column 'prediction'.
    """
    logger.info("Running inference on %d samples …", len(X_infer))

    if not isinstance(X_infer, pd.DataFrame):
        raise TypeError(
            "Inference input must be a pandas DataFrame, "
            f"got {type(X_infer).__name__}."
        )
    if not hasattr(model, "predict"):
        raise TypeError(
            "The provided artifact does not have a .predict() method."
        )

    predictions = model.predict(X_infer)

    result = pd.DataFrame(
        {"prediction": predictions},
        index=X_infer.index,
    )

    logger.info("Inference complete — %d predictions generated.", len(result))
    return result
