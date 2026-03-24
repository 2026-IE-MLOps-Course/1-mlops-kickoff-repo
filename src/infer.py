"""
Module: Inference
-----------------
Role: Download the promoted 'prod' model from W&B and make predictions.
Input: New customer data (pandas.DataFrame).
Output: Churn predictions (numpy.ndarray) and probabilities.

IMPORTANT: This module NEVER loads a local .joblib file directly.
           It always pulls the model artifact aliased 'prod' from W&B.
"""

import os
import logging
import yaml
import joblib
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_config(path="config.yaml"):
    """Load the central YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def download_prod_model(config=None):
    """
    Download the model artifact aliased 'prod' from W&B.

    Returns
    -------
    model : sklearn.pipeline.Pipeline
        The fitted pipeline (preprocessor + classifier).
    """
    if config is None:
        config = load_config()

    load_dotenv()

    artifact_name = config["wandb"]["model_artifact_name"]
    model_alias = config["wandb"]["model_alias"]
    project = config["wandb"]["project"]

    # Use the W&B API to fetch the artifact without starting a full run
    api = wandb.Api()

    # Build the full artifact path: entity/project/artifact:alias
    entity = api.default_entity
    artifact_path = f"{entity}/{project}/{artifact_name}:{model_alias}"

    logger.info("Downloading model artifact: %s", artifact_path)
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download()
    logger.info("Model downloaded to: %s", artifact_dir)

    model_file = os.path.join(artifact_dir, "model.joblib")
    model = joblib.load(model_file)
    logger.info("Model loaded successfully from W&B artifact.")

    return model


def run_inference(input_data, config=None):
    """
    Run churn prediction on new customer data using the W&B 'prod' model.

    Parameters
    ----------
    input_data : pd.DataFrame
        Customer data with the same schema as the training data
        (excluding the target column 'Churn').
    config : dict, optional
        Configuration dictionary. Loaded from config.yaml if not provided.

    Returns
    -------
    dict
        {
            "predictions": list of int (0 or 1),
            "probabilities": list of float (churn probability),
        }
    """
    if config is None:
        config = load_config()

    # Download the promoted 'prod' model from W&B
    model = download_prod_model(config)

    # Generate predictions
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1]

    logger.info(
        "Inference complete: %d samples, %d predicted churn",
        len(predictions),
        int(np.sum(predictions)),
    )

    return {
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
    }
