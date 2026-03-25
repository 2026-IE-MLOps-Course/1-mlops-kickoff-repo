"""
Module: Inference
-----------------
Role: Download the promoted 'prod' model from W&B or use local model.
Input: New customer data (pandas.DataFrame).
Output: Churn predictions (numpy.ndarray) and probabilities.
"""

import os
import logging
import yaml
import joblib
import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def load_config(path="config.yaml"):
    """Load the central YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def download_prod_model(config=None):
    """
    Download the model artifact aliased 'prod' from W&B.
    """
    import wandb
    if config is None:
        config = load_config()

    load_dotenv()

    artifact_name = config["wandb"]["model_artifact_name"]
    model_alias = config["wandb"]["model_alias"]
    project = config["wandb"]["project"]

    api = wandb.Api()
    entity = api.default_entity
    artifact_path = f"{entity}/{project}/{artifact_name}:{model_alias}"

    logger.info("Downloading model artifact: %s", artifact_path)
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download()
    
    model_file = os.path.join(artifact_dir, "model.joblib")
    model = joblib.load(model_file)
    return model


def run_inference(
    model,
    X: pd.DataFrame,
    include_proba: bool = True
) -> pd.DataFrame:
    """
    Run inference using a trained sklearn-like model/pipeline.
    Returns predictions DataFrame.
    """
    preds = model.predict(X)

    out = pd.DataFrame(index=X.index)
    out["prediction"] = preds

    if include_proba and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        # binary classification → take positive class
        if hasattr(proba, "shape") and len(proba.shape) == 2 and proba.shape[1] >= 2:
            out["proba"] = proba[:, 1]
        else:
            out["proba"] = proba.ravel()

    return out


def run_inference_from_wandb(input_data, config=None):
    """
    Run churn prediction on new customer data using the W&B 'prod' model.
    """
    if config is None:
        config = load_config()

    model = download_prod_model(config)
    
    # We use the standardized run_inference for the actual work
    return run_inference(model, input_data, include_proba=True)
