"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow (Load -> Clean -> Validate -> Train -> Evaluate).
      Owns W&B lifecycle: init, config logging, metric logging, artifact logging.
Usage: python -m src.main
"""

import os
import logging
import yaml
import wandb
from dotenv import load_dotenv

from src.load_data import load_data
from src.clean_data import clean_dataframe
from src.validate import validate_dataframe
from src.features import get_feature_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model

# --------------------------------------------------------
# Logger (replaces print — professor requirement)
# --------------------------------------------------------
logger = logging.getLogger(__name__)


def load_config(path="config.yaml"):
    """Load the central YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline():
    """
    End-to-end pipeline orchestrator.
    W&B is initialized and finished here — nowhere else.
    """

    # --------------------------------------------------
    # 0. Load environment variables and config
    # --------------------------------------------------
    load_dotenv()
    config = load_config()

    # --------------------------------------------------
    # 1. Initialize W&B (centrally, from main.py only)
    # --------------------------------------------------
    wandb.init(
        project=config["wandb"]["project"],
        job_type=config["wandb"]["job_type"],
        config={
            "test_size": config["train"]["test_size"],
            "seed": config["train"]["seed"],
            "raw_data_path": config["data"]["raw"],
        },
    )
    logger.info("W&B run initialized: %s", wandb.run.name)

    try:
        # --------------------------------------------------
        # 2. Load raw data
        # --------------------------------------------------
        logger.info("Loading raw data...")
        df_raw = load_data(config["data"]["raw"])
        logger.info("Raw data loaded: %d rows, %d columns", *df_raw.shape)

        wandb.log({
            "data/raw_rows": df_raw.shape[0],
            "data/raw_columns": df_raw.shape[1],
        })

        # --------------------------------------------------
        # 3. Clean data
        # --------------------------------------------------
        logger.info("Cleaning data...")
        df_clean = clean_dataframe(df_raw)
        logger.info("Clean data: %d rows, %d columns", *df_clean.shape)

        # Save processed CSV
        os.makedirs(os.path.dirname(config["data"]["processed"]), exist_ok=True)
        df_clean.to_csv(config["data"]["processed"], index=False)

        # --------------------------------------------------
        # 4. Validate data
        # --------------------------------------------------
        logger.info("Validating data schema...")
        validate_dataframe(df_clean)
        logger.info("Validation passed.")

        # --------------------------------------------------
        # 5. Feature engineering + Train
        # --------------------------------------------------
        logger.info("Training model...")
        pipeline, metrics, X_test, y_test = train_model(
            df_clean, config, get_feature_preprocessor
        )

        # Log hyperparameters that the model was trained with
        wandb.config.update({
            "model_type": type(pipeline.named_steps["classifier"]).__name__,
        }, allow_val_change=True)

        # --------------------------------------------------
        # 6. Evaluate
        # --------------------------------------------------
        logger.info("Evaluating model...")
        eval_metrics = evaluate_model(pipeline, X_test, y_test)

        # Merge training + evaluation metrics
        all_metrics = {**metrics, **eval_metrics}

        # Log all metrics to W&B
        wandb.log(all_metrics)
        logger.info("Metrics logged to W&B: %s", all_metrics)

        # --------------------------------------------------
        # 7. Log model artifact to W&B with 'prod' alias
        # --------------------------------------------------
        model_path = os.path.join("models", "model.joblib")
        artifact_name = config["wandb"]["model_artifact_name"]
        model_alias = config["wandb"]["model_alias"]

        artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description="XGBoost churn prediction pipeline (preprocessor + classifier)",
            metadata=all_metrics,
        )
        artifact.add_file(model_path)
        wandb.log_artifact(artifact, aliases=["latest", model_alias])
        logger.info(
            "Model artifact '%s' logged with alias '%s'",
            artifact_name, model_alias,
        )

        # --------------------------------------------------
        # 8. Log pipeline logs as artifact
        # --------------------------------------------------
        log_file = os.path.join("logs", "main.log")
        if os.path.exists(log_file):
            log_artifact = wandb.Artifact(
                name="pipeline-logs",
                type="logs",
                description="Pipeline execution logs for this run",
            )
            log_artifact.add_file(log_file)
            wandb.log_artifact(log_artifact)
            logger.info("Log artifact uploaded to W&B.")

    except Exception:
        logger.exception("Pipeline failed.")
        raise

    finally:
        # --------------------------------------------------
        # 9. Close W&B run cleanly
        # --------------------------------------------------
        wandb.finish()
        logger.info("W&B run finished.")


# --------------------------------------------------------
# Entry point
# --------------------------------------------------------
if __name__ == "__main__":
    # Configure root logger: console + file (dual output)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join("logs", "main.log"), mode="w"),
        ],
    )
    run_pipeline()
