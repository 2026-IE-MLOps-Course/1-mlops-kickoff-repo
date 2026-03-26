"""
Module: main.py
---------------
Role: Orchestrates the entire ML pipeline.

Pipeline steps:
1. Load raw data
2. Clean dataset
3. Validate schema
4. Feature engineering
5. Train/Test split
6. Model training
7. Model evaluation
8. Run inference and save predictions
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml
import pandas as pd
import wandb
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

from src.load_data import load_raw_data
from src.clean_data import clean_data
from src.validate import validate_dataframe
from src.features import engineer_features, get_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model
from src.infer import run_inference
from src.logger import configure_logging

logger = logging.getLogger(__name__)


# -----------------------------------------------------
# Config loader
# -----------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# -----------------------------------------------------
# W&B helpers
# -----------------------------------------------------

def _wandb_is_enabled(cfg: Dict[str, Any]) -> bool:
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return False
    return bool(wandb_cfg.get("enabled", False))


def _wandb_get_str(cfg: Dict[str, Any], key: str, default: str = "") -> str:
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return default
    value = wandb_cfg.get(key, default)
    return str(value).strip() if value is not None else default


def _wandb_get_bool(cfg: Dict[str, Any], key: str, default: bool = False) -> bool:
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return default
    return bool(wandb_cfg.get(key, default))


def _wandb_get_int(cfg: Dict[str, Any], key: str, default: int = 0) -> int:
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return default
    try:
        return int(wandb_cfg.get(key, default))
    except Exception:
        return default


def _wandb_get_list(cfg: Dict[str, Any], key: str) -> List[str]:
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return []
    value = wandb_cfg.get(key, [])
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if v is not None and str(v).strip()]


def _log_wandb_classification_artifacts(
    cfg: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model,
    stage_name: str,
) -> None:
    if not hasattr(model, "predict_proba"):
        logger.warning(
            "Skipping W&B classification artifacts: model has no predict_proba()"
        )
        return

    class_names = cfg.get("wandb", {}).get("class_names", None)
    y_probas = model.predict_proba(X_test)

    if _wandb_get_bool(cfg, "log_auc_plots"):
        wandb.log({
            f"plots/roc_curve_{stage_name}": wandb.plot.roc_curve(
                y_true=y_test.tolist(),
                y_probas=y_probas,
                labels=class_names,
                title=f"ROC Curve ({stage_name})",
            ),
            f"plots/pr_curve_{stage_name}": wandb.plot.pr_curve(
                y_true=y_test.tolist(),
                y_probas=y_probas,
                labels=class_names,
                title=f"Precision-Recall Curve ({stage_name})",
            ),
        })

    if _wandb_get_bool(cfg, "log_confusion_matrix"):
        y_pred = model.predict(X_test)
        wandb.log({
            f"plots/confusion_matrix_{stage_name}": wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_test.tolist(),
                preds=y_pred.tolist() if hasattr(y_pred, "tolist") else list(y_pred),
                class_names=class_names,
                title=f"Confusion Matrix ({stage_name})",
            )
        })


# -----------------------------------------------------
# Main Pipeline
# -----------------------------------------------------

def main() -> None:

    # Load .env for secrets (e.g. WANDB_API_KEY)
    load_dotenv(override=False)

    # Load configuration
    cfg = load_config()

    # Configure logging first
    configure_logging(
        log_level=cfg["logging"]["level"],
        log_file=Path(cfg["paths"]["log_file"]),
    )

    logger.info("Starting ML pipeline...")

    # Paths
    raw_path = Path(cfg["paths"]["raw_data"])
    clean_path = Path(cfg["paths"]["processed_data"])
    model_path = Path(cfg["paths"]["model_artifact"])
    preds_path = Path(cfg["paths"]["predictions_artifact"])

    # Validation config
    drop_na = cfg["validation"]["drop_na"]
    required_columns = cfg["validation"]["required_columns"]

    # Problem config
    target_column = cfg["problem"]["target_column"]
    problem_type = cfg["problem"]["problem_type"]
    identifier_column = cfg["problem"]["identifier_column"]

    # Split config
    test_size = cfg["split"]["test_size"]
    random_state = cfg["split"]["random_state"]

    # Model hyperparameters
    max_iter = cfg["training"]["classification"]["max_iter"]
    model_random_state = cfg["training"]["classification"]["random_state"]

    # Ensure folders exist
    for p in [clean_path, model_path, preds_path]:
        os.makedirs(p.parent, exist_ok=True)

    # -----------------------------------------------------
    # Initialize W&B
    # -----------------------------------------------------
    wandb_run = None
    if _wandb_is_enabled(cfg):
        wandb_project = _wandb_get_str(cfg, "project")
        if not wandb_project:
            raise ValueError("config.yaml: wandb.project must be a non-empty string when wandb.enabled is true")

        wandb_run = wandb.init(
            project=wandb_project,
            name=_wandb_get_str(cfg, "name") or None,
            job_type=_wandb_get_str(cfg, "job_type", default="training"),
            group=_wandb_get_str(cfg, "group") or None,
            notes=_wandb_get_str(cfg, "notes") or None,
            tags=_wandb_get_list(cfg, "tags") or None,
            config=cfg,
        )
        logger.info("W&B run initialized | project=%s | name=%s", wandb_project, wandb_run.name)
    else:
        logger.info("W&B disabled, continuing without experiment tracking")

    try:

        # -------------------------------------------------
        # STEP 1: Load Data
        # -------------------------------------------------

        logger.info("STEP 1: Loading raw data")
        df_raw = load_raw_data(raw_path)

        if wandb_run is not None:
            wandb.log({"data/raw_rows": int(df_raw.shape[0]), "data/raw_cols": int(df_raw.shape[1])})

        # -------------------------------------------------
        # STEP 2: Clean Data
        # -------------------------------------------------

        logger.info("STEP 2: Cleaning data")
        df_clean = clean_data(df_raw, drop_na=drop_na)

        df_clean.to_csv(clean_path, index=False)
        logger.info(f"Clean dataset saved to {clean_path}")

        if wandb_run is not None:
            wandb.log({"data/clean_rows": int(df_clean.shape[0]), "data/clean_cols": int(df_clean.shape[1])})

        # -------------------------------------------------
        # STEP 3: Validate Data
        # -------------------------------------------------

        logger.info("STEP 3: Validating dataset")
        validate_dataframe(df_clean, required_columns)

        # -------------------------------------------------
        # STEP 4: Feature Engineering
        # -------------------------------------------------

        logger.info("STEP 4: Feature engineering")

        df_eng = engineer_features(df_clean)

        X = df_eng.drop(columns=[target_column])
        y = df_eng[target_column].map({"No": 0, "Yes": 1})

        # -------------------------------------------------
        # STEP 5: Train/Test Split
        # -------------------------------------------------

        logger.info("STEP 5: Train/Test split")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state
        )

        # -------------------------------------------------
        # STEP 6: Preprocessor
        # -------------------------------------------------

        logger.info("STEP 6: Creating preprocessing pipeline")

        preprocessor = get_preprocessor(X_train)

        # -------------------------------------------------
        # STEP 7: Train Model
        # -------------------------------------------------

        logger.info("STEP 7: Training model")

        pipeline = train_model(
            X_train,
            y_train,
            preprocessor,
            model_path=str(model_path),
            max_iter=max_iter,
            random_state=model_random_state
        )

        # -------------------------------------------------
        # STEP 8: Evaluate Model
        # -------------------------------------------------

        logger.info("STEP 8: Evaluating model")

        metric_value = evaluate_model(
            pipeline,
            X_test,
            y_test,
            problem_type=problem_type
        )

        logger.info(f"Evaluation metric value: {metric_value:.4f}")

        if wandb_run is not None:
            wandb.log({"metrics/auc": metric_value})
            _log_wandb_classification_artifacts(cfg, X_test, y_test, pipeline, stage_name="test")

        # -------------------------------------------------
        # STEP 8.5: Log model artifact to W&B
        # -------------------------------------------------

        if wandb_run is not None:
            model_artifact_name = _wandb_get_str(cfg, "model_artifact_name", default="model")
            model_artifact = wandb.Artifact(
                name=model_artifact_name,
                type="model",
                description="Scikit-learn pipeline: preprocessor + LogisticRegression",
            )
            model_artifact.add_file(str(model_path))
            wandb.log_artifact(model_artifact)

        # -------------------------------------------------
        # STEP 9: Inference
        # -------------------------------------------------

        logger.info("STEP 9: Running inference")

        predictions_df = run_inference(pipeline, X_test)

        predictions_df.to_csv(preds_path, index=False)

        logger.info(f"Predictions saved to {preds_path}")

        if wandb_run is not None:
            if _wandb_get_bool(cfg, "log_predictions_table"):
                n_rows = _wandb_get_int(cfg, "predictions_table_rows", default=200)
                wandb.log({"tables/predictions_preview": wandb.Table(dataframe=predictions_df.head(n_rows))})

            if _wandb_get_bool(cfg, "log_predictions"):
                model_artifact_name = _wandb_get_str(cfg, "model_artifact_name", default="model")
                pred_artifact = wandb.Artifact(
                    name=f"{model_artifact_name}-predictions",
                    type="predictions",
                    description="Inference outputs from the pipeline",
                )
                pred_artifact.add_file(str(preds_path))
                wandb.log_artifact(pred_artifact)

        logger.info("Pipeline completed successfully!")

    except Exception:
        logger.exception("Pipeline failed")
        if wandb_run is not None:
            wandb.finish(exit_code=1)
        raise

    finally:
        if wandb_run is not None and wandb.run is not None:
            wandb.finish()


# -----------------------------------------------------
# Script Entry Point
# -----------------------------------------------------

if __name__ == "__main__":
    main()
