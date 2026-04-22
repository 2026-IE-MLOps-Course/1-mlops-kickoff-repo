# src/main.py
"""
Pipeline orchestrator — single entry point.

Owns:
- Loading and validating config.yaml
- Resolving repo-relative paths
- Initialising W&B experiment tracking
- Orchestrating: load → clean → validate → split → recipe → train → evaluate → save → infer
- Logging all W&B metrics, artifacts, and promoted model
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split

import wandb

from src.logger import configure_logging

from src.clean_data import clean_dataframe
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.train import train_model
from src.utils import save_csv, save_model

logger = logging.getLogger(__name__)


# ─────────────────────────────────────
# Config helpers
# ─────────────────────────────────────

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration from disk."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("config.yaml must parse into a dictionary at the top level")

    return cfg


def resolve_repo_path(project_root: Path, relative_path: str) -> Path:
    """Resolve a config path relative to the repo root."""
    return project_root / relative_path.strip()


# ─────────────────────────────────────
# W&B helpers
# ─────────────────────────────────────

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


# ─────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────

def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    # Load .env from repo root
    load_dotenv(dotenv_path=project_root / ".env", override=False)

    # ── Load config.yaml ──
    cfg = load_config(project_root / "config.yaml")

    # ── Configure logging ──
    log_file_path = resolve_repo_path(project_root, cfg["paths"]["log_file"])
    configure_logging(
        log_level=cfg.get("logging", {}).get("level", "INFO"),
        log_file=log_file_path,
    )

    # ── Initialise W&B ──
    wandb_run = None
    if _wandb_is_enabled(cfg):
        wandb_project = _wandb_get_str(cfg, "project")
        if not wandb_project:
            raise ValueError(
                "config.yaml: wandb.project must be a non-empty string when wandb.enabled is true"
            )

        wandb_run = wandb.init(
            project=wandb_project,
            config=cfg,
            job_type="training-pipeline",
        )
        logger.info("Initialised W&B run | name=%s | project=%s", wandb_run.name, wandb_project)
    else:
        logger.info("W&B disabled, continuing without experiment tracking")

    try:
        logger.info("=" * 60)
        logger.info("Pipeline started")
        logger.info("=" * 60)

        # ── Extract settings from config ──
        paths_cfg = cfg["paths"]
        features_cfg = cfg["features"]
        splitting_cfg = cfg["splitting"]
        target_col = features_cfg["target"]
        problem_type = cfg["project"]["problem_type"]

        raw_data_path = resolve_repo_path(project_root, paths_cfg["raw_data"])
        processed_data_path = resolve_repo_path(project_root, paths_cfg["processed_data"])
        model_artifact_path = resolve_repo_path(project_root, paths_cfg["model_artifact"])
        predictions_path = resolve_repo_path(project_root, paths_cfg["predictions_output"])

        seed = splitting_cfg["random_seed"]
        test_size = splitting_cfg["test_size"]
        val_size = splitting_cfg["val_size"]

        # Feature columns
        numeric_passthrough_cols = features_cfg.get("numeric_features", []) + features_cfg.get("date_derived_features", [])
        categorical_onehot_cols = features_cfg.get("categorical_features", [])
        quantile_bin_cols = features_cfg.get("quantile_bin", [])
        n_bins = features_cfg.get("n_bins", 3)

        # ── 1. Create output directories ──
        for path_key in ["processed_data", "model_artifact", "predictions_output", "log_file"]:
            resolve_repo_path(project_root, paths_cfg[path_key]).parent.mkdir(parents=True, exist_ok=True)

        # ── 2. Load raw data ──
        logger.info("STEP 1 — Loading raw data")
        df_raw = load_raw_data(raw_data_path)

        if wandb_run is not None:
            wandb.log({"data/raw_rows": int(df_raw.shape[0]), "data/raw_cols": int(df_raw.shape[1])})

        # ── 3. Clean data ──
        logger.info("STEP 2 — Cleaning data")
        df_clean = clean_dataframe(df_raw, target_col)

        if wandb_run is not None:
            wandb.log({"data/clean_rows": int(df_clean.shape[0]), "data/clean_cols": int(df_clean.shape[1])})

        # ── 4. Save processed CSV ──
        logger.info("Saving processed data …")
        save_csv(df_clean, processed_data_path)

        # ── 5. Validate data ──
        logger.info("STEP 3 — Validating data")
        from src.validate import validate_dataframe

        required_columns = (
            numeric_passthrough_cols
            + categorical_onehot_cols
            + quantile_bin_cols
            + [target_col]
        )
        validate_dataframe(df_clean, required_columns)

        # ── 6. Train / Val / Test split ──
        logger.info("STEP 4 — Splitting data (train/val/test)")
        keep_cols = (
            numeric_passthrough_cols
            + categorical_onehot_cols
            + quantile_bin_cols
        )
        keep_cols = [c for c in keep_cols if c in df_clean.columns]

        X = df_clean[keep_cols]
        y = df_clean[target_col]

        stratify_arg = y if problem_type == "classification" else None
        try:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed, stratify=stratify_arg,
            )
        except ValueError:
            logger.warning("Stratification failed, falling back to unstratified split.")
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=seed,
            )

        relative_val = val_size / (1 - test_size)
        stratify_arg = y_temp if problem_type == "classification" else None
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=relative_val, random_state=seed, stratify=stratify_arg,
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=relative_val, random_state=seed,
            )

        logger.info("Split sizes — train: %d, val: %d, test: %d", len(X_train), len(X_val), len(X_test))

        # ── 7. Fail-fast feature checks ──
        logger.info("STEP 5 — Checking feature columns …")
        all_configured_cols = quantile_bin_cols + categorical_onehot_cols + numeric_passthrough_cols
        missing_cols = [c for c in all_configured_cols if c not in X_train.columns]
        if missing_cols:
            raise ValueError(f"Configured feature columns missing from training data: {missing_cols}")

        # ── 8. Build feature recipe ──
        logger.info("STEP 6 — Building feature preprocessor recipe")
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=quantile_bin_cols,
            categorical_onehot_cols=categorical_onehot_cols,
            numeric_passthrough_cols=numeric_passthrough_cols,
            n_bins=n_bins,
        )

        # ── 9. Train model ──
        logger.info("STEP 7 — Training model")
        pipeline = train_model(X_train, y_train, preprocessor, problem_type)

        # ── 10. Evaluate on validation split ──
        logger.info("STEP 8 — Evaluating on validation split")
        val_metrics = evaluate_model(pipeline, X_val, y_val, problem_type)
        logger.info("Validation metrics: %s", val_metrics)

        if wandb_run is not None:
            wandb.log({f"metrics/val/{k}": float(v) for k, v in val_metrics.items()})

        # ── 11. Evaluate on test split ──
        logger.info("STEP 9 — Evaluating on test split")
        test_metrics = evaluate_model(pipeline, X_test, y_test, problem_type)
        logger.info("Test metrics: %s", test_metrics)

        if wandb_run is not None:
            wandb.log({f"metrics/test/{k}": float(v) for k, v in test_metrics.items()})

        # ── 12. Save model artifact ──
        logger.info("Saving model artifact …")
        save_model(pipeline, model_artifact_path)

        # Log model artifact to W&B
        if wandb_run is not None:
            model_artifact_name = _wandb_get_str(cfg, "model_artifact_name", default="voyageiq_model")
            model_artifact = wandb.Artifact(
                name=model_artifact_name,
                type="model",
                description="Scikit-learn pipeline (preprocessing + estimator)",
            )
            model_artifact.add_file(str(model_artifact_path))
            wandb.log_artifact(model_artifact)
            logger.info("Model artifact logged to W&B: %s", model_artifact_name)

            # Optionally log processed data
            if _wandb_get_bool(cfg, "log_processed_data", default=False):
                data_artifact = wandb.Artifact(
                    name=f"{model_artifact_name}-processed-data",
                    type="dataset",
                    description="Processed training dataset",
                )
                data_artifact.add_file(str(processed_data_path))
                wandb.log_artifact(data_artifact)

        # ── 13. Inference on test split (demonstration) ──
        logger.info("Running inference on test split …")
        preds = run_inference(pipeline, X_test)
        save_csv(preds, predictions_path)

        # Log predictions artifact to W&B
        if wandb_run is not None and _wandb_get_bool(cfg, "log_predictions", default=False):
            model_artifact_name = _wandb_get_str(cfg, "model_artifact_name", default="voyageiq_model")
            pred_artifact = wandb.Artifact(
                name=f"{model_artifact_name}-predictions",
                type="predictions",
                description="Inference outputs from the pipeline",
            )
            pred_artifact.add_file(str(predictions_path))
            wandb.log_artifact(pred_artifact)

        logger.info("=" * 60)
        logger.info("Pipeline completed successfully.")
        logger.info("=" * 60)

    except Exception:
        logger.exception("Pipeline failed")
        if wandb_run is not None:
            wandb.finish(exit_code=1)
        raise

    finally:
        if wandb_run is not None and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
