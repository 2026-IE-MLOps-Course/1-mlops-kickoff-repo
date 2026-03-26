"""
Main pipeline orchestration.

Usage:
    python -m src.main
"""

from pathlib import Path
import sys

import pandas as pd
import wandb
import joblib
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.load_data import load_data
from src.clean_data import clean_dataframe
from src.validate import validate_dataframe
from src.features import get_feature_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import save_model, save_csv
from src.logger import get_logger


# ========================================================
# CONFIGURATION (SETTINGS dictionary bridge)
# ========================================================
SETTINGS = {
    "is_example_config": True,
    "problem_type": "classification",  # "regression" or "classification"
    "target_column": "target",
    "raw_data_path": "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "processed_data_path": "data/processed/clean.csv",
    "model_path": "models/model.joblib",  # pickle content acceptable
    "predictions_path": "reports/predictions.csv",
    "random_state": 42,
    # 3-way split (enforced early)
    "test_size": 0.2,
    "val_size": 0.2,  # % of the *remaining train* after test split
    "features": {
        "quantile_bin": ["num_feature"],
        "categorical_onehot": ["cat_feature"],
        "numeric_passthrough": [],
        "n_bins": 3,
    },

}


def _ensure_dirs(logger) -> None:
    for p in ["data/raw", "data/processed", "models", "reports", "logs"]:
        Path(p).mkdir(parents=True, exist_ok=True)
        logger.info("Ensured directory exists: %s", p)


def _maybe_switch_to_telco(logger) -> None:
    telco_repo_path = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    telco_alt_path = Path("/mnt/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    if (not telco_repo_path.exists()) and telco_alt_path.exists():
        logger.info("Found Telco CSV at /mnt/data; copying into data/raw")
        df_alt = pd.read_csv(telco_alt_path)
        save_csv(df_alt, telco_repo_path)

    if telco_repo_path.exists():
        logger.info(
            "Telco dataset detected. Switching SETTINGS to Telco schema."
        )
        SETTINGS["is_example_config"] = False
        SETTINGS["problem_type"] = "classification"
        SETTINGS["target_column"] = "Churn"
        SETTINGS["raw_data_path"] = str(telco_repo_path)
        SETTINGS["features"] = {
            "quantile_bin": ["tenure", "MonthlyCharges", "TotalCharges"],
            "numeric_passthrough": ["SeniorCitizen"],
            "categorical_onehot": [
                "gender",
                "Partner",
                "Dependents",
                "PhoneService",
                "MultipleLines",
                "InternetService",
                "OnlineSecurity",
                "OnlineBackup",
                "DeviceProtection",
                "TechSupport",
                "StreamingTV",
                "StreamingMovies",
                "Contract",
                "PaperlessBilling",
                "PaymentMethod",
            ],
            "n_bins": 5,
        }
        SETTINGS["schema"] = {
            "gender": {'type': 'categorical', 'accept_nan': False},
            "SeniorCitizen": {'type': 'numeric', 'accept_nan': False},
            "Partner": {'type': 'categorical', 'accept_nan': False},
            "Dependents": {'type': 'categorical', 'accept_nan': False},
            "tenure": {'type': 'numeric', 'accept_nan': False},
            "PhoneService": {'type': 'categorical', 'accept_nan': False},
            "MultipleLines": {'type': 'categorical', 'accept_nan': False},
            "InternetService": {'type': 'categorical', 'accept_nan': False},
            "OnlineSecurity": {'type': 'categorical', 'accept_nan': False},
            "OnlineBackup": {'type': 'categorical', 'accept_nan': False},
            "DeviceProtection": {'type': 'categorical', 'accept_nan': False},
            "TechSupport": {'type': 'categorical', 'accept_nan': False},
            "StreamingTV": {'type': 'categorical', 'accept_nan': False},
            "StreamingMovies": {'type': 'categorical', 'accept_nan': False},
            "Contract": {'type': 'categorical', 'accept_nan': False},
            "PaperlessBilling": {'type': 'categorical', 'accept_nan': False},
            "PaymentMethod": {'type': 'categorical', 'accept_nan': False},
            "MonthlyCharges": {'type': 'numeric', 'accept_nan': False},
            "TotalCharges": {'type': 'numeric', 'accept_nan': False},
        }
        SETTINGS["target_config"] = {
            'column': 'Churn',
            'type': 'classification',
            'allowed_classes': [1, 0]
        }


def _fail_fast_feature_checks(
    X: pd.DataFrame,
    target_col: str,
    feature_cfg: dict,
) -> list[str]:
    configured_feature_cols = (
        feature_cfg.get("quantile_bin", [])
        + feature_cfg.get("categorical_onehot", [])
        + feature_cfg.get("numeric_passthrough", [])
    )

    if target_col not in X.columns and target_col in X.columns:
        # defensive (normally unreachable)
        raise ValueError(f"Target column '{target_col}' should not be in X.")

    missing_in_X = [c for c in configured_feature_cols if c not in X.columns]
    if missing_in_X:
        raise ValueError(
            "Configured feature columns missing in X. "
            f"Missing: {missing_in_X}. "
            "Update SETTINGS['features'] to match your dataset."
        )

    for c in feature_cfg.get("quantile_bin", []):
        if not pd.api.types.is_numeric_dtype(X[c]):
            raise ValueError(
                f"Column '{c}' is configured for quantile binning "
                "but is not numeric. "
                "Fix cleaning or change SETTINGS['features']['quantile_bin']."
            )

    return configured_feature_cols


def main() -> None:
    logger = get_logger("main")

    logger.info("Starting pipeline")
    run = None
    try:
        _ensure_dirs(logger)
        _maybe_switch_to_telco(logger)

        cfg = load_config()

        paths_cfg = cfg["paths"]
        ml_cfg = cfg["ml"]
        features_cfg = cfg["features"]
        schema_cfg = cfg["schema"]
        target_cfg = cfg["target_config"]
        wandb_cfg = cfg.get("wandb", {})

        raw_data_path = Path(paths_cfg["raw_data"])
        processed_data_path = Path(paths_cfg["processed_data"])
        model_path = Path(paths_cfg["model_path"])
        predictions_path = Path(paths_cfg["predictions_path"])

        model_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        processed_data_path.parent.mkdir(parents=True, exist_ok=True)

        # ── W&B: Initialize run ──────────────────────────────────
        run = wandb.init(
            project=wandb_cfg.get("project", "mlops-churn-prediction"),
            job_type=wandb_cfg.get("job_type", "training-pipeline"),
            config={
                "problem_type": ml_cfg["problem_type"],
                "test_size": ml_cfg["test_size"],
                "random_state": ml_cfg["random_state"],
                "raw_data_path": str(raw_data_path),
                "features": features_cfg,
            },
        )
        logger.info("W&B run initialized: %s", run.name)

        # 1. Load
        logger.info("Loading raw data from: %s", raw_data_path)
        df_raw = load_data(str(raw_data_path))
        logger.info("Raw shape: %s", df_raw.shape)

        # W&B: Log data stats
        wandb.log({
            "data/raw_rows": df_raw.shape[0],
            "data/raw_columns": df_raw.shape[1],
        })

        # 2. Clean
        target_column = target_cfg["column"]
        logger.info("Cleaning data (target=%s)", target_column)
        df_clean = clean_dataframe(df_raw, target_column=target_column)
        logger.info("Clean shape: %s", df_clean.shape)

        wandb.log({
            "data/clean_rows": df_clean.shape[0],
            "data/clean_columns": df_clean.shape[1],
            "data/rows_dropped": df_raw.shape[0] - df_clean.shape[0],
        })

        # 3. Validate
        logger.info("Validating required columns against schema")
        validate_dataframe(
            df=df_clean,
            schema=schema_cfg,
            target_config=target_cfg,
        )

        # 4. Save cleaned data
        logger.info("Saving processed data to: %s", processed_data_path)
        save_csv(df_clean, processed_data_path)

        # 5. Split
        logger.info("Splitting data into Train/Test")
        X = df_clean.drop(columns=[target_column]).copy()
        y = df_clean[target_column].copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=ml_cfg["test_size"],
            random_state=ml_cfg["random_state"],
            stratify=y if ml_cfg["problem_type"] == "classification" else None,
        )
        logger.info("Split sizes -> train=%d, test=%d", len(X_train), len(X_test))

        wandb.log({
            "data/train_size": len(X_train),
            "data/test_size": len(X_test),
        })

        # 6. Preprocessor
        logger.info("Building feature preprocessor (unfitted)")
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=features_cfg["quantile_bin"],
            categorical_onehot_cols=features_cfg["categorical_onehot"],
            numeric_passthrough_cols=features_cfg["numeric_passthrough"],
        )

        # 7. Train
        logger.info("Training model (fit only on TRAIN split)")
        model = train_model(
            X_train=X_train,
            y_train=y_train,
            preprocessor=preprocessor,
            problem_type=ml_cfg["problem_type"],
            param_grid=None,
        )

        # 8. Save model locally
        logger.info("Saving model artifact to: %s", model_path)
        save_model(model, model_path)

        # ── W&B: Upload model artifact ───────────────────────────
        artifact_name = wandb_cfg.get("model_artifact_name", "churn-model")
        model_alias = wandb_cfg.get("model_alias", "prod")

        model_artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=f"Trained {ml_cfg['problem_type']} pipeline",
        )
        model_artifact.add_file(str(model_path))
        wandb.log_artifact(model_artifact, aliases=["latest", model_alias])
        logger.info("Model artifact '%s' uploaded to W&B with alias '%s'",
                     artifact_name, model_alias)

        # 9. Evaluate
        logger.info("Evaluating on TEST split")
        metric_value = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            problem_type=ml_cfg["problem_type"],
        )
        logger.info("Validation metric = %s", metric_value)

        # W&B: Log evaluation metrics
        metric_label = "f1" if ml_cfg["problem_type"] == "classification" else "rmse"
        wandb.log({f"eval/{metric_label}": metric_value})

        # 10. Inference sample
        logger.info("Running sample inference on TEST split (first 20 rows)")
        y_pred_sample = model.predict(X_test.iloc[:20])
        preds_df = pd.DataFrame({"prediction": y_pred_sample}, index=X_test.iloc[:20].index)
        if hasattr(model, "predict_proba"):
            preds_df["proba"] = model.predict_proba(X_test.iloc[:20])[:, 1]
        save_csv(preds_df.reset_index(drop=True), predictions_path)

        # W&B: Log predictions table
        wandb.log({
            "predictions_sample": wandb.Table(dataframe=preds_df.reset_index(drop=True))
        })

        logger.info("Pipeline complete")
        print("\n=== PIPELINE FINISHED ===")
        print(f"Metric ({ml_cfg['problem_type']}): {metric_value}")
        print(f"Saved cleaned data to: {processed_data_path}")
        print(f"Saved model to: {model_path}")
        print(f"Saved predictions to: {predictions_path}")
        print(f"W&B run: {run.url}")

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)
    finally:
        # W&B: Always close the run (even on failure)
        if run is not None:
            wandb.finish()
            logger.info("W&B run finished.")


if __name__ == "__main__":
    main()
