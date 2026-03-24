"""
Main pipeline orchestration.

Usage:
    python -m src.main
"""

from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.load_data import load_data
from src.clean_data import clean_dataframe
from src.validate import validate_dataframe
from src.features import get_feature_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model
from src.infer import run_inference
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
    try:
        _ensure_dirs(logger)
        _maybe_switch_to_telco(logger)

        cfg = load_config()

        paths_cfg = cfg["paths"]
        ml_cfg = cfg["ml"]
        features_cfg = cfg["features"]
        schema_cfg = cfg["schema"]
        target_cfg = cfg["target_config"]

        raw_data_path = Path(paths_cfg["raw_data"])
        processed_data_path = Path(paths_cfg["processed_data"])
        model_path = Path(paths_cfg["model_path"])
        predictions_path = Path(paths_cfg["predictions_path"])

        model_path.parent.mkdir(parents=True, exist_ok=True)
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        processed_data_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Load
        logger.info("Loading raw data from: %s", raw_data_path)
        df_raw = load_data(str(raw_data_path))
        logger.info("Raw shape: %s", df_raw.shape)

        # 2. Clean
        target_column = target_cfg["column"]
        logger.info("Cleaning data (target=%s)", target_column)
        df_clean = clean_dataframe(df_raw, target_column=target_column)
        logger.info("Clean shape: %s", df_clean.shape)

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

        # 8. Save model
        logger.info("Saving model artifact to: %s", model_path)
        save_model(model, model_path)

        # 9. Evaluate
        logger.info("Evaluating on TEST split")
        metric_value = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            problem_type=ml_cfg["problem_type"],
        )
        logger.info("Validation metric = %s", metric_value)

        # 10. Inference
        logger.info("Running inference on TEST sample and saving predictions")
        predictions = run_inference(model, X_test.iloc[:20], include_proba=True)
        save_csv(predictions.reset_index(drop=True), predictions_path)

        logger.info("Pipeline complete ✅")
        print("\n=== PIPELINE FINISHED ===")
        print(f"Metric ({ml_cfg['problem_type']}): {metric_value}")
        print(f"Saved cleaned data to: {processed_data_path}")
        print(f"Saved model to: {model_path}")
        print(f"Saved predictions to: {predictions_path}")

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
