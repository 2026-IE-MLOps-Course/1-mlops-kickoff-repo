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
# Removed SETTINGS dictionary in favor of config.yaml


def _ensure_dirs(logger) -> None:
    for p in ["data/raw", "data/processed", "models", "reports", "logs"]:
        Path(p).mkdir(parents=True, exist_ok=True)
        logger.info("Ensured directory exists: %s", p)





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
            "Update config.yaml features to match your dataset."
        )

    for c in feature_cfg.get("quantile_bin", []):
        if not pd.api.types.is_numeric_dtype(X[c]):
            raise ValueError(
                f"Column '{c}' is configured for quantile binning "
                "but is not numeric. "
                "Fix cleaning or change config.yaml features quantile_bin."
            )

    return configured_feature_cols


def main() -> None:
    logger = get_logger("main")

    logger.info("Starting pipeline")
    try:
        _ensure_dirs(logger)

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
        logger.info("=== PIPELINE FINISHED ===")
        logger.info("Metric (%s): %s", ml_cfg['problem_type'], metric_value)
        logger.info("Saved cleaned data to: %s", processed_data_path)
        logger.info("Saved model to: %s", model_path)
        logger.info("Saved predictions to: %s", predictions_path)

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
