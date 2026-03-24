"""
Main pipeline orchestration.

Usage:
    python -m src.main
"""

from pathlib import Path

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


def main() -> None:
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
    df_raw = load_data(str(raw_data_path))

    # 2. Clean
    target_column = target_cfg["column"]
    df_clean = clean_dataframe(df_raw, target_column=target_column)

    # 3. Validate
    validate_dataframe(
        df=df_clean,
        schema=schema_cfg,
        target_config=target_cfg,
    )

    # 4. Save cleaned data
    save_csv(df_clean, processed_data_path)

    # 5. Split
    X = df_clean.drop(columns=[target_column]).copy()
    y = df_clean[target_column].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=ml_cfg["test_size"],
        random_state=ml_cfg["random_state"],
        stratify=y if ml_cfg["problem_type"] == "classification" else None,
    )

    # 6. Preprocessor
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=features_cfg["quantile_bin"],
        categorical_onehot_cols=features_cfg["categorical_onehot"],
        numeric_passthrough_cols=features_cfg["numeric_passthrough"],
    )

    # 7. Train
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        problem_type=ml_cfg["problem_type"],
        param_grid=None,
    )

    # 8. Save model
    save_model(model, model_path)

    # 9. Evaluate
    metric_value = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        problem_type=ml_cfg["problem_type"],
    )

    # 10. Inference
    predictions = run_inference(model, X_test.iloc[:20], include_proba=True)
    save_csv(predictions.reset_index(drop=True), predictions_path)

    print("\n=== PIPELINE FINISHED ===")
    print(f"Metric ({ml_cfg['problem_type']}): {metric_value}")
    print(f"Saved cleaned data to: {processed_data_path}")
    print(f"Saved model to: {model_path}")
    print(f"Saved predictions to: {predictions_path}")


if __name__ == "__main__":
    main()