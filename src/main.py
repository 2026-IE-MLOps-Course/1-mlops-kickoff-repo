"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow (Load -> Clean -> Validate -> Train -> Evaluate).
Usage: python -m src.main
"""
from src.infer import run_inference
from src.evaluate import evaluate_model
from src.train import train_model
from src.features import get_feature_preprocessor
from src.validate import validate_dataframe
from src.clean_data import clean_dataframe
from src.load_data import load_raw_data
from src.utils import save_csv, save_model
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
if __name__ == "__main__":
    print("Pipeline not implemented yet.")


"""
Educational Goal:
- Why this module exists in an MLOps system: Serves as the orchestration layer binding all modular steps into a single executable pipeline.
- Responsibility (separation of concerns): High-level flow control, configuration mapping, and data routing.
- Pipeline contract (inputs and outputs): Script entry point. No explicit returns, but generates all physical artifacts.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""


# ==========================================
# CONFIGURATION BLOCK
# !!! STUDENT REMINDER: Map this SETTINGS block to your real dataset !!!
# ==========================================
SETTINGS = {
    "is_example_config": True,
    "target_column": "target",
    "problem_type": "classification",  # Change to "regression" if needed
    "features": {
        "quantile_bin": ["num_feature"],
        "categorical_onehot": ["cat_feature"],
        "numeric_passthrough": [],
        "n_bins": 3
    }
}


def main():
    # TODO: replace with logging later
    print("=== STARTING MLOPS PIPELINE ===")

    # 1. Setup paths and directories
    raw_path = Path("data/raw/data.csv")
    clean_path = Path("data/processed/clean.csv")
    model_path = Path("models/model.joblib")
    preds_path = Path("reports/predictions.csv")

    for p in [raw_path, clean_path, model_path, preds_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    if SETTINGS.get("is_example_config"):
        # TODO: replace with logging later
        print("Note: Running with dummy example configuration.")

    # 2. Load
    print("\n--- STEP 1: LOAD ---")  # TODO: replace with logging later
    df_raw = load_raw_data(raw_path)

    # 3. Clean
    print("\n--- STEP 2: CLEAN ---")  # TODO: replace with logging later
    df_clean = clean_dataframe(df_raw, target_column=SETTINGS["target_column"])
    save_csv(df_clean, clean_path)

    # 4. Validate
    print("\n--- STEP 3: VALIDATE ---")  # TODO: replace with logging later
    # Base required columns = features + target
    req_cols = SETTINGS["features"]["quantile_bin"] + \
        SETTINGS["features"]["categorical_onehot"] + \
        SETTINGS["features"]["numeric_passthrough"] + \
        [SETTINGS["target_column"]]
    validate_dataframe(df_clean, required_columns=req_cols)

    # 5. Train/Test Split (Done BEFORE features to prevent leakage)
    print("\n--- STEP 4: SPLIT ---")  # TODO: replace with logging later
    X = df_clean.drop(columns=[SETTINGS["target_column"]])
    y = df_clean[SETTINGS["target_column"]]

    if SETTINGS["problem_type"] == "classification":
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=42)
            # TODO: replace with logging later
            print("Stratified train/test split successful.")
        except ValueError:
            # TODO: replace with logging later
            print(
                "Stratify failed (likely too few samples in dummy data). Falling back to random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

    # Fail-fast feature checks
    for col in SETTINGS["features"]["quantile_bin"]:
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            raise ValueError(
                f"Feature Check Failed: '{col}' intended for quantile_bin is not numeric!")

    # 6. Features & Train
    # TODO: replace with logging later
    print("\n--- STEP 5: BUILD & TRAIN ---")
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=SETTINGS["features"]["quantile_bin"],
        categorical_onehot_cols=SETTINGS["features"]["categorical_onehot"],
        numeric_passthrough_cols=SETTINGS["features"]["numeric_passthrough"],
        n_bins=SETTINGS["features"]["n_bins"]
    )

    pipeline = train_model(X_train, y_train, preprocessor,
                           problem_type=SETTINGS["problem_type"])
    save_model(pipeline, model_path)

    # 7. Evaluate
    print("\n--- STEP 6: EVALUATE ---")  # TODO: replace with logging later
    metric_value = evaluate_model(
        pipeline, X_test, y_test, problem_type=SETTINGS["problem_type"])

    # 8. Inference
    print("\n--- STEP 7: INFERENCE ---")  # TODO: replace with logging later
    # Generating dummy inference using test set for demonstration
    predictions_df = run_inference(pipeline, X_test)
    save_csv(predictions_df, preds_path)

    # TODO: replace with logging later
    print("\n=== MLOPS PIPELINE COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
