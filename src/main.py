"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow (Load -> Clean -> Validate -> Train -> Evaluate).
Usage: python -m src.main
"""
import os
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Importaciones correctas desde tus módulos
from src.load_data import load_raw_data
from src.clean_data import clean_data
from src.validate import validate_dataframe
from src.features import engineer_features, get_preprocessor
from src.train import train_model
from src.evaluate import evaluate_model
from src.infer import run_inference

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    logger.info("=== STARTING MLOPS PIPELINE ===")

    # 1. Configuración de rutas
    raw_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    clean_path = "data/processed/clean.csv"
    model_path = "models/model.joblib"
    preds_path = "reports/predictions.csv"

    for p in [clean_path, model_path, preds_path]:
        os.makedirs(os.path.dirname(p), exist_ok=True)

    # 2. Carga
    logger.info("--- STEP 1: LOAD ---")
    df_raw = load_raw_data(raw_path)

    # 3. Limpieza
    logger.info("--- STEP 2: CLEAN ---")
    df_clean = clean_data(df_raw, drop_na=True)
    df_clean.to_csv(clean_path, index=False)

    # 4. Validación
    logger.info("--- STEP 3: VALIDATE ---")
    req_cols = ['customerID', 'tenure',
                'MonthlyCharges', 'TotalCharges', 'Churn']
    validate_dataframe(df_clean, required_columns=req_cols)

    # 5. Feature Engineering & Split
    logger.info("--- STEP 4: FEATURE ENGINEERING & SPLIT ---")
    df_eng = engineer_features(df_clean)

    # Mapeo del Target
    X = df_eng.drop(columns=["Churn"])
    y = df_eng["Churn"].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # 6. Preprocesamiento y Entrenamiento
    logger.info("--- STEP 5: BUILD & TRAIN ---")
    preprocessor = get_preprocessor(X_train)
    pipeline = train_model(
        X_train, y_train, preprocessor, model_path=model_path)

    # 7. Evaluación
    logger.info("--- STEP 6: EVALUATE ---")
    metric_value = evaluate_model(
        pipeline, X_test, y_test, problem_type="classification")

    # 8. Inferencia
    logger.info("--- STEP 7: INFERENCE ---")
    predictions_df = run_inference(pipeline, X_test)
    predictions_df.to_csv(preds_path)

    logger.info("=== MLOPS PIPELINE COMPLETED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
