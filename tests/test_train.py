import pytest
import pandas as pd
import os
from src.train import train_model
from src.features import get_preprocessor
from sklearn.pipeline import Pipeline


def test_train_model(tmp_path):
    """Verifica la creación del pipeline de entrenamiento y guardado del artefacto."""
    df = pd.DataFrame({
        "tenure": [10, 20, 30],
        "MonthlyCharges": [10.0, 20.0, 30.0],
        "TotalCharges": [100.0, 200.0, 300.0],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "Churn": [0, 1, 0]
    })
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    preprocessor = get_preprocessor(X)

    # tmp_path es una carpeta temporal que crea pytest para no ensuciar tu repo real
    model_path = tmp_path / "test_model.joblib"

    # Corregido: Eliminamos 'problem_type' para respetar el contrato de tu train.py
    pipeline = train_model(X, y, preprocessor, model_path=str(model_path))

    assert isinstance(pipeline, Pipeline)
    assert os.path.exists(str(model_path))
