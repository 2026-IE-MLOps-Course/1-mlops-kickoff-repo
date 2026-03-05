import pytest
import pandas as pd
from src.infer import run_inference
from src.features import get_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def trained_pipeline_and_data():
    data = pd.DataFrame({
        "tenure": [10, 20, 30],
        "MonthlyCharges": [10.0, 20.0, 30.0],
        "TotalCharges": [100.0, 200.0, 300.0],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "Churn": [0, 1, 0]
    })
    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    preprocessor = get_preprocessor(X)
    # Tu train.py nombra los pasos como 'preprocess' y 'model'
    pipeline = Pipeline([("preprocess", preprocessor),
                        ("model", LogisticRegression())])
    pipeline.fit(X, y)
    return pipeline, X


def test_run_inference(trained_pipeline_and_data):
    """Verifica que la inferencia devuelva el formato correcto."""
    model, X_infer = trained_pipeline_and_data
    preds = run_inference(model, X_infer)

    assert "prediction" in preds.columns
    assert len(preds) == len(X_infer)
