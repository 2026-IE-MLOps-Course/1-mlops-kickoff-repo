import pytest
import pandas as pd
from src.evaluate import evaluate_model
from src.features import get_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def trained_pipeline_and_data():
    data = {
        "tenure": [10, 20, 30, 40],
        "MonthlyCharges": [10.0, 20.0, 30.0, 40.0],
        "TotalCharges": [100.0, 200.0, 300.0, 400.0],
        "InternetService": ["DSL", "Fiber optic", "No", "DSL"],
        "Churn": [0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    preprocessor = get_preprocessor(X)
    pipeline = Pipeline(
        [("pre", preprocessor), ("model", LogisticRegression())])
    pipeline.fit(X, y)
    return pipeline, X, y


def test_evaluate_model(trained_pipeline_and_data):
    """Verifica que el modelo devuelva la métrica AUC correctamente."""
    model, X_test, y_test = trained_pipeline_and_data
    score = evaluate_model(model, X_test, y_test, problem_type="classification")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
