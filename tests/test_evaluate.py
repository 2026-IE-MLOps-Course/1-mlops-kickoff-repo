import pandas as pd
from src.evaluate import evaluate_model


class MockModel:
    def predict(self, X):
        return [0] * len(X)


def test_evaluate_classification_binary():
    model = MockModel()
    X_test = pd.DataFrame({"feat": [1, 2]})
    y_test = pd.Series([0, 1])
    # with 2 unique classes, returns f1 score. mock predicts all 0.
    # f1 for class 1 will be 0.
    score = evaluate_model(
        model, X_test, y_test, problem_type="classification"
    )
    assert isinstance(score, float)


def test_evaluate_regression():
    model = MockModel()
    X_test = pd.DataFrame({"feat": [1, 2]})
    y_test = pd.Series([0.5, 1.5])
    score = evaluate_model(model, X_test, y_test, problem_type="regression")
    assert isinstance(score, float)
    assert score > 0
