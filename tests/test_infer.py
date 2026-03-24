import pandas as pd
import numpy as np
from src.infer import run_inference


class MockModelWithProba:
    def predict(self, X):
        return np.array([0, 1])

    def predict_proba(self, X):
        # Mock probabilities for a binary classifier (class 0, class 1)
        return np.array([[0.8, 0.2], [0.3, 0.7]])


class MockModelNoProba:
    def predict(self, X):
        return np.array([1, 1])


def test_run_inference_with_proba():
    model = MockModelWithProba()
    X = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]}, index=["A", "B"])

    out = run_inference(model, X, include_proba=True)

    # Check predictions
    assert "prediction" in out.columns
    assert list(out["prediction"]) == [0, 1]

    # Check probabilities (should take class 1 proba)
    assert "proba" in out.columns
    assert list(out["proba"]) == [0.2, 0.7]

    # Check index retention
    assert list(out.index) == ["A", "B"]


def test_run_inference_without_include_proba():
    model = MockModelWithProba()
    X = pd.DataFrame({"feat1": [1, 2]}, index=["A", "B"])

    out = run_inference(model, X, include_proba=False)

    assert "prediction" in out.columns
    assert "proba" not in out.columns


def test_run_inference_model_without_predict_proba():
    model = MockModelNoProba()
    X = pd.DataFrame({"feat1": [1, 2]}, index=["A", "B"])

    # Even if include_proba=True, it shouldn't fail if model lacks the method
    out = run_inference(model, X, include_proba=True)

    assert "prediction" in out.columns
    assert list(out["prediction"]) == [1, 1]
    assert "proba" not in out.columns
