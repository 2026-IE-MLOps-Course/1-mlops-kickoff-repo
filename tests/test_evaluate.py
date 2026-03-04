import pandas as pd
import pytest
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.train import train_model


# ---------------------------------------------------------
# Regression Test
# ---------------------------------------------------------

def test_evaluate_regression_returns_float():
    # Create tiny regression dataset
    X_array, y_array = make_regression(
        n_samples=50,
        n_features=3,
        noise=0.1,
        random_state=42,
    )

    X = pd.DataFrame(X_array, columns=["f1", "f2", "f3"])
    y = pd.Series(y_array)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42
    )

    preprocessor = get_feature_preprocessor(
        numeric_passthrough_cols=["f1", "f2", "f3"]
    )

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        problem_type="regression",
    )

    score = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        problem_type="regression",
    )

    assert isinstance(score, float)
    assert score >= 0


# ---------------------------------------------------------
# Classification Test
# ---------------------------------------------------------

def test_evaluate_classification_returns_auc():
    # Create tiny binary classification dataset
    X_array, y_array = make_classification(
        n_samples=50,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=42,
    )

    X = pd.DataFrame(X_array, columns=["f1", "f2", "f3", "f4"])
    y = pd.Series(y_array)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42
    )

    preprocessor = get_feature_preprocessor(
        numeric_passthrough_cols=["f1", "f2", "f3", "f4"]
    )

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        problem_type="classification",
    )

    score = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        problem_type="classification",
    )

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
