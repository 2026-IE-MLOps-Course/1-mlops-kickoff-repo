"""
test_evaluate.py
----------------
Unit tests for the model evaluation module.
"""

import pandas as pd
import pytest

from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.train import train_model


@pytest.fixture()
def fitted_regression_pipeline(sample_feature_df, sample_target):
    """Train a minimal regression pipeline and return (pipeline, X, y)."""
    preprocessor = get_feature_preprocessor(
        numeric_passthrough_cols=["duration_days", "traveler_age",
                                  "travel_month", "day_of_week"],
        categorical_onehot_cols=["destination_country", "traveler_gender",
                                 "traveler_nationality",
                                 "accommodation_type",
                                 "transportation_type"],
    )
    pipeline = train_model(
        sample_feature_df, sample_target, preprocessor, "regression"
    )
    return pipeline, sample_feature_df, sample_target


@pytest.fixture()
def fitted_classification_pipeline(sample_feature_df):
    """Train a minimal classification pipeline and return (pipeline, X, y)."""
    y_class = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    preprocessor = get_feature_preprocessor(
        numeric_passthrough_cols=["duration_days", "traveler_age",
                                  "travel_month", "day_of_week"],
        categorical_onehot_cols=["destination_country", "traveler_gender",
                                 "traveler_nationality",
                                 "accommodation_type",
                                 "transportation_type"],
    )
    pipeline = train_model(
        sample_feature_df, y_class, preprocessor, "classification"
    )
    return pipeline, sample_feature_df, y_class


class TestEvaluateModel:
    """Tests for the evaluate_model function."""

    def test_returns_float_regression(self, fitted_regression_pipeline):
        """Regression: returns a single float (RMSE)."""
        pipeline, X, y = fitted_regression_pipeline
        metric = evaluate_model(pipeline, X, y, "regression")
        assert isinstance(metric, float)

    def test_returns_float_classification(self, fitted_classification_pipeline):
        """Classification: returns a single float (F1)."""
        pipeline, X, y = fitted_classification_pipeline
        metric = evaluate_model(pipeline, X, y, "classification")
        assert isinstance(metric, float)

    def test_rmse_is_non_negative(self, fitted_regression_pipeline):
        """RMSE is always >= 0."""
        pipeline, X, y = fitted_regression_pipeline
        metric = evaluate_model(pipeline, X, y, "regression")
        assert metric >= 0.0

    def test_f1_in_valid_range(self, fitted_classification_pipeline):
        """F1 score is between 0 and 1."""
        pipeline, X, y = fitted_classification_pipeline
        metric = evaluate_model(pipeline, X, y, "classification")
        assert 0.0 <= metric <= 1.0

    def test_no_predict_raises(self):
        """TypeError if the object has no .predict() method."""
        X = pd.DataFrame({"duration_days": [7]})
        y = pd.Series([1800])
        with pytest.raises(TypeError, match="predict"):
            evaluate_model("not_a_model", X, y, "regression")

    def test_shape_mismatch_raises(self, fitted_regression_pipeline):
        """ValueError when X and y have different lengths."""
        pipeline, X, y = fitted_regression_pipeline
        with pytest.raises(ValueError, match="mismatch"):
            evaluate_model(pipeline, X, y.iloc[:5], "regression")

    def test_unknown_problem_type_raises(self, fitted_regression_pipeline):
        """ValueError for unrecognised problem_type."""
        pipeline, X, y = fitted_regression_pipeline
        with pytest.raises(ValueError, match="Unknown"):
            evaluate_model(pipeline, X, y, "unknown")
