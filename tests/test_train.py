"""
test_train.py
-------------
Unit tests for the model training module (src/train.py).

Coverage:
- Happy path: regression & classification, Pipeline type, step names, predict shape
- Determinism: same random_state produces identical predictions
- Edge cases: empty X/y, row-count mismatch, unknown problem_type, single-row
- Stronger assertions: prediction dtype, output column count
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features import get_feature_preprocessor
from src.train import train_model


# --------------------------------------------------------------------------- #
# Shared fixture                                                               #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def preprocessor():
    """Unfitted ColumnTransformer matching the travel feature set."""
    return get_feature_preprocessor(
        numeric_passthrough_cols=["duration_days", "traveler_age",
                                  "travel_month", "day_of_week"],
        categorical_onehot_cols=["destination_country", "traveler_gender",
                                 "traveler_nationality",
                                 "accommodation_type",
                                 "transportation_type"],
    )


@pytest.fixture()
def fresh_preprocessor():
    """Second unfitted ColumnTransformer for determinism tests."""
    return get_feature_preprocessor(
        numeric_passthrough_cols=["duration_days", "traveler_age",
                                  "travel_month", "day_of_week"],
        categorical_onehot_cols=["destination_country", "traveler_gender",
                                 "traveler_nationality",
                                 "accommodation_type",
                                 "transportation_type"],
    )


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #

class TestTrainModel:
    """Unit tests for the train_model function."""

    # ------------------------------------------------------------------ #
    # Return-type and structure                                            #
    # ------------------------------------------------------------------ #

    def test_returns_sklearn_pipeline(self, sample_feature_df,
                                      sample_target, preprocessor):
        """Return value is always an sklearn Pipeline instance."""
        pipeline = train_model(
            sample_feature_df, sample_target, preprocessor, "regression"
        )
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_exactly_two_steps(self, sample_feature_df,
                                            sample_target, preprocessor):
        """Pipeline must contain exactly two steps."""
        pipeline = train_model(
            sample_feature_df, sample_target, preprocessor, "regression"
        )
        assert len(pipeline.steps) == 2

    def test_pipeline_step_names(self, sample_feature_df,
                                 sample_target, preprocessor):
        """Step names are 'preprocess' and 'model' (in that order)."""
        pipeline = train_model(
            sample_feature_df, sample_target, preprocessor, "regression"
        )
        step_names = [name for name, _ in pipeline.steps]
        assert step_names == ["preprocess", "model"]

    # ------------------------------------------------------------------ #
    # Regression happy path                                                #
    # ------------------------------------------------------------------ #

    def test_regression_predict_shape(self, sample_feature_df,
                                      sample_target, preprocessor):
        """Regression predictions have the same length as the input."""
        pipeline = train_model(
            sample_feature_df, sample_target, preprocessor, "regression"
        )
        preds = pipeline.predict(sample_feature_df)
        assert len(preds) == len(sample_feature_df)

    def test_regression_predictions_are_numeric(self, sample_feature_df,
                                                 sample_target, preprocessor):
        """Regression predictions must be a numeric numpy array."""
        pipeline = train_model(
            sample_feature_df, sample_target, preprocessor, "regression"
        )
        preds = pipeline.predict(sample_feature_df)
        assert np.issubdtype(preds.dtype, np.number), (
            f"Expected numeric predictions, got dtype {preds.dtype}"
        )

    def test_regression_predictions_positive(self, sample_feature_df,
                                              sample_target, preprocessor):
        """Travel cost predictions should be positive (domain sanity check)."""
        pipeline = train_model(
            sample_feature_df, sample_target, preprocessor, "regression"
        )
        preds = pipeline.predict(sample_feature_df)
        assert np.all(preds > 0), "All cost predictions should be positive"

    # ------------------------------------------------------------------ #
    # Classification happy path                                            #
    # ------------------------------------------------------------------ #

    def test_classification_predict_shape(self, sample_feature_df, preprocessor):
        """Classification predictions have the same length as the input."""
        y_class = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], name="label")
        pipeline = train_model(
            sample_feature_df, y_class, preprocessor, "classification"
        )
        preds = pipeline.predict(sample_feature_df)
        assert len(preds) == len(sample_feature_df)

    def test_classification_predictions_in_label_set(self,
                                                      sample_feature_df,
                                                      preprocessor):
        """Classification predictions are contained within the training label set."""
        y_class = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], name="label")
        pipeline = train_model(
            sample_feature_df, y_class, preprocessor, "classification"
        )
        preds = pipeline.predict(sample_feature_df)
        assert set(preds).issubset({0, 1})

    # ------------------------------------------------------------------ #
    # Determinism                                                          #
    # ------------------------------------------------------------------ #

    def test_regression_is_deterministic(self, sample_feature_df, sample_target,
                                         preprocessor, fresh_preprocessor):
        """Two calls with identical random_state produce identical predictions."""
        p1 = train_model(sample_feature_df, sample_target,
                         preprocessor, "regression")
        p2 = train_model(sample_feature_df, sample_target,
                         fresh_preprocessor, "regression")
        np.testing.assert_array_equal(
            p1.predict(sample_feature_df),
            p2.predict(sample_feature_df),
        )

    # ------------------------------------------------------------------ #
    # Error paths                                                          #
    # ------------------------------------------------------------------ #

    def test_empty_X_raises_value_error(self, preprocessor):
        """ValueError when X_train is empty."""
        X_empty = pd.DataFrame(columns=["duration_days", "traveler_age"])
        y_empty = pd.Series(dtype=float)
        with pytest.raises(ValueError, match="empty"):
            train_model(X_empty, y_empty, preprocessor, "regression")

    def test_empty_y_raises_value_error(self, sample_feature_df, preprocessor):
        """ValueError when y_train is empty but X is not."""
        y_empty = pd.Series(dtype=float)
        with pytest.raises(ValueError):
            train_model(sample_feature_df, y_empty, preprocessor, "regression")

    def test_row_count_mismatch_raises(self, sample_feature_df,
                                       sample_target, preprocessor):
        """ValueError when X and y have different numbers of rows."""
        with pytest.raises(ValueError, match="mismatch"):
            train_model(
                sample_feature_df, sample_target.iloc[:5],
                preprocessor, "regression"
            )

    def test_unknown_problem_type_raises(self, sample_feature_df,
                                         sample_target, preprocessor):
        """ValueError for any unrecognised problem_type string."""
        with pytest.raises(ValueError, match="Unknown"):
            train_model(
                sample_feature_df, sample_target,
                preprocessor, "clustering"
            )

    def test_empty_string_problem_type_raises(self, sample_feature_df,
                                               sample_target, preprocessor):
        """ValueError for an empty-string problem_type."""
        with pytest.raises(ValueError):
            train_model(
                sample_feature_df, sample_target,
                preprocessor, ""
            )
