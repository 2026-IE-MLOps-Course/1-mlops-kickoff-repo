"""
test_infer.py
-------------
Unit tests for the inference module (src/infer.py).

Coverage:
- Happy path: output shape, column name, index preservation, dtype
- Edge cases: custom index, single row, empty DataFrame, non-contiguous index
- Error paths: non-DataFrame input (ndarray, list, dict), artifact without .predict()
"""

import numpy as np
import pandas as pd
import pytest

from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.train import train_model


# --------------------------------------------------------------------------- #
# Shared fixture: a pre-trained regression pipeline                            #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def fitted_pipeline(sample_feature_df, sample_target):
    """Return a fitted regression Pipeline trained on the 10-row sample."""
    preprocessor = get_feature_preprocessor(
        numeric_passthrough_cols=["duration_days", "traveler_age",
                                  "travel_month", "day_of_week"],
        categorical_onehot_cols=["destination_country", "traveler_gender",
                                 "traveler_nationality",
                                 "accommodation_type",
                                 "transportation_type"],
    )
    return train_model(
        sample_feature_df, sample_target, preprocessor, "regression"
    )


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #

class TestRunInference:
    """Unit tests for the run_inference function."""

    # ------------------------------------------------------------------ #
    # Output shape and schema                                              #
    # ------------------------------------------------------------------ #

    def test_returns_dataframe(self, fitted_pipeline, sample_feature_df):
        """Output is a pandas DataFrame."""
        result = run_inference(fitted_pipeline, sample_feature_df)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_exactly_one_column(self, fitted_pipeline,
                                           sample_feature_df):
        """Output DataFrame has exactly one column."""
        result = run_inference(fitted_pipeline, sample_feature_df)
        assert result.shape[1] == 1, (
            f"Expected 1 column, got {result.shape[1]}: {list(result.columns)}"
        )

    def test_output_column_named_prediction(self, fitted_pipeline,
                                             sample_feature_df):
        """The single output column is named 'prediction'."""
        result = run_inference(fitted_pipeline, sample_feature_df)
        assert list(result.columns) == ["prediction"]

    def test_output_row_count_matches_input(self, fitted_pipeline,
                                            sample_feature_df):
        """Number of rows in output equals number of rows in input."""
        result = run_inference(fitted_pipeline, sample_feature_df)
        assert len(result) == len(sample_feature_df)

    def test_prediction_values_are_numeric(self, fitted_pipeline,
                                            sample_feature_df):
        """'prediction' column contains numeric (float/int) values."""
        result = run_inference(fitted_pipeline, sample_feature_df)
        assert pd.api.types.is_numeric_dtype(result["prediction"]), (
            f"Prediction column dtype is {result['prediction'].dtype}, "
            "expected numeric"
        )

    def test_no_nan_in_predictions(self, fitted_pipeline, sample_feature_df):
        """There must be no NaN values in the prediction column."""
        result = run_inference(fitted_pipeline, sample_feature_df)
        assert not result["prediction"].isna().any(), (
            "Predictions contain NaN values"
        )

    # ------------------------------------------------------------------ #
    # Index preservation                                                   #
    # ------------------------------------------------------------------ #

    def test_preserves_default_index(self, fitted_pipeline, sample_feature_df):
        """Output index matches the default RangeIndex of the input."""
        result = run_inference(fitted_pipeline, sample_feature_df)
        pd.testing.assert_index_equal(result.index, sample_feature_df.index)

    def test_preserves_custom_integer_index(self, fitted_pipeline,
                                             sample_feature_df):
        """Output index matches a custom integer index on the input."""
        X = sample_feature_df.copy()
        X.index = range(100, 110)
        result = run_inference(fitted_pipeline, X)
        assert list(result.index) == list(range(100, 110))

    def test_preserves_non_contiguous_index(self, fitted_pipeline,
                                             sample_feature_df):
        """Output index matches a non-contiguous (e.g., [0,2,4,…]) index."""
        X = sample_feature_df.copy()
        X.index = [i * 2 for i in range(len(X))]
        result = run_inference(fitted_pipeline, X)
        assert list(result.index) == list(X.index)

    # ------------------------------------------------------------------ #
    # Edge cases                                                           #
    # ------------------------------------------------------------------ #

    def test_single_row_inference(self, fitted_pipeline, sample_feature_df):
        """Inference on a single-row DataFrame returns exactly one prediction."""
        X_one = sample_feature_df.iloc[[0]]
        result = run_inference(fitted_pipeline, X_one)
        assert len(result) == 1
        assert list(result.columns) == ["prediction"]

    # ------------------------------------------------------------------ #
    # Error paths                                                          #
    # ------------------------------------------------------------------ #

    def test_numpy_array_raises_type_error(self, fitted_pipeline,
                                            sample_feature_df):
        """TypeError when a numpy ndarray is passed instead of a DataFrame."""
        with pytest.raises(TypeError, match="DataFrame"):
            run_inference(fitted_pipeline, sample_feature_df.values)

    def test_list_raises_type_error(self, fitted_pipeline):
        """TypeError when a plain Python list is passed."""
        with pytest.raises(TypeError, match="DataFrame"):
            run_inference(fitted_pipeline, [[7, 35, 5, 0]])

    def test_dict_raises_type_error(self, fitted_pipeline):
        """TypeError when a dict is passed instead of a DataFrame."""
        with pytest.raises(TypeError, match="DataFrame"):
            run_inference(fitted_pipeline, {"duration_days": [7]})

    def test_artifact_without_predict_raises(self):
        """TypeError when the model artifact lacks a .predict() method."""
        X = pd.DataFrame({"duration_days": [7]})
        with pytest.raises(TypeError, match="predict"):
            run_inference("not_a_model", X)

    def test_none_model_raises(self):
        """TypeError when None is passed as the model."""
        X = pd.DataFrame({"duration_days": [7]})
        with pytest.raises(TypeError):
            run_inference(None, X)
