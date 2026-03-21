"""
test_features.py
----------------
Unit tests for the feature engineering module.
"""

from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from src.features import get_feature_preprocessor


class TestGetFeaturePreprocessor:
    """Tests for the preprocessor recipe builder."""

    def test_returns_column_transformer(self):
        """The recipe should be an unfitted ColumnTransformer."""
        preprocessor = get_feature_preprocessor(
            numeric_passthrough_cols=["duration_days", "traveler_age"],
            categorical_onehot_cols=["traveler_gender"],
        )
        assert isinstance(preprocessor, ColumnTransformer)

    def test_empty_feature_lists_raises(self):
        """ValueError when all feature lists are empty."""
        with pytest.raises(ValueError, match="empty"):
            get_feature_preprocessor()

    def test_preprocessor_not_fitted(self):
        """The returned object should NOT already be fitted."""
        preprocessor = get_feature_preprocessor(
            numeric_passthrough_cols=["duration_days"],
        )
        with pytest.raises(Exception):
            preprocessor.transform(
                pd.DataFrame({"duration_days": [5]})
            )

    def test_remainder_is_drop(self):
        """Unauthorised columns are dropped (remainder='drop')."""
        preprocessor = get_feature_preprocessor(
            numeric_passthrough_cols=["duration_days"],
        )
        assert preprocessor.remainder == "drop"

    def test_with_quantile_bin_cols(self):
        """KBinsDiscretizer is used for quantile_bin columns."""
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=["traveler_age"],
            n_bins=4,
        )
        assert isinstance(preprocessor, ColumnTransformer)
        # Verify the transformer name exists
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert "quantile_bin" in transformer_names

    def test_with_all_column_types(self):
        """Preprocessor handles quantile_bin, passthrough, and onehot together."""
        preprocessor = get_feature_preprocessor(
            quantile_bin_cols=["traveler_age"],
            categorical_onehot_cols=["traveler_gender"],
            numeric_passthrough_cols=["duration_days"],
            n_bins=3,
        )
        transformer_names = [name for name, _, _ in preprocessor.transformers]
        assert "quantile_bin" in transformer_names
        assert "num_passthrough" in transformer_names
        assert "cat_onehot" in transformer_names

    def test_fits_on_real_data(self, sample_feature_df):
        """Preprocessor can be fitted on the sample feature DataFrame."""
        preprocessor = get_feature_preprocessor(
            numeric_passthrough_cols=["duration_days", "traveler_age",
                                      "travel_month", "day_of_week"],
            categorical_onehot_cols=["destination_country", "traveler_gender",
                                     "traveler_nationality",
                                     "accommodation_type",
                                     "transportation_type"],
        )
        # Should not raise
        preprocessor.fit(sample_feature_df)
        transformed = preprocessor.transform(sample_feature_df)
        assert transformed.shape[0] == len(sample_feature_df)

    # ------------------------------------------------------------------ #
    # NEW: Cover lines 106-107 — TypeError fallback for older sklearn     #
    # ------------------------------------------------------------------ #

    def test_onehot_fallback_for_older_sklearn(self):
        """When OneHotEncoder raises TypeError on sparse_output, the code
        falls back to the sparse parameter (covers lines 106-107).

        Since modern sklearn doesn't accept 'sparse' either, we patch
        OneHotEncoder so the first call (with sparse_output) raises TypeError
        and the second call (with sparse) succeeds."""
        real_ohe_class = OneHotEncoder

        call_count = [0]

        def mock_ohe_constructor(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1 and "sparse_output" in kwargs:
                # First call: simulate old sklearn rejecting sparse_output
                raise TypeError("unexpected keyword argument 'sparse_output'")
            # Second call (or any without sparse_output): create a real
            # OneHotEncoder but only with the params it actually accepts.
            # Strip 'sparse' if present — modern sklearn uses 'sparse_output'.
            clean_kwargs = {}
            if "handle_unknown" in kwargs:
                clean_kwargs["handle_unknown"] = kwargs["handle_unknown"]
            if "sparse_output" in kwargs:
                clean_kwargs["sparse_output"] = kwargs["sparse_output"]
            elif "sparse" in kwargs:
                clean_kwargs["sparse_output"] = kwargs["sparse"]
            return real_ohe_class(**clean_kwargs)

        with patch("src.features.OneHotEncoder", side_effect=mock_ohe_constructor):
            preprocessor = get_feature_preprocessor(
                categorical_onehot_cols=["traveler_gender"],
            )
            assert isinstance(preprocessor, ColumnTransformer)
            transformer_names = [name for name, _, _ in preprocessor.transformers]
            assert "cat_onehot" in transformer_names
