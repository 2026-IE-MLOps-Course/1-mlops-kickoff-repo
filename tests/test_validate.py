"""
test_validate.py
----------------
Unit tests for the data validation module.
"""

import pandas as pd
import pytest

from src.validate import validate_dataframe


class TestValidateDataframe:
    """Tests for the validate_dataframe function."""

    def test_passes_on_valid_data(self, sample_clean_df):
        """No exception on well-formed data."""
        required = ["duration_days", "traveler_age", "total_cost"]
        result = validate_dataframe(sample_clean_df, required)
        assert result is True

    def test_returns_bool(self, sample_clean_df):
        """Return type is bool."""
        result = validate_dataframe(sample_clean_df, ["total_cost"])
        assert isinstance(result, bool)

    def test_empty_df_raises(self):
        """ValueError when the DataFrame has zero rows."""
        df_empty = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            validate_dataframe(df_empty, ["col_a"])

    def test_missing_required_column_raises(self, sample_clean_df):
        """ValueError when a required column is missing."""
        with pytest.raises(ValueError):
            validate_dataframe(sample_clean_df, ["nonexistent_column"])

    def test_negative_age_caught(self, sample_clean_df):
        """Negative traveler_age violates range and triggers error."""
        sample_clean_df.loc[0, "traveler_age"] = -5
        required = ["duration_days", "traveler_age", "total_cost"]
        with pytest.raises(ValueError):
            validate_dataframe(sample_clean_df, required)

    def test_over_max_duration_caught(self, sample_clean_df):
        """Duration above configured max triggers error."""
        sample_clean_df.loc[0, "duration_days"] = 999
        required = ["duration_days", "traveler_age", "total_cost"]
        with pytest.raises(ValueError):
            validate_dataframe(sample_clean_df, required)

    def test_null_target_caught(self, sample_clean_df):
        """Null values in target column trigger validation failure."""
        sample_clean_df.loc[0, "total_cost"] = None
        required = ["duration_days", "traveler_age", "total_cost"]
        with pytest.raises(ValueError):
            validate_dataframe(sample_clean_df, required)

    # ------------------------------------------------------------------ #
    # NEW: Cover line 79 — continue when range-check column is absent     #
    # ------------------------------------------------------------------ #

    def test_range_check_skipped_for_missing_columns(self):
        """When range-check columns (duration_days, traveler_age, etc.) are
        not present in the DataFrame, the validator skips them gracefully
        (covers line 79)."""
        # DataFrame without any of the range-check columns
        df = pd.DataFrame({
            "some_feature": [1, 2, 3],
            "total_cost": [100.0, 200.0, 300.0],
        })
        # Only require columns that actually exist
        required = ["some_feature", "total_cost"]
        result = validate_dataframe(df, required)
        assert result is True
