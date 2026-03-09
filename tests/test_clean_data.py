"""
test_clean_data.py
------------------
Unit tests for the data cleaning module.
"""

import pandas as pd
import pytest

from src.clean_data import clean_data, _standardise_columns


class TestStandardiseColumns:
    """Tests for column name normalisation."""

    def test_lowercases_and_underscores(self):
        """Spaces and special chars are replaced with underscores."""
        df = pd.DataFrame({"Trip ID": [1], "Duration (days)": [5]})
        result = _standardise_columns(df)
        assert "trip_id" in result.columns
        assert "duration_days" in result.columns

    def test_strips_whitespace(self):
        """Leading/trailing whitespace in column names is removed."""
        df = pd.DataFrame({" Name ": [1]})
        result = _standardise_columns(df)
        assert "name" in result.columns


class TestCleanData:
    """Tests for the main clean_data function."""

    def test_drops_configured_columns(self, sample_raw_df, sample_config):
        """Columns listed in config.cleaning.drop_columns are removed."""
        result = clean_data(sample_raw_df, sample_config)
        assert "trip_id" not in result.columns
        assert "traveler_name" not in result.columns

    def test_creates_target_column(self, sample_raw_df, sample_config):
        """The total_cost target column is created."""
        result = clean_data(sample_raw_df, sample_config)
        assert "total_cost" in result.columns
        # First row: 1200 + 600 = 1800
        assert result["total_cost"].iloc[0] == 1800.0

    def test_removes_duplicates(self, sample_raw_df, sample_config):
        """Duplicate rows are dropped."""
        df_dup = pd.concat([sample_raw_df, sample_raw_df.iloc[[0]]])
        result = clean_data(df_dup, sample_config)
        assert len(result) == 10

    def test_drops_na_rows(self, sample_raw_df, sample_config):
        """Rows with missing values are removed."""
        sample_raw_df.loc[0, "Accommodation cost"] = None
        result = clean_data(sample_raw_df, sample_config)
        assert len(result) == 9

    def test_index_is_reset(self, sample_raw_df, sample_config):
        """Index is sequential after cleaning."""
        sample_raw_df.loc[0, "Accommodation cost"] = None
        result = clean_data(sample_raw_df, sample_config)
        assert list(result.index) == list(range(len(result)))

    def test_idempotent(self, sample_raw_df, sample_config):
        """Running clean_data twice produces the same result."""
        first = clean_data(sample_raw_df, sample_config)
        second = clean_data(sample_raw_df, sample_config)
        pd.testing.assert_frame_equal(first, second)

    def test_missing_cost_columns_raises(self, sample_config):
        """KeyError when accommodation/transportation cost missing."""
        df_bad = pd.DataFrame({"Col A": [1], "Col B": [2]})
        with pytest.raises(KeyError):
            clean_data(df_bad, sample_config)
