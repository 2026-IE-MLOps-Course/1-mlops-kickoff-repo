"""
test_clean_data.py
------------------
Unit tests for the data cleaning module.
"""

import pandas as pd
import pytest

from src.clean_data import clean_dataframe


class TestCleanDataframe:
    """Tests for the clean_dataframe function."""

    def test_drops_configured_columns(self, sample_raw_df):
        """Columns like trip_id and traveler_name are removed."""
        result = clean_dataframe(sample_raw_df, "total_cost")
        assert "trip_id" not in result.columns
        assert "traveler_name" not in result.columns

    def test_creates_target_column(self, sample_raw_df):
        """The total_cost target column is created."""
        result = clean_dataframe(sample_raw_df, "total_cost")
        assert "total_cost" in result.columns
        # First row: 1200 + 600 = 1800
        assert result["total_cost"].iloc[0] == 1800.0

    def test_standardises_column_names(self, sample_raw_df):
        """Column names are lowercased and spaces replaced."""
        result = clean_dataframe(sample_raw_df, "total_cost")
        for col in result.columns:
            assert col == col.lower()
            assert " " not in col

    def test_removes_duplicates(self, sample_raw_df):
        """Duplicate rows are dropped."""
        df_dup = pd.concat([sample_raw_df, sample_raw_df.iloc[[0]]])
        result = clean_dataframe(df_dup, "total_cost")
        assert len(result) == 10

    def test_drops_na_rows(self, sample_raw_df):
        """Rows with missing values are removed."""
        sample_raw_df.loc[0, "Accommodation cost"] = None
        result = clean_dataframe(sample_raw_df, "total_cost")
        assert len(result) == 9

    def test_index_is_reset(self, sample_raw_df):
        """Index is sequential after cleaning."""
        sample_raw_df.loc[0, "Accommodation cost"] = None
        result = clean_dataframe(sample_raw_df, "total_cost")
        assert list(result.index) == list(range(len(result)))

    def test_idempotent(self, sample_raw_df):
        """Running clean_dataframe twice produces the same result."""
        first = clean_dataframe(sample_raw_df, "total_cost")
        second = clean_dataframe(sample_raw_df, "total_cost")
        pd.testing.assert_frame_equal(first, second)

    def test_missing_cost_columns_raises(self):
        """KeyError when accommodation/transportation cost columns are absent."""
        df_bad = pd.DataFrame({"Col A": [1], "Col B": [2]})
        with pytest.raises(KeyError):
            clean_dataframe(df_bad, "total_cost")

    def test_destination_country_created(self, sample_raw_df):
        """destination_country column is extracted from destination."""
        result = clean_dataframe(sample_raw_df, "total_cost")
        assert "destination_country" in result.columns

    def test_date_features_created(self, sample_raw_df):
        """travel_month and day_of_week are extracted from start_date."""
        result = clean_dataframe(sample_raw_df, "total_cost")
        assert "travel_month" in result.columns
        assert "day_of_week" in result.columns
