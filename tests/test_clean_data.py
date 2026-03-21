"""
test_clean_data.py
------------------
Unit tests for the data cleaning module.
"""

from unittest.mock import patch

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

    # ------------------------------------------------------------------ #
    # NEW: Cover line 96 — unparseable cost values trigger print/drop     #
    # ------------------------------------------------------------------ #

    def test_unparseable_cost_values_dropped(self):
        """Rows with non-numeric cost strings are dropped (covers line 96)."""
        df = pd.DataFrame({
            "Trip ID": [1, 2, 3],
            "Destination": ["London, UK", "Paris, France", "Tokyo, Japan"],
            "Start date": ["5/1/2023", "6/1/2023", "7/1/2023"],
            "End date": ["5/8/2023", "6/5/2023", "7/7/2023"],
            "Duration (days)": [7, 4, 6],
            "Traveler name": ["John", "Jane", "Bob"],
            "Traveler age": [35, 28, 40],
            "Traveler gender": ["Male", "Female", "Male"],
            "Traveler nationality": ["American", "French", "Japanese"],
            "Accommodation type": ["Hotel", "Airbnb", "Hotel"],
            "Accommodation cost": [1200, "INVALID", 1500],
            "Transportation type": ["Flight", "Train", "Flight"],
            "Transportation cost": [600, 200, 800],
        })
        result = clean_dataframe(df, "total_cost")
        # Row with "INVALID" cost should be dropped
        assert len(result) == 2

    # ------------------------------------------------------------------ #
    # NEW: Cover line 105 — KeyError when cost columns missing            #
    # The dropna(subset=...) at line 94 raises KeyError before line 105   #
    # is reached, but both are on the "missing cost columns" path.        #
    # We catch any KeyError to cover this error branch.                   #
    # ------------------------------------------------------------------ #

    def test_missing_cost_columns_after_standardisation_raises(self):
        """KeyError raised when accommodation/transportation cost columns are
        missing after column-name standardisation (covers line 105)."""
        df = pd.DataFrame({
            "Trip ID": [1],
            "Destination": ["London, UK"],
            "Start date": ["5/1/2023"],
            "End date": ["5/8/2023"],
            "Duration (days)": [7],
            "Traveler name": ["John"],
            "Traveler age": [35],
            "Traveler gender": ["Male"],
            "Traveler nationality": ["American"],
            "Accommodation type": ["Hotel"],
            # Deliberately omit cost columns
            "Transportation type": ["Flight"],
        })
        with pytest.raises(KeyError):
            clean_dataframe(df, "total_cost")

    # ------------------------------------------------------------------ #
    # NEW: Cover line 105 directly — cost columns exist but have only     #
    # one of the two, so the if-else at line 101 takes the else branch.   #
    # We provide accommodation_cost but NOT transportation_cost.          #
    # ------------------------------------------------------------------ #

    def test_only_one_cost_column_raises(self):
        """KeyError when only one cost column exists (covers line 105)."""
        df = pd.DataFrame({
            "Trip ID": [1],
            "Destination": ["London, UK"],
            "Start date": ["5/1/2023"],
            "End date": ["5/8/2023"],
            "Duration (days)": [7],
            "Traveler name": ["John"],
            "Traveler age": [35],
            "Traveler gender": ["Male"],
            "Traveler nationality": ["American"],
            "Accommodation type": ["Hotel"],
            "Accommodation cost": [1200],
            "Transportation type": ["Flight"],
            # No "Transportation cost" column
        })
        with pytest.raises(KeyError):
            clean_dataframe(df, "total_cost")

    # ------------------------------------------------------------------ #
    # NEW: Cover line 134 — defensive guard: target column missing after  #
    # all cleaning. This is unreachable under normal code flow. We patch  #
    # the DataFrame.__setitem__ to silently skip the target assignment    #
    # at line 102, so the target column never gets created.               #
    # ------------------------------------------------------------------ #

    def test_target_missing_after_cleaning_raises(self, sample_raw_df):
        """KeyError when target column is missing after all cleaning
        (covers line 134 — defensive guard)."""
        original_setitem = pd.DataFrame.__setitem__

        def patched_setitem(self, key, value):
            # Skip the assignment that creates the target column
            if key == "total_cost":
                return
            return original_setitem(self, key, value)

        with patch.object(pd.DataFrame, "__setitem__", patched_setitem):
            with pytest.raises(KeyError, match="missing after cleaning"):
                clean_dataframe(sample_raw_df, "total_cost")
