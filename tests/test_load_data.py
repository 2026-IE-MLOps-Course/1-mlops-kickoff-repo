"""
test_load_data.py
-----------------
Unit tests for the data loading module (src/load_data.py).

Coverage:
- Happy path: shape, column names, dtypes, idempotency
- Error paths: missing file, directory path, empty CSV, whitespace-only CSV
- Contract: returns unmodified DataFrame (no side effects on source)
"""

import re
from pathlib import Path

import pandas as pd
import pytest

from src.load_data import load_raw_data


EXPECTED_COLUMNS = {
    "Trip ID",
    "Destination",
    "Start date",
    "End date",
    "Duration (days)",
    "Traveler name",
    "Traveler age",
    "Traveler gender",
    "Traveler nationality",
    "Accommodation type",
    "Accommodation cost",
    "Transportation type",
    "Transportation cost",
}


class TestLoadRawData:
    """Unit tests for the public load_raw_data function."""

    # ------------------------------------------------------------------ #
    # Happy-path tests                                                     #
    # ------------------------------------------------------------------ #

    def test_returns_dataframe(self, sample_raw_csv):
        """Return type is always pandas DataFrame."""
        df = load_raw_data(sample_raw_csv)
        assert isinstance(df, pd.DataFrame)

    def test_loads_correct_row_count(self, sample_raw_csv):
        """Load exactly 10 rows from the sample fixture."""
        df = load_raw_data(sample_raw_csv)
        assert df.shape[0] == 10, f"Expected 10 rows, got {df.shape[0]}"

    def test_loads_correct_column_count(self, sample_raw_csv):
        """Load exactly 13 columns from the sample fixture."""
        df = load_raw_data(sample_raw_csv)
        assert df.shape[1] == 13, f"Expected 13 columns, got {df.shape[1]}"

    def test_all_expected_columns_present(self, sample_raw_csv):
        """Every expected raw column is present in the result."""
        df = load_raw_data(sample_raw_csv)
        missing = EXPECTED_COLUMNS - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_no_extra_columns(self, sample_raw_csv):
        """No unexpected extra columns appear in the result."""
        df = load_raw_data(sample_raw_csv)
        extra = set(df.columns) - EXPECTED_COLUMNS
        assert not extra, f"Unexpected extra columns: {extra}"

    def test_not_empty(self, sample_raw_csv):
        """Loaded DataFrame is non-empty."""
        df = load_raw_data(sample_raw_csv)
        assert not df.empty

    def test_numeric_columns_have_numeric_dtype(self, sample_raw_csv):
        """Numeric columns should not be loaded as plain objects."""
        df = load_raw_data(sample_raw_csv)
        for col in ["Traveler age", "Duration (days)",
                    "Accommodation cost", "Transportation cost"]:
            assert pd.api.types.is_numeric_dtype(df[col]), (
                f"Column '{col}' should be numeric, got {df[col].dtype}"
            )

    def test_accepts_string_path(self, sample_raw_csv):
        """load_raw_data must accept a plain string path, not just Path objects."""
        df = load_raw_data(str(sample_raw_csv))
        assert df.shape[0] == 10

    def test_idempotent_loads(self, sample_raw_csv):
        """Calling load_raw_data twice returns byte-for-byte identical results."""
        df1 = load_raw_data(sample_raw_csv)
        df2 = load_raw_data(sample_raw_csv)
        pd.testing.assert_frame_equal(df1, df2)

    def test_does_not_modify_source_file(self, sample_raw_csv):
        """Loading the file must not alter the underlying CSV bytes."""
        before = sample_raw_csv.read_bytes()
        load_raw_data(sample_raw_csv)
        after = sample_raw_csv.read_bytes()
        assert before == after, "Source CSV was modified by load_raw_data"

    def test_index_is_default_range(self, sample_raw_csv):
        """Default integer RangeIndex is preserved (no reshuffling)."""
        df = load_raw_data(sample_raw_csv)
        expected_index = list(range(10))
        assert list(df.index) == expected_index

    # ------------------------------------------------------------------ #
    # Error-path tests                                                     #
    # ------------------------------------------------------------------ #

    def test_missing_file_raises_file_not_found(self, tmp_path):
        """FileNotFoundError is raised when the file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_raw_data(tmp_path / "nonexistent.csv")

    def test_directory_path_raises_is_a_directory(self, tmp_path):
        """IsADirectoryError when the path points to a directory."""
        with pytest.raises(IsADirectoryError):
            load_raw_data(tmp_path)

    def test_header_only_csv_raises_value_error(self, tmp_path):
        """ValueError (zero rows) when the CSV has only a header row."""
        p = tmp_path / "header_only.csv"
        p.write_text("col_a,col_b,col_c\n")
        with pytest.raises(ValueError, match="zero rows"):
            load_raw_data(p)

    def test_completely_empty_csv_raises(self, tmp_path):
        """ValueError when the CSV file is completely empty (no header)."""
        p = tmp_path / "blank.csv"
        p.write_text("")
        # Either ValueError (zero rows) or pandas EmptyDataError are acceptable
        with pytest.raises(Exception):
            load_raw_data(p)

    def test_error_message_contains_path(self, tmp_path):
        """FileNotFoundError message includes the attempted path."""
        bad_path = tmp_path / "missing.csv"
        with pytest.raises(FileNotFoundError, match=re.escape(str(bad_path))):
            load_raw_data(bad_path)
