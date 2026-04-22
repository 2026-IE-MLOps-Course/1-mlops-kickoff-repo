"""
test_utils.py
-------------
Unit tests for the shared utility functions.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.utils import load_csv, save_csv, save_model, load_model


class TestLoadCsv:
    """Tests for the CSV loading utility."""

    def test_loads_valid_csv(self, tmp_path):
        """A well-formed CSV is loaded as a DataFrame."""
        p = tmp_path / "test.csv"
        p.write_text("col_a,col_b\n1,2\n3,4\n")
        df = load_csv(p)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    def test_missing_file_raises(self, tmp_path):
        """FileNotFoundError propagated when file is missing."""
        with pytest.raises(Exception):
            load_csv(tmp_path / "nonexistent.csv")

    # ------------------------------------------------------------------ #
    # NEW: Cover line 49 — UnicodeDecodeError fallback to latin-1         #
    # ------------------------------------------------------------------ #

    def test_latin1_fallback_on_unicode_error(self, tmp_path):
        """When utf-8-sig decoding fails, load_csv falls back to latin-1
        (covers line 49)."""
        p = tmp_path / "latin1.csv"
        # Write a CSV with latin-1 encoding containing non-UTF-8 bytes
        content = "col_a,col_b\ncaf\xe9,1\nstra\xdfe,2\n"
        p.write_bytes(content.encode("latin-1"))
        df = load_csv(p)
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    # ------------------------------------------------------------------ #
    # NEW: Cover line 51 — ParserError re-raise                           #
    # ------------------------------------------------------------------ #

    def test_parser_error_reraise(self, tmp_path):
        """When pandas raises a ParserError, load_csv re-raises it with
        a descriptive message (covers line 51)."""
        p = tmp_path / "valid.csv"
        p.write_text("col_a,col_b\n1,2\n")

        with patch("src.utils.pd.read_csv",
                   side_effect=pd.errors.ParserError("mock parse failure")):
            with pytest.raises(pd.errors.ParserError, match="Failed to parse"):
                load_csv(p)


class TestSaveCsv:
    """Tests for the CSV saving utility."""

    def test_creates_file(self, tmp_path):
        """CSV file is created at the given path."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        path = tmp_path / "sub" / "output.csv"
        save_csv(df, path)
        assert path.is_file()

    def test_creates_parent_dirs(self, tmp_path):
        """Parent directories are created automatically."""
        df = pd.DataFrame({"a": [1]})
        path = tmp_path / "deep" / "nested" / "dir" / "out.csv"
        save_csv(df, path)
        assert path.is_file()

    def test_roundtrip(self, tmp_path):
        """Data survives a save/load roundtrip."""
        df = pd.DataFrame({"x": [10, 20], "y": [30, 40]})
        path = tmp_path / "round.csv"
        save_csv(df, path)
        loaded = load_csv(path)
        pd.testing.assert_frame_equal(df, loaded)


class TestSaveLoadModel:
    """Tests for model serialisation utilities."""

    def test_save_creates_file(self, tmp_path):
        """Model file is created at the given path."""
        from sklearn.linear_model import Ridge
        model = Ridge()
        path = tmp_path / "models" / "test.joblib"
        save_model(model, path)
        assert path.is_file()

    def test_roundtrip(self, tmp_path):
        """A saved model can be loaded back."""
        from sklearn.linear_model import Ridge
        model = Ridge()
        path = tmp_path / "test.joblib"
        save_model(model, path)
        loaded = load_model(path)
        assert hasattr(loaded, "fit")
        assert hasattr(loaded, "predict")

    def test_load_missing_raises(self, tmp_path):
        """FileNotFoundError when model file does not exist."""
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent.joblib")
