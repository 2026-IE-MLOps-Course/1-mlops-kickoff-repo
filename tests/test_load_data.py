"""
Module: Test Data Loader
------------------------
Tests for src/load_data.py
"""
import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import patch
from src.load_data import load_raw_data


@patch("pandas.read_csv")
def test_load_raw_data(mock_read_csv):
    """Verifica que los datos se carguen pasando la ruta correcta a pandas."""
    mock_read_csv.return_value = pd.DataFrame({"col1": [1, 2, 3]})
    df = load_raw_data("dummy/path.csv")
    assert not df.empty
    assert "col1" in df.columns
    mock_read_csv.assert_called_once_with("dummy/path.csv")


def test_load_raw_data_returns_dataframe(tmp_path):
    """load_raw_data must return a DataFrame when file exists."""
    dummy_csv = tmp_path / "test_data.csv"
    dummy_csv.write_text("col1,col2\n1,a\n2,b\n")
    result = load_raw_data(dummy_csv)
    assert isinstance(result, pd.DataFrame)


def test_load_raw_data_correct_shape(tmp_path):
    """DataFrame must have the expected number of rows and columns."""
    dummy_csv = tmp_path / "test_data.csv"
    dummy_csv.write_text("col1,col2\n1,a\n2,b\n3,c\n")
    result = load_raw_data(dummy_csv)
    assert result.shape == (3, 2)


def test_load_raw_data_correct_columns(tmp_path):
    """DataFrame must preserve column names from the CSV."""
    dummy_csv = tmp_path / "test_data.csv"
    dummy_csv.write_text("col1,col2\n1,a\n2,b\n")
    result = load_raw_data(dummy_csv)
    assert list(result.columns) == ["col1", "col2"]


def test_load_raw_data_file_not_found():
    """load_raw_data must raise FileNotFoundError for missing files."""
    missing_path = Path("non/existent/file.csv")
    with pytest.raises(FileNotFoundError):
        load_raw_data(missing_path)
