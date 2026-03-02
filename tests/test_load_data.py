from pathlib import Path

import pandas as pd
import pytest

from src.load_data import load_raw_data


def test_load_raw_data_happy_path(tmp_path: Path):
    p = tmp_path / "raw.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(p, index=False)

    df = load_raw_data(p)
    assert df.shape == (2, 2)


def test_load_raw_data_missing_file(tmp_path: Path):
    p = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_raw_data(p)


def test_load_raw_data_directory_path(tmp_path: Path):
    d = tmp_path / "folder"
    d.mkdir()
    with pytest.raises(IsADirectoryError):
        load_raw_data(d)


def test_load_raw_data_empty_file(tmp_path: Path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(ValueError):
        load_raw_data(p)