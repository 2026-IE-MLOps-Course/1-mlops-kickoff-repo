import pandas as pd
import pytest

from src.clean_data import clean_dataframe


def test_clean_dataframe_renames_rx_ds_and_dedupes():
    df_raw = pd.DataFrame(
        {
            "rx ds": [10, 10],
            "OD": [0, 0],
            "A": [1, 1],
        }
    )
    df_clean = clean_dataframe(df_raw, target_column="OD")

    assert "rx_ds" in df_clean.columns
    assert "rx ds" not in df_clean.columns
    assert len(df_clean) == 1


def test_clean_dataframe_requires_target():
    df_raw = pd.DataFrame({"rx ds": [10], "A": [1]})
    with pytest.raises(ValueError):
        clean_dataframe(df_raw, target_column="OD")


def test_clean_dataframe_coerces_binary_flags():
    df_raw = pd.DataFrame(
        {
            "rx ds": [10, 20],
            "OD": [0, 1],
            "A": [True, False],
        }
    )
    df_clean = clean_dataframe(df_raw, target_column="OD")
    assert set(df_clean["A"].unique()) <= {0, 1}