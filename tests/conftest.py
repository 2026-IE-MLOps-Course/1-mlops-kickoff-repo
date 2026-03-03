"""Shared fixtures for all test modules."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_raw_df():
    """Mimics a small slice of the raw NHL CSV."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "Rank": range(1, n + 1),
        "Name": [f"Player_{i}" for i in range(n)],
        "Team": np.random.choice(["TOR", "MTL", "BOS"], n),
        "Pos": np.random.choice(["C", "L", "R", "D"], n),
        "Goals": np.random.randint(0, 40, n),
        "Assists": np.random.randint(0, 50, n),
        "Primary_Assists": np.random.randint(0, 30, n),
        "Secondary_Assists": np.random.randint(0, 20, n),
        "Icetime_Minutes": np.random.uniform(200, 1400, n),
        "Shot_Attempts": np.random.randint(50, 300, n),
        "Faceoff_Win_Pct": np.random.uniform(30, 70, n),
        "Takeaways": np.random.randint(0, 80, n),
        "Giveaways": np.random.randint(0, 80, n),
        "Shooting_Pct_On_Unblocked": np.random.uniform(0, 25, n),
        "PIM_Drawn": np.random.randint(0, 30, n),
        "Pct_Shift_Starts_Offensive_Zone": np.random.uniform(30, 70, n),
        "On_Ice_Corsi_Pct": np.random.uniform(40, 60, n),
    })


@pytest.fixture
def mock_clean_df(mock_raw_df):
    """A cleaned version of mock_raw_df (target computed, leakage dropped)."""
    from src.clean_data import clean_dataframe
    return clean_dataframe(mock_raw_df, "Points")


@pytest.fixture
def tmp_csv(tmp_path, mock_raw_df):
    """Write mock_raw_df to a temp CSV and return the path."""
    path = tmp_path / "test.csv"
    mock_raw_df.to_csv(path, index=False)
    return path
