"""
Module: Data Loading
--------------------
Role: Load the raw CSV dataset into a pandas DataFrame.
      If the file is missing, generate a small dummy dataset so the
      pipeline can still run end-to-end during development.
Input: Path to a CSV file.
Output: Raw pandas DataFrame.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Load the raw NHL dataset from *raw_data_path*.

    If the file does not exist a small dummy DataFrame is created and
    saved so that the rest of the pipeline can be exercised without
    the real data.

    Args:
        raw_data_path: Path to the raw CSV file.

    Returns:
        Raw DataFrame (unmodified).
    """
    path = Path(raw_data_path)

    if path.exists():
        print(f"[load_data] Loading CSV from {path}")
        return pd.read_csv(path)

    print(f"[load_data] {path} not found — generating dummy data")
    np.random.seed(42)
    n = 100
    dummy = pd.DataFrame({
        "Rank": range(1, n + 1),
        "Name": [f"Player_{i}" for i in range(n)],
        "Team": np.random.choice(["TOR", "MTL", "BOS", "NYR"], n),
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

    path.parent.mkdir(parents=True, exist_ok=True)
    dummy.to_csv(path, index=False)
    print(f"[load_data] Dummy data saved to {path}")
    return dummy
