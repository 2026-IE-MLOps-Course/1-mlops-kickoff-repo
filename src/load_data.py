"""
Module: Data Loader
-------------------
Role: Ingest raw data from sources (CSV, SQL, API).
Input: Path to file or connection string.
Output: pandas.DataFrame (Raw).
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Data ingestion is a frequent failure point; isolating it makes sources swappable and testable.
- Responsibility (separation of concerns): Only load raw data (and in scaffolding, create deterministic dummy data if missing).
- Pipeline contract (inputs and outputs): Input is a path to raw CSV; output is a DataFrame with expected columns.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path

import pandas as pd

from src.utils import load_csv, save_csv


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Inputs:
    - raw_data_path: Path to the raw CSV file.
    Outputs:
    - df_raw: Raw DataFrame loaded from disk (or created as deterministic dummy scaffolding).
    Why this contract matters for reliable ML delivery:
    - Keeping ingestion deterministic and explicit prevents "it worked on my notebook" surprises in CI and production.
    """
    print(f"[load_data.load_raw_data] Loading raw data from: {raw_data_path}")  # TODO: replace with logging later

    if not raw_data_path.exists():
        raw_data_path.parent.mkdir(parents=True, exist_ok=True)

        # Deterministic, tiny scaffolding dataset
        df_dummy = pd.DataFrame(
            {
                "num_feature": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "cat_feature": ["A", "B", "A", "C", "B", "C"],
                "target": [0.0, 1.0, 1.5, 2.5, 3.0, 3.5],
            }
        )

        print(
            "\n" + "!" * 80 + "\n"
            "[SCAFFOLDING ONLY] Raw dataset file was missing.\n"
            f"Created a DUMMY CSV at: {raw_data_path}\n"
            "Columns are hardcoded to: ['num_feature', 'cat_feature', 'target']\n"
            "Students MUST replace this with real ingestion and update SETTINGS in src/main.py.\n"
            + "!" * 80 + "\n"
        )  # TODO: replace with logging later

        save_csv(df_dummy, raw_data_path)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Replace this section to load your real dataset (DB query, parquet, APIs, feature store)
    # Why: Real ingestion varies by data platform, security constraints, and freshness requirements
    # Examples:
    # 1. Read from parquet + partition filters
    # 2. Pull from warehouse and materialize to data/raw/
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    df_raw = load_csv(raw_data_path)
    return df_raw