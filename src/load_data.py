"""
Module: Data Loader
-------------------
Role: Ingest raw data from sources (CSV, SQL, API).
Input: Path to file or connection string.
Output: pandas.DataFrame (Raw).
"""


"""
Educational Goal:
- Why this module exists in an MLOps system: Isolates data extraction from downstream processing.
- Responsibility (separation of concerns): Acquiring data from external systems, databases, or local files.
- Pipeline contract (inputs and outputs): Takes a source location, outputs a raw DataFrame.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path
from src.utils import load_csv, save_csv
import pandas as pd


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Inputs:
    - raw_data_path: Path to the raw data file.
    Outputs:
    - pd.DataFrame containing the raw, unmodified data.
    Why this contract matters for reliable ML delivery:
    - Provides a single entry point for data ingestion, making it easy to swap flat files for database queries later.
    """
    print(
        f"Executing load_raw_data for {raw_data_path}")  # TODO: replace with logging later

    if not raw_data_path.exists():
        print(f"!!! LOUD WARNING: {raw_data_path} not found !!!")
        print("Generating a dummy dataset for scaffolding only. You MUST replace this and update SETTINGS.")
        dummy_df = pd.DataFrame({
            "num_feature": [1.5, 2.1, 3.8, 1.1, 4.0, 5.5, 2.2, 3.1, 4.5, 1.9],
            "cat_feature": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"],
            "target": [0, 1, 0, 1, 1, 0, 1, 0, 1, 0]
        })
        save_csv(dummy_df, raw_data_path)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Real data extraction might require SQL queries, API calls, or joining multiple tables.
    # Examples:
    # 1. df = pd.read_sql("SELECT * FROM table", engine)
    # 2. df = requests.get(api_endpoint).json()
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return load_csv(raw_data_path)
