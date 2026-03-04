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
from src.load_data import load_raw_data
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

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    load_csv = pd.read_csv("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return load_csv(raw_data_path)
