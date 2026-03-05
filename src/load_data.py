"""
Educational Goal:
- Why this module exists in an MLOps system: Ingesting raw data is the very first
  step of any ML pipeline. A dedicated loader enforces file-dependency checks,
  guards against empty data, and centralises CSV parsing so every downstream
  module trusts the data shape.
- Responsibility (separation of concerns): Load raw data from disk.
  Never mutate the source file. Return an unmodified DataFrame.
- Pipeline contract (inputs and outputs):
  Input  — raw_data_path (pathlib.Path) pointing to the CSV on disk.
  Output — pd.DataFrame containing the raw, unmodified data.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path

import pandas as pd

from src.utils import load_csv


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Inputs:
    - raw_data_path (Path): File system path to the raw CSV dataset.
    Outputs:
    - pd.DataFrame: Raw data loaded from disk, unmodified.
    Why this contract matters for reliable ML delivery:
    - Enforcing file-dependency checks and empty-data guards at load time
      means that problems are caught immediately, before expensive cleaning,
      feature engineering, or training steps waste compute.
    """
    print(f"[load_data] Loading raw data from: {raw_data_path}")  # TODO: replace with logging later

    raw_data_path = Path(raw_data_path)

    # Enforce file dependency — fail immediately if missing
    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {raw_data_path}. "
            "Please place the dataset in the configured path."
        )

    # Validate the path type — must be a file, not a directory
    if raw_data_path.is_dir():
        raise IsADirectoryError(
            f"Expected a file but got a directory: {raw_data_path}"
        )

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Each dataset has unique file formats, encodings, and separators
    # Examples:
    # 1. df = load_csv(raw_data_path)  # for standard CSV
    # 2. df = pd.read_excel(raw_data_path)  # for Excel files
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Student implementation for VoyageIQ travel dataset:
    df = load_csv(raw_data_path)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    # Guard against empty data
    if df.empty:
        raise ValueError(
            f"Loaded DataFrame from {raw_data_path} has zero rows. "
            "Check the upstream data export."
        )

    print(  # TODO: replace with logging later
        f"[load_data] Data loaded successfully — "
        f"shape: {df.shape[0]} rows x {df.shape[1]} columns"
    )
    return df