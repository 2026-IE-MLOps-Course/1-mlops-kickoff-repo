"""
Educational Goal:
- Why this module exists in an MLOps system: Data validation acts as a quality gate
  between raw ingestion and model training. It catches schema mismatches, empty
  dataframes, and obvious data issues early — before they silently corrupt downstream
  steps or produce misleading model metrics.
- Responsibility (separation of concerns): This module only checks data quality and
  raises errors or returns a boolean. It does NOT clean, transform, or load data.
- Pipeline contract (inputs and outputs): Receives a cleaned pandas DataFrame and a
  list of required column names. Returns True if the data passes all checks, or raises
  ValueError immediately on the first failure (fail-fast pattern).

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Inputs:
    - df: A pandas DataFrame to validate (typically the cleaned DataFrame).
    - required_columns: A list of column name strings that must be present in df.

    Outputs:
    - True if all validation checks pass.
    - Raises ValueError immediately if any check fails (fail-fast).

    Why this contract matters for reliable ML delivery:
    - Catching an empty DataFrame or a missing column here prevents cryptic errors
      deep inside the feature engineering or training steps, which are much harder
      to debug. A loud, early failure is always preferable to a silent wrong result.
    - Passing required_columns as an argument (not hardcoding them) keeps this
      function reusable across different datasets and projects.
    """
    print("validate_dataframe: Starting data validation checks...")  # TODO: replace with logging later

    # --- Baseline check 1: Fail fast on empty DataFrame ---
    if df.empty:
        raise ValueError(
            "Validation failed: The DataFrame is empty. "
            "Check your load_data and clean_data steps."
        )
    print(f"validate_dataframe: DataFrame shape is {df.shape} — not empty.")  # TODO: replace with logging later

    # --- Baseline check 2: Fail fast on missing required columns ---
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Validation failed: The following required columns are missing from the "
            f"DataFrame: {missing_cols}. "
            f"Columns present: {df.columns.tolist()}"
        )
    print(f"validate_dataframe: All required columns present: {required_columns}")  # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------

    # Check 1: No missing values in any column
    for col in df.columns:
        if df[col].isnull().any():
            raise ValueError(f"Validation failed: Column '{col}' contains missing values.")
    print("validate_dataframe: No missing values found.")  # TODO: replace with logging later

    # Check 2: No duplicate rows
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        raise ValueError(f"Validation failed: Dataset contains {n_duplicates} duplicate row(s).")
    print(f"validate_dataframe: No duplicate rows found.")  # TODO: replace with logging later

    # Check 3: Non-negative values in numeric columns
    non_negative_cols = [
        "AccountWeeks", "DataUsage", "CustServCalls",
        "DayMins", "DayCalls", "MonthlyCharge", "OverageFee", "RoamMins",
    ]
    for col in non_negative_cols:
        if (df[col] < 0).any():
            raise ValueError(f"Validation failed: Column '{col}' contains negative values.")
    print("validate_dataframe: All numeric columns are non-negative.")  # TODO: replace with logging later

    # Check 4: Binary columns contain only 0 or 1
    binary_cols = ["Churn", "ContractRenewal", "DataPlan"]
    for col in binary_cols:
        invalid = df[col][~df[col].isin([0, 1])]
        if not invalid.empty:
            raise ValueError(
                f"Validation failed: Column '{col}' contains values outside {{0, 1}}: "
                f"{invalid.unique().tolist()}"
            )
    print("validate_dataframe: All binary columns contain only 0 and 1.")  # TODO: replace with logging later

    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    print("validate_dataframe: All validation checks passed.")  # TODO: replace with logging later
    return True


if __name__ == "__main__":
    from src.load_data import load_data

    REQUIRED_COLUMNS = [
        "Churn",
        "AccountWeeks",
        "ContractRenewal",
        "DataPlan",
        "DataUsage",
        "CustServCalls",
        "DayMins",
        "DayCalls",
        "MonthlyCharge",
        "OverageFee",
        "RoamMins",
    ]

    df = load_data()
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    result = validate_dataframe(df, required_columns=REQUIRED_COLUMNS)
    print(f"Validation result: {result}")
