"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature engineering.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""


"""
Educational Goal:
- Why this module exists in an MLOps system: Separates domain-specific cleaning rules from pure ML feature engineering.
- Responsibility (separation of concerns): Handling missing values, dropping duplicates, and correcting data types.
- Pipeline contract (inputs and outputs): Takes a raw DataFrame, outputs a cleaned DataFrame ready for split and feature engineering.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""


import pandas as pd
def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Inputs:
    - df_raw: The raw input DataFrame.
    - target_column: The name of the target column (to ensure it isn't accidentally dropped).
    Outputs:
    - pd.DataFrame that has been cleaned.
    Why this contract matters for reliable ML delivery:
    - Ensures data quality rules are applied uniformly before any modeling steps occur.
    """
    print("Executing clean_dataframe")  # TODO: replace with logging later
    df_clean = df_raw.copy()

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Every dataset has unique garbage values, missingness patterns, and invalid outliers.
    # Examples:
    # 1. df_clean = df_clean.dropna(subset=[target_column])
    # 2. df_clean['age'] = df_clean['age'].fillna(df_clean['age'].median())
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return df_clean
