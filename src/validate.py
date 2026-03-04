"""
Module: Data Validation
-----------------------
Role: Check data quality (schema, types, ranges) before training.
Input: pandas.DataFrame.
Output: Boolean (True if valid) or raises Error.
"""


"""
Educational Goal:
- Why this module exists in an MLOps system: Acts as a quality gate to prevent bad data from silently ruining models.
- Responsibility (separation of concerns): Fails fast if data expectations (schema, nulls, ranges) are violated.
- Pipeline contract (inputs and outputs): Takes a DataFrame and required schema, returns a boolean or raises an Exception.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""


import pandas as pd
def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Inputs:
    - df: The DataFrame to validate.
    - required_columns: List of string column names that must exist in the DataFrame.
    Outputs:
    - True if validation passes, otherwise raises ValueError.
    Why this contract matters for reliable ML delivery:
    - Catches schema drift or data corruption immediately, rather than failing deep inside the training loop.
    """
    print("Executing validate_dataframe")  # TODO: replace with logging later

    if df.empty:
        raise ValueError(
            "Validation Failed: The DataFrame is completely empty!")

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Validation Failed: Required column '{col}' is missing from the dataset.")

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: You might need to check for specific data types, value ranges, or allowed categories.
    # Examples:
    # 1. assert df['age'].min() >= 0, "Age cannot be negative"
    # 2. assert df['status'].isin(['active', 'inactive']).all()
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    # TODO: replace with logging later
    print("Validation passed successfully.")
    return True
