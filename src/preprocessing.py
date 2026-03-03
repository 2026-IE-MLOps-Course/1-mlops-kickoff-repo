"""
Module: Preprocessing
---------------------
Role: Prepare cleaned data for modeling — separate features from
      target, split into train/test sets, and apply any additional
      transformations before the pipeline.
Input: Cleaned DataFrame, target column name, and split parameters.
Output: Train/test splits ready for the modeling pipeline.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def separate_features_target(df: pd.DataFrame,
                             target_column: str):
    """
    Split a DataFrame into feature matrix X and target vector y.

    Args:
        df: Cleaned DataFrame containing both features and target.
        target_column: Name of the target column.

    Returns:
        Tuple of (X, y) where X is a DataFrame and y is a Series.

    Raises:
        ValueError: If target_column is not in the DataFrame.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    X = df.drop(columns=[target_column])
    y = df[target_column]
    print(f"[preprocessing] Features: {X.shape[1]} columns, "
          f"Target: '{target_column}'")
    return X, y


def split_train_test(X: pd.DataFrame, y: pd.Series,
                     test_size: float = 0.2,
                     random_state: int = 42):
    """
    Split features and target into training and test sets.

    Attempts stratified splitting first (useful for classification).
    Falls back to a standard random split if stratification fails
    (e.g. continuous regression targets).

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Fraction of data to reserve for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
        )

    print(f"[preprocessing] Train: {len(X_train)}, Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def preprocess_splits(df: pd.DataFrame, target_column: str,
                      test_size: float = 0.2,
                      random_state: int = 42):
    """
    End-to-end preprocessing: separate target then split.

    Convenience wrapper that calls separate_features_target and
    split_train_test in sequence.

    Args:
        df: Cleaned DataFrame.
        target_column: Name of the target column.
        test_size: Fraction of data for testing.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X, y = separate_features_target(df, target_column)
    return split_train_test(X, y, test_size, random_state)
