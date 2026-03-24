"""
Module: Data Loader
-------------------
Role: Ingest raw data from sources (CSV, SQL, API).
Input: Path to file or connection string.
Output: pandas.DataFrame (Raw).
"""
import pandas as pd


def load_data(path):
    """Load raw data from CSV file."""
    return pd.read_csv(path)