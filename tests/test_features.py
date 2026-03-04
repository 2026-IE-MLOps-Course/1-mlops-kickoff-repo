# python -m pytest tests/test_features.py

import pytest
import pandas as pd
import numpy as np
# Import logic from your modular src files
from src.features import engineer_features, get_preprocessor


@pytest.fixture
def sample_data():
    """Industry Grade: Use a fixture to provide consistent mock data for tests."""
    data = {
        'customerID': ['7590-VHVEG', '5575-GNVDE'],
        'tenure': [1, 34],
        'MonthlyCharges': [29.85, 56.95],
        'TotalCharges': ['29.85', '1889.5'],
        'InternetService': ['DSL', 'DSL'],
        'Churn': ['No', 'No']
    }
    return pd.DataFrame(data)


def test_engineer_features_output(sample_data):
    """
    Why: Ensures modular code replicates the notebook's data cleaning.
    Checks dropping IDs and converting TotalCharges to numeric.
    """
    df_engineered = engineer_features(sample_data)
    assert 'customerID' not in df_engineered.columns
    assert pd.api.types.is_numeric_dtype(df_engineered['TotalCharges'])


def test_preprocessor_scaling(sample_data):
    """
    Why: Ensures the ColumnTransformer correctly scales numeric features.
    Critical for Logistic Regression convergence[cite: 77].
    """
    df_eng = engineer_features(sample_data)
    X = df_eng.drop(columns=['Churn'])
    preprocessor = get_preprocessor(X)
    X_transformed = preprocessor.fit_transform(X)

    # Assert that preprocessor scales the 3 numeric features
    assert X_transformed.shape[1] >= 3
    # Scaled data should have a mean near 0
    assert np.allclose(X_transformed[:, :3].mean(), 0, atol=1e-1)
