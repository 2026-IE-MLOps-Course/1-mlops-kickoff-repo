import pytest
import pandas as pd
from src.features import engineer_features, get_preprocessor
from sklearn.compose import ColumnTransformer


@pytest.fixture
def clean_data():
    return pd.DataFrame({
        "customerID": ["123", "456"],
        "tenure": [10, 20],
        "MonthlyCharges": [10.0, 20.0],
        "TotalCharges": ["100.0", ""],
        "InternetService": ["DSL", "Fiber optic"]
    })


def test_engineer_features(clean_data):
    """Verifica eliminación de IDs y corrección de nulos."""
    df_eng = engineer_features(clean_data)
    assert "customerID" not in df_eng.columns
    assert pd.api.types.is_numeric_dtype(df_eng["TotalCharges"])


def test_get_preprocessor(clean_data):
    """Verifica que se devuelva el objeto ColumnTransformer."""
    df_eng = engineer_features(clean_data)
    preprocessor = get_preprocessor(df_eng)
    assert isinstance(preprocessor, ColumnTransformer)
