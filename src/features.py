"""
Module: features.py
-------------------
Role: Create, select, and define the transformation logic for engineered features.
"""
import logging
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Iniciando feature engineering...")
    df_eng = df.copy()
    if 'customerID' in df_eng.columns:
        df_eng = df_eng.drop(columns=['customerID'])
    if 'TotalCharges' in df_eng.columns:
        df_eng['TotalCharges'] = pd.to_numeric(
            df_eng['TotalCharges'], errors='coerce')
        df_eng['TotalCharges'] = df_eng['TotalCharges'].fillna(0)
    return df_eng


def get_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Defines the ColumnTransformer.
    Why: Prevents 'data leakage' and ensures consistency.
    """
    logger.info("Definiendo el preprocesador...")
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [
        col for col in X.columns if col not in numeric_features and X[col].dtype == 'object']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore',
             drop='first'), categorical_features)
        ]
    )
    return preprocessor
