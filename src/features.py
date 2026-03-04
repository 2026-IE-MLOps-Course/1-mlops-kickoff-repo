"""
Module: features.py
-------------------
Role: Create, select, and define the transformation logic for engineered features.
Input: Cleaned pandas.DataFrame.
Output: A scikit-learn ColumnTransformer object (the "recipe") and the engineered DataFrame.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Configuration of logging to ensure traceability of the feature engineering process [cite: 14, 38]
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies domain-specific feature engineering to the dataset.

    Why: Transforming raw variables into predictive features allows the 
    Logistic Regression model to capture complex patterns, such as the 
    relationship between customer loyalty (tenure) and financial commitment[cite: 84].
    """
    try:
        logging.info("Starting feature engineering process...")
        df_eng = df.copy()

        # 1. Feature Selection: Removing unique identifiers that do not contribute to prediction
        if 'customerID' in df_eng.columns:
            df_eng = df_eng.drop(columns=['customerID'])
            logging.info("Successfully dropped 'customerID' identifier.")

        # 2. Variable Consistency: Ensuring TotalCharges is numeric and handling empty values
        # Why: TotalCharges often arrives as a string due to empty records for new customers
        if 'TotalCharges' in df_eng.columns:
            df_eng['TotalCharges'] = pd.to_numeric(
                df_eng['TotalCharges'], errors='coerce')
            df_eng['TotalCharges'] = df_eng['TotalCharges'].fillna(0)
            logging.info(
                "Converted 'TotalCharges' to numeric and handled nulls.")

        logging.info("Feature engineering completed successfully.")
        return df_eng

    except Exception as e:
        logging.error(f"Failed during feature engineering: {e}")
        raise e


def get_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Defines the ColumnTransformer (The ML 'Recipe').

    Why Industry Grade? 
    Using a ColumnTransformer instead of manual dummy encoding (pd.get_dummies) 
    prevents 'data leakage' and ensures that the exact same transformation 
    logic is applied during both training and real-time inference[cite: 22, 50].
    """
    try:
        logging.info("Defining the transformation preprocessor...")

        # Defining features based on the notebook experiment analysis
        numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
        categorical_features = [
            col for col in X.columns
            if col not in numeric_features and X[col].dtype == 'object'
        ]

        # Numeric Transformation: Scaling is vital for Logistic Regression stability and convergence
        numeric_transformer = StandardScaler()

        # Categorical Transformation: One-Hot Encoding for non-numeric features
        # handle_unknown='ignore' ensures the pipeline does not break if new categories appear in production
        categorical_transformer = OneHotEncoder(
            handle_unknown='ignore', drop='first')

        # Bundling all transformations into a single object for the Pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        logging.info(
            f"Preprocessor defined with {len(numeric_features)} numeric and "
            f"{len(categorical_features)} categorical transformation steps."
        )
        return preprocessor

    except Exception as e:
        logging.error(f"Error while defining the preprocessor: {e}")
        raise e
