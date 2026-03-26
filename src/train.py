"""
Module: train.py
----------------
Role: Trains a Logistic Regression model for Telco Churn prediction.
Input: Training features (DataFrame), Target (Series), and a Preprocessor.
Output: A fitted scikit-learn Pipeline saved in the 'models/' directory.
"""

import os
import logging
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Configuration  logging to track the training process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor, model_path: str = "models/model.joblib", max_iter: int = 2000, random_state: int = 42):
    """
    Fits a machine learning pipeline and saves the resulting artifact.

    Why a Pipeline?
    It encapsulates the preprocessing steps and the estimator into a single object.
    This prevents 'data leakage' during training and ensures that inference data
    undergoes the exact same transformations[cite: 5, 30].

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target (Churn).
        preprocessor: An unfitted scikit-learn ColumnTransformer.
        model_path (str): File path to save the trained model.
        max_iter (int): Maximum iterations for LogisticRegression convergence.
        random_state (int): Random seed for reproducibility.

    Returns:
        Pipeline: The fully fitted scikit-learn Pipeline.
    """

    logging.info("Initializing the ML Pipeline...")

    estimator = LogisticRegression(max_iter=max_iter, random_state=random_state)

    # Creating the pipeline
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator)
    ])

    try:
        logging.info("Fitting the model on the training data...")
        pipeline.fit(X_train, y_train)
        logging.info("Model training completed successfully.")

        # Ensure the output directory exists [cite: 50]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Artifacting: Saving the model for production use
        joblib.dump(pipeline, model_path)
        logging.info(f"Model artifact saved at: {model_path}")

    except Exception as e:
        # Comprehensive error handling [cite: 10, 50]
        logging.error(f"An error occurred during training or saving: {e}")
        raise e

    return pipeline


def load_trained_model(model_path: str = "models/model.joblib"):
    """
    Loads a serialized model artifact.

    Why: Essential for the inference.py module to apply the model to new data[cite: 25].
    """
    try:
        model = joblib.load(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_path}")
        raise
