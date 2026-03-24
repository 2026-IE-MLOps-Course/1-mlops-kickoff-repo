"""
Module: Model Training
----------------------
Role: Bundle preprocessing and algorithms into a single Pipeline and fit on training data.
Input: pandas.DataFrame (Processed) + ColumnTransformer (Recipe).
Output: Serialized scikit-learn Pipeline in `models/`.
"""
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import os


def train_model(df, config, preprocessor_fn):
    """Train a model using the preprocessor and training data."""
    # This is a stub - returns mock objects for testing
    pipeline = Pipeline([
        ("preprocessor", preprocessor_fn()),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/model.joblib")

    # Return mock test data and metrics
    metrics = {"train_f1": 0.75}
    return pipeline, metrics, None, None