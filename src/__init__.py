# src/__init__.py
"""
MLOps Pipeline Package

This package contains the modular components of the ML pipeline:
- Data loading
- Data cleaning
- Feature engineering
- Model training
- Evaluation
- Inference
"""

# Data loading.
from .load_data import load_raw_data

# Data cleaning
from .clean_data import load_raw, clean_data, save_df, validate_df

# Feature engineering
from .features import get_preprocessor

# Model training
from .train import train_model

# Model evaluation
from .evaluate import evaluate_model

# Inference
from .infer import run_inference

__all__ = [
    # Loading
    "load_raw_data",

    # Cleaning
    "load_raw",
    "clean_data",
    "save_df",
    "validate_df",

    # Features
    "get_preprocessor",

    # Training
    "train_model",

    # Evaluation
    "evaluate_model",

    # Inference
    "run_inference",
]

__version__ = "0.1.0"

