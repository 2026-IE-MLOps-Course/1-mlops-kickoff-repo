"""
Educational Goal:
- Why this module exists in an MLOps system: Centralize simple I/O primitives (CSV + model artifacts) so pipeline steps stay focused on ML logic.
- Responsibility (separation of concerns): This module only handles reading/writing. Business logic lives in load/clean/features/train/evaluate/infer.
- Pipeline contract (inputs and outputs): Reliable persistence and retrieval of tabular data and trained models via stable functions.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path

import joblib
import pandas as pd


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Inputs:
    - filepath: Path to a CSV file.
    Outputs:
    - df: Loaded pandas DataFrame.
    Why this contract matters for reliable ML delivery:
    - Centralizing CSV reads makes pipelines more reproducible and easier to debug when file formats change.
    """
    print(f"[utils.load_csv] Loading CSV from: {filepath}")  # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Adjust read_csv parameters (dtype, parse_dates, encoding) if your dataset needs it
    # Why: CSV dialect and schema assumptions vary by source systems
    # Examples:
    # 1. pd.read_csv(filepath, encoding="utf-8")
    # 2. pd.read_csv(filepath, parse_dates=["event_time"])
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return pd.read_csv(filepath)


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Inputs:
    - df: DataFrame to save.
    - filepath: Path where the CSV should be written.
    Outputs:
    - None (writes file as a side-effect).
    Why this contract matters for reliable ML delivery:
    - Standardizing output locations makes downstream automation and reviews predictable (CI, reports, deployments).
    """
    print(f"[utils.save_csv] Saving CSV to: {filepath}")  # TODO: replace with logging later
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Decide whether to keep index, set float_format, or control NA serialization
    # Why: Output formatting affects reproducibility and downstream ingestion
    # Examples:
    # 1. df.to_csv(filepath, index=False, float_format="%.6f")
    # 2. df.to_csv(filepath, index=True)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    df.to_csv(filepath, index=False)


def save_model(model, filepath: Path) -> None:
    """
    Inputs:
    - model: Any fitted scikit-learn compatible object (here: Pipeline).
    - filepath: Path to write the serialized model artifact.
    Outputs:
    - None (writes file as a side-effect).
    Why this contract matters for reliable ML delivery:
    - Persisting the exact trained Pipeline enables consistent inference and supports promotion across environments.
    """
    print(f"[utils.save_model] Saving model to: {filepath}")  # TODO: replace with logging later
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Tune joblib parameters (compress) if model artifacts get large
    # Why: Artifact size impacts storage costs and deployment latency
    # Examples:
    # 1. joblib.dump(model, filepath, compress=3)
    # 2. joblib.dump(model, filepath)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    joblib.dump(model, filepath)


def load_model(filepath: Path):
    """
    Inputs:
    - filepath: Path to a serialized model artifact.
    Outputs:
    - model: Loaded model object.
    Why this contract matters for reliable ML delivery:
    - Inference systems must load the same artifact format across training and serving to prevent drift in behavior.
    """
    print(f"[utils.load_model] Loading model from: {filepath}")  # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add compatibility handling if you change serialization strategy later
    # Why: Teams sometimes migrate from joblib to other formats (e.g., ONNX) based on serving constraints
    # Examples:
    # 1. joblib.load(filepath)
    # 2. (future) load ONNX runtime session
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return joblib.load(filepath)