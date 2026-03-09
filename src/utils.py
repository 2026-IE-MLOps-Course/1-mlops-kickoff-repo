"""
Educational Goal:
- Why this module exists in an MLOps system: Centralises all file I/O operations
  (reading CSV, saving CSV, persisting models) so that every other module calls the
  same functions. This eliminates duplicated or inconsistent data loading logic.
- Responsibility (separation of concerns): Pure I/O plumbing — read files, write
  files, serialise/deserialise model artifacts. Contains zero business logic.
- Pipeline contract (inputs and outputs): Accepts pathlib.Path objects and returns
  pandas DataFrames (for data) or sklearn-compatible objects (for models).

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path

import joblib
import pandas as pd


def load_csv(filepath: Path) -> pd.DataFrame:
    """
    Inputs:
    - filepath (Path): Path to the CSV file to read.
    Outputs:
    - pd.DataFrame: The loaded data.
    Why this contract matters for reliable ML delivery:
    - A single CSV-loading function prevents inconsistent parsing across modules.
      If encoding or separator rules need to change, we update one place.
    """
    print(f"[utils] Loading CSV from: {filepath}")  # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Different datasets may need custom encoding, separators, or dtype hints
    # Examples:
    # 1. pd.read_csv(filepath, encoding="latin-1", sep=";")
    # 2. pd.read_csv(filepath, dtype={"col_a": str, "col_b": float})
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Student implementation for VoyageIQ travel dataset:
    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding="latin-1")
    except pd.errors.ParserError as exc:
        raise pd.errors.ParserError(
            f"Failed to parse CSV at {filepath}. "
            "Check the file format and separator."
        ) from exc
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    print(f"[utils] CSV loaded — shape: {df.shape}")  # TODO: replace with logging later
    return df


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    """
    Inputs:
    - df (pd.DataFrame): The DataFrame to persist.
    - filepath (Path): Destination path for the CSV file.
    Outputs:
    - None (side effect: writes a CSV file to disk).
    Why this contract matters for reliable ML delivery:
    - A centralised save function guarantees that parent directories are
      always created and that index=False is the safe default, preventing
      mysterious unnamed index columns from polluting downstream reads.
    """
    print(f"[utils] Saving CSV to: {filepath}")  # TODO: replace with logging later

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Some projects may need to save with a specific encoding or separator
    # Examples:
    # 1. df.to_csv(filepath, index=False, encoding="utf-8-sig")
    # 2. df.to_csv(filepath, index=True)  # if index carries patient IDs
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Student implementation:
    df.to_csv(filepath, index=False)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    print(f"[utils] CSV saved successfully.")  # TODO: replace with logging later


def save_model(model, filepath: Path) -> None:
    """
    Inputs:
    - model: A fitted sklearn Pipeline or estimator.
    - filepath (Path): Destination path for the serialised model.
    Outputs:
    - None (side effect: writes a .joblib file to disk).
    Why this contract matters for reliable ML delivery:
    - Persisting the fitted pipeline as a single artifact bundles
      preprocessing logic with model weights. This prevents training-serving
      skew where production transforms differ from what the model learned.
    """
    print(f"[utils] Saving model to: {filepath}")  # TODO: replace with logging later

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Some teams prefer pickle over joblib, or add compression
    # Examples:
    # 1. joblib.dump(model, filepath, compress=3)
    # 2. pickle.dump(model, open(filepath, "wb"))
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Student implementation:
    joblib.dump(model, filepath)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    print(f"[utils] Model saved successfully.")  # TODO: replace with logging later


def load_model(filepath: Path):
    """
    Inputs:
    - filepath (Path): Path to the serialised .joblib model artifact.
    Outputs:
    - A fitted sklearn Pipeline or estimator.
    Why this contract matters for reliable ML delivery:
    - Loading a single artifact that bundles preprocessing + model guarantees
      that inference uses identical transformations as training.
    """
    print(f"[utils] Loading model from: {filepath}")  # TODO: replace with logging later

    filepath = Path(filepath)

    if not filepath.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at {filepath}. Run training first."
        )

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Different serialisation formats may be used (pickle, ONNX, etc.)
    # Examples:
    # 1. model = pickle.load(open(filepath, "rb"))
    # 2. model = onnxruntime.InferenceSession(str(filepath))
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Student implementation:
    model = joblib.load(filepath)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    print(f"[utils] Model loaded successfully.")  # TODO: replace with logging later
    return model
