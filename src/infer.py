"""
Educational Goal:
- Why this module exists in an MLOps system: Inference is decoupled from training
  so the same module can be reused in production APIs, batch scoring jobs, or
  interactive notebooks. It enforces a strict output schema.
- Responsibility (separation of concerns): Accept a fitted model and new data,
  return predictions. No training, no evaluation, no file I/O.
- Pipeline contract (inputs and outputs):
  Input  — model (fitted Pipeline), X_infer (pd.DataFrame).
  Output — pd.DataFrame with a single column "prediction", preserving the
           original input index.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


def run_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
    - model: Fitted sklearn Pipeline (must have .predict()).
    - X_infer (pd.DataFrame): Feature DataFrame for new/unseen data.
    Outputs:
    - pd.DataFrame: A DataFrame with a SINGLE column named "prediction"
      that preserves the input index.
    Why this contract matters for reliable ML delivery:
    - A strict output schema (single column, preserved index) means
      downstream systems can reliably join predictions back to source
      records by index. Requiring a DataFrame (not a numpy array) prevents
      Scikit-Learn from dropping feature names.
    """
    print(f"[infer] Running inference on {len(X_infer)} samples …")  # TODO: replace with logging later

    # Enforce strict input types
    if not isinstance(X_infer, pd.DataFrame):
        raise TypeError(
            "Inference input must be a pandas DataFrame, "
            f"got {type(X_infer).__name__}."
        )
    if not hasattr(model, "predict"):
        raise TypeError(
            "The provided artifact does not have a .predict() method."
        )

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Some use cases need post-processing (e.g., rounding prices,
    #       clipping probabilities, adding probability columns for risk scoring)
    # Examples:
    # 1. predictions = model.predict(X_infer).clip(min=0)  # no negative costs
    # 2. proba = model.predict_proba(X_infer)[:, 1]  # for classification
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Student implementation for VoyageIQ travel dataset:
    predictions = model.predict(X_infer)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    result = pd.DataFrame(
        {"prediction": predictions},
        index=X_infer.index,
    )

    print(f"[infer] Inference complete — {len(result)} predictions generated.")  # TODO: replace with logging later
    return result
