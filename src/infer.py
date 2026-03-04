"""
Module: Inference
-----------------
Role: Make predictions on new, unseen data.
Input: Trained Model + New Data.
Output: Predictions (Array or DataFrame).
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Defines how the model generates predictions on new, unseen data in production.
- Responsibility (separation of concerns): Loading the model and formatting the output predictions.
- Pipeline contract (inputs and outputs): Takes a fitted model and new data, returns a DataFrame containing only predictions.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""


import pandas as pd
def run_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
    - model: The fitted scikit-learn Pipeline.
    - X_infer: DataFrame containing new features to predict on.
    Outputs:
    - pd.DataFrame with a single column "prediction" matching the input index.
    Why this contract matters for reliable ML delivery:
    - Maintaining the original index ensures predictions can be accurately joined back to business records (like user IDs).
    """
    print("Executing run_inference")  # TODO: replace with logging later

    predictions = model.predict(X_infer)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: You might need to return probabilities instead of labels, or map predictions to business logic (e.g., 1="Approve", 0="Deny").
    # Examples:
    # 1. probs = model.predict_proba(X_infer)[:, 1]
    # 2. preds = ["High Risk" if p == 1 else "Low Risk" for p in predictions]
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    result_df = pd.DataFrame({"prediction": predictions}, index=X_infer.index)
    return result_df
