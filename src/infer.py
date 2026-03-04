"""
Module: infer.py
----------------
Role: Generate predictions and risk probabilities on new data using a fitted model.
Input: Fitted scikit-learn Pipeline + Processed features DataFrame.
Output: pandas.DataFrame with class predictions and churn probabilities.
"""

import logging
import pandas as pd
import numpy as np

# Configuration of logging for production traceability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def run_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Generates predictions and risk scores for unseen data.

    Why this design? 
    1. Index Preservation: We maintain the original DataFrame index so 
       predictions can be joined back to customer IDs in the database.
    2. Probability Output: Providing a 'Churn_Probability' allows business 
       teams to rank customers by risk rather than using a rigid 0.5 cutoff.

    Args:
        model: The fitted scikit-learn Pipeline (preprocessor + estimator).
        X_infer (pd.DataFrame): Preprocessed features to predict on.

    Returns:
        pd.DataFrame: Contains "prediction" (0/1) and "churn_probability" (0.0-1.0).
    """
    logging.info(f"Starting inference on {len(X_infer)} records...")

    try:
        # 1. Generate class predictions (Yes/No)
        # Uses the default scikit-learn threshold of 0.5
        predictions = model.predict(X_infer)

        # 2. Generate churn probabilities (Business Risk Score)
        # predict_proba returns [prob_no, prob_yes]. We take index 1 for 'Yes'.
        probabilities = model.predict_proba(X_infer)[:, 1]

        logging.info("Inference completed successfully.")

        # 3. Consolidate results into a structured DataFrame
        result_df = pd.DataFrame({
            "prediction": predictions,
            "churn_probability": np.round(probabilities, 4)
        }, index=X_infer.index)

        return result_df

    except AttributeError as e:
        logging.error(
            "The provided model does not support probability predictions.")
        raise e
    except Exception as e:
        logging.error(f"An unexpected error occurred during inference: {e}")
        raise e


def interpret_results(result_df: pd.DataFrame, threshold: float = 0.7):
    """
    Higher-level business logic to tag 'High Risk' customers.

    Why: MLOps is about business value. This helper allows teams to 
    set a custom risk threshold (e.g., > 70%) for marketing campaigns.
    """
    result_df['risk_level'] = [
        "High Risk" if p >= threshold else "Standard"
        for p in result_df['churn_probability']
    ]
    return result_df
