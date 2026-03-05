"""
Module: Main Pipeline
---------------------
Role: Orchestrate the entire flow (Load -> Validate -> Clean -> Train -> Evaluate -> Infer).
Usage: python src/main.py
"""

from src.load_data import load_data
from src.validate import validate_dataframe
from src.infer import run_inference

# TODO: import when available
# from src.clean_data import clean_data
# from src.features import build_features
# from src.utils import ...
# from src.train import train_model
# from src.evaluate import evaluate_model

REQUIRED_COLUMNS = [
    "AccountWeeks", "DataUsage", "CustServCalls",
    "DayMins", "DayCalls", "MonthlyCharge",
    "OverageFee", "RoamMins", "Churn",
    "ContractRenewal", "DataPlan",
]

if __name__ == "__main__":
    # Step 1: Load data
    df = load_data()

    # Step 2: Validate
    validate_dataframe(df, REQUIRED_COLUMNS)

    # Step 3: Clean (TODO: uncomment when clean_data.py is implemented)
    # df = clean_data(df)

    # Step 4: Feature engineering (TODO: uncomment when features.py is implemented)
    # df = build_features(df)

    # Step 5: Train (TODO: uncomment when train.py is implemented)
    # model = train_model(df)

    # Step 6: Evaluate (TODO: uncomment when evaluate.py is implemented)
    # evaluate_model(model, df)

    # Step 7: Infer (TODO: uncomment when model is ready)
    # predictions = run_inference(model, df.drop(columns=["Churn"]))

    print("Pipeline running: Load and Validate steps complete.")