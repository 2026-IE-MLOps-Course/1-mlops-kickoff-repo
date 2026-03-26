import pandas as pd
from unittest.mock import patch
from src.main import main


def test_main_pipeline_runs(tmp_path):

    # Fake dataset
    df = pd.DataFrame({
        "customerID": ["A", "B", "C"],
        "tenure": [1, 2, 3],
        "MonthlyCharges": [10, 20, 30],
        "TotalCharges": [10, 40, 90],
        "SeniorCitizen": [0, 1, 0],
        "Churn": ["No", "Yes", "No"]
    })

    predictions = pd.DataFrame({
        "prediction": [0, 1, 0],
        "churn_probability": [0.1, 0.8, 0.2]
    })

    with patch("src.main.load_raw_data", return_value=df), \
         patch("src.main.clean_data", return_value=df), \
         patch("src.main.validate_dataframe", return_value=True), \
         patch("src.main.engineer_features", return_value=df), \
         patch("src.main.get_preprocessor"), \
         patch("src.main.train_model"), \
         patch("src.main.evaluate_model", return_value=0.85), \
         patch("src.main.run_inference", return_value=predictions), \
         patch("src.main._wandb_is_enabled", return_value=False):

        # Run pipeline
        main()

        # If no exception → test passes
        assert True
