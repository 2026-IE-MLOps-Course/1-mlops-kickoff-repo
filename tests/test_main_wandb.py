import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import os
from src.main import run_pipeline

@pytest.fixture
def mock_config():
    return {
        "wandb": {
            "project": "test-project",
            "job_type": "test-job",
            "model_artifact_name": "test-model",
            "model_alias": "prod"
        },
        "train": {
            "test_size": 0.2,
            "seed": 42
        },
        "data": {
            "raw": "data/raw.csv",
            "processed": "data/processed.csv"
        }
    }

@patch("src.main.wandb")
@patch("src.main.load_config")
@patch("src.main.load_data")
@patch("src.main.clean_dataframe")
@patch("src.main.validate_dataframe")
@patch("src.main.train_model")
@patch("src.main.evaluate_model")
@patch("src.main.os.makedirs")
@patch("src.main.os.path.exists")
def test_run_pipeline_wandb_logic(
    mock_exists,
    mock_makedirs,
    mock_evaluate,
    mock_train,
    mock_validate,
    mock_clean,
    mock_load,
    mock_load_config,
    mock_wandb,
    mock_config
):
    """
    Test that run_pipeline calls W&B functions with the correct arguments.
    """
    # Setup mocks
    mock_load_config.return_value = mock_config
    
    mock_df = MagicMock(spec=pd.DataFrame)
    mock_df.shape = (100, 10)
    mock_load.return_value = mock_df
    mock_clean.return_value = mock_df
    
    # Mock model and metrics
    mock_pipeline = MagicMock()
    mock_classifier = MagicMock()
    mock_pipeline.named_steps = {"classifier": mock_classifier}
    mock_train.return_value = (mock_pipeline, {"train_acc": 0.9}, MagicMock(), MagicMock())
    mock_evaluate.return_value = {"test_acc": 0.8}
    
    # Mock file existence for logs
    mock_exists.return_value = True
    
    # Execute
    run_pipeline()
    
    # 1. Verify wandb.init
    mock_wandb.init.assert_called_once_with(
        project="test-project",
        job_type="test-job",
        config={
            "test_size": 0.2,
            "seed": 42,
            "raw_data_path": "data/raw.csv",
        }
    )
    
    # 2. Verify wandb.log for data stats
    mock_wandb.log.assert_any_call({
        "data/raw_rows": 100,
        "data/raw_columns": 10,
    })
    
    # 3. Verify wandb.config.update for model type
    mock_wandb.config.update.assert_called()
    
    # 4. Verify wandb.log for all metrics
    # It merges metrics and eval_metrics
    expected_metrics = {"train_acc": 0.9, "test_acc": 0.8}
    mock_wandb.log.assert_any_call(expected_metrics)
    
    # 5. Verify wandb.log_artifact for model
    # We check that at least one Artifact was created and logged
    assert mock_wandb.Artifact.call_count >= 1
    mock_wandb.log_artifact.assert_called()
    
    # 6. Verify wandb.finish
    mock_wandb.finish.assert_called_once()

@patch("src.main.wandb")
@patch("src.main.load_config")
@patch("src.main.load_data")
def test_run_pipeline_failure_cleanup(mock_load, mock_load_config, mock_wandb, mock_config):
    """
    Test that wandb.finish is called even if the pipeline fails.
    """
    mock_load_config.return_value = mock_config
    mock_load.side_effect = Exception("Data loading failed")
    
    with pytest.raises(Exception):
        run_pipeline()
    
    mock_wandb.finish.assert_called_once()
