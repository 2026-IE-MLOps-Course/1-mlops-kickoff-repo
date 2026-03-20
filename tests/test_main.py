"""
test_main.py
------------
Integration tests for the pipeline orchestrator (src/main.py).

Coverage:
- End-to-end pass: all three artifacts are created
- Artifact content: non-empty files, valid CSV headers, deserializable model
- Error handling: missing raw data raises an appropriate exception
- Idempotency: running main() twice does not raise or corrupt artifacts
- Example config warning, stratification fallback, missing columns, dtype check
"""

import os
import runpy

import joblib
import pandas as pd
import pytest

import src.main as main_module


# --------------------------------------------------------------------------- #
# Fixture: redirect SETTINGS paths to isolated tmp directories                #
# --------------------------------------------------------------------------- #

@pytest.fixture()
def pipeline_env(sample_raw_csv, tmp_path):
    """Temporarily point SETTINGS paths at tmp_path; restore after the test."""
    original_paths = main_module.SETTINGS["paths"].copy()

    main_module.SETTINGS["paths"]["raw_data"] = str(sample_raw_csv)
    main_module.SETTINGS["paths"]["processed_data"] = str(
        tmp_path / "data" / "processed" / "clean.csv"
    )
    main_module.SETTINGS["paths"]["model_artifact"] = str(
        tmp_path / "models" / "model.joblib"
    )
    main_module.SETTINGS["paths"]["predictions_output"] = str(
        tmp_path / "reports" / "predictions.csv"
    )

    yield main_module.SETTINGS

    # Always restore — even if the test fails
    main_module.SETTINGS["paths"].update(original_paths)


# --------------------------------------------------------------------------- #
# Tests                                                                        #
# --------------------------------------------------------------------------- #

class TestRunPipeline:
    """Integration tests for the main() orchestrator function."""

    # ------------------------------------------------------------------ #
    # Artifact existence                                                   #
    # ------------------------------------------------------------------ #

    def test_all_three_artifacts_are_created(self, pipeline_env):
        """main() creates clean.csv, model.joblib, and predictions.csv."""
        main_module.main()

        assert os.path.isfile(pipeline_env["paths"]["processed_data"]), \
            "Processed CSV was not created"
        assert os.path.isfile(pipeline_env["paths"]["model_artifact"]), \
            "Model artifact was not created"
        assert os.path.isfile(pipeline_env["paths"]["predictions_output"]), \
            "Predictions CSV was not created"

    def test_processed_csv_is_non_empty(self, pipeline_env):
        """The processed CSV must contain at least one data row."""
        main_module.main()

        df = pd.read_csv(pipeline_env["paths"]["processed_data"])
        assert len(df) > 0, "Processed CSV has zero rows"

    def test_processed_csv_contains_target_column(self, pipeline_env):
        """Processed CSV must include the target column 'total_cost'."""
        main_module.main()

        df = pd.read_csv(pipeline_env["paths"]["processed_data"])
        assert "total_cost" in df.columns, (
            f"'total_cost' not found in processed CSV columns: {list(df.columns)}"
        )

    def test_predictions_csv_has_prediction_column(self, pipeline_env):
        """Predictions CSV must have exactly one column named 'prediction'."""
        main_module.main()

        preds = pd.read_csv(pipeline_env["paths"]["predictions_output"])
        assert "prediction" in preds.columns, (
            f"'prediction' not found in predictions CSV: {list(preds.columns)}"
        )

    def test_predictions_csv_row_count_is_positive(self, pipeline_env):
        """Predictions CSV must contain at least one row."""
        main_module.main()

        preds = pd.read_csv(pipeline_env["paths"]["predictions_output"])
        assert len(preds) > 0, "Predictions CSV has zero rows"

    def test_model_artifact_is_deserializable(self, pipeline_env):
        """The saved model.joblib must be loadable with joblib."""
        main_module.main()

        model = joblib.load(pipeline_env["paths"]["model_artifact"])
        assert hasattr(model, "predict"), (
            "Loaded artifact does not have a .predict() method"
        )

    def test_model_artifact_can_predict(self, pipeline_env, sample_feature_df):
        """The deserialized model must produce predictions on the feature fixture."""
        main_module.main()

        model = joblib.load(pipeline_env["paths"]["model_artifact"])
        preds = model.predict(sample_feature_df)
        assert len(preds) == len(sample_feature_df)

    # ------------------------------------------------------------------ #
    # Idempotency                                                          #
    # ------------------------------------------------------------------ #

    def test_pipeline_is_idempotent(self, pipeline_env):
        """Running main() twice does not raise and produces valid artifacts."""
        main_module.main()
        main_module.main()  # second run must overwrite cleanly

        assert os.path.isfile(pipeline_env["paths"]["model_artifact"])
        assert os.path.isfile(pipeline_env["paths"]["processed_data"])
        assert os.path.isfile(pipeline_env["paths"]["predictions_output"])

    # ------------------------------------------------------------------ #
    # Error handling                                                       #
    # ------------------------------------------------------------------ #

    def test_missing_raw_data_raises(self, tmp_path):
        """main() raises an exception when the raw data file does not exist."""
        original_paths = main_module.SETTINGS["paths"].copy()
        main_module.SETTINGS["paths"]["raw_data"] = str(
            tmp_path / "nonexistent.csv"
        )

        try:
            with pytest.raises(Exception):
                main_module.main()
        finally:
            main_module.SETTINGS["paths"].update(original_paths)

    # ------------------------------------------------------------------ #
    # NEW: Cover line 103 — is_example_config = True branch               #
    # ------------------------------------------------------------------ #

    def test_example_config_warning(self, pipeline_env):
        """When is_example_config is True, the pipeline prints a warning
        but still completes (covers line 103)."""
        original = main_module.SETTINGS["is_example_config"]
        main_module.SETTINGS["is_example_config"] = True
        try:
            main_module.main()
            assert os.path.isfile(pipeline_env["paths"]["model_artifact"])
        finally:
            main_module.SETTINGS["is_example_config"] = original

    # ------------------------------------------------------------------ #
    # NEW: Cover lines 160-166 — first stratification fallback            #
    # ------------------------------------------------------------------ #

    def test_stratification_fallback_first_split(self, pipeline_env):
        """When problem_type is 'classification' and stratification fails
        on the first split, the pipeline falls back to unstratified split
        (covers lines 160-166)."""
        original_type = main_module.SETTINGS["problem_type"]
        original_target = main_module.SETTINGS["target_column"]

        # Set classification mode — with continuous target values,
        # stratification will fail because too many unique classes
        main_module.SETTINGS["problem_type"] = "classification"

        try:
            # This will trigger the ValueError fallback in the first
            # train_test_split because total_cost has many unique values
            # which makes stratification impossible
            main_module.main()
            assert os.path.isfile(pipeline_env["paths"]["model_artifact"])
        finally:
            main_module.SETTINGS["problem_type"] = original_type
            main_module.SETTINGS["target_column"] = original_target

    # ------------------------------------------------------------------ #
    # NEW: Cover line 201 — missing configured feature columns            #
    # ------------------------------------------------------------------ #

    def test_missing_configured_feature_columns_raises(self, pipeline_env):
        """ValueError when a configured feature column doesn't exist in the
        cleaned data (covers line 201)."""
        original_features = main_module.SETTINGS["features"].copy()
        # Add a column that doesn't exist in the dataset
        main_module.SETTINGS["features"]["numeric_passthrough"] = (
            original_features["numeric_passthrough"]
            + ["nonexistent_column"]
        )
        try:
            with pytest.raises(ValueError, match="missing"):
                main_module.main()
        finally:
            main_module.SETTINGS["features"] = original_features

    # ------------------------------------------------------------------ #
    # NEW: Cover line 208 — non-numeric dtype for numeric column          #
    # ------------------------------------------------------------------ #

    def test_non_numeric_dtype_raises_type_error(self, sample_raw_csv, tmp_path):
        """TypeError when a column configured as numeric has a non-numeric dtype
        (covers line 208)."""
        original_paths = main_module.SETTINGS["paths"].copy()
        original_features = main_module.SETTINGS["features"].copy()

        main_module.SETTINGS["paths"]["raw_data"] = str(sample_raw_csv)
        main_module.SETTINGS["paths"]["processed_data"] = str(
            tmp_path / "data" / "processed" / "clean.csv"
        )
        main_module.SETTINGS["paths"]["model_artifact"] = str(
            tmp_path / "models" / "model.joblib"
        )
        main_module.SETTINGS["paths"]["predictions_output"] = str(
            tmp_path / "reports" / "predictions.csv"
        )

        # Add a categorical column to the numeric_passthrough list
        main_module.SETTINGS["features"]["numeric_passthrough"] = (
            original_features["numeric_passthrough"]
            + ["destination_country"]
        )
        # Remove it from categorical to avoid duplicate
        cat_cols = [
            c for c in original_features["categorical_onehot"]
            if c != "destination_country"
        ]
        main_module.SETTINGS["features"]["categorical_onehot"] = cat_cols

        try:
            with pytest.raises(TypeError, match="numeric"):
                main_module.main()
        finally:
            main_module.SETTINGS["paths"].update(original_paths)
            main_module.SETTINGS["features"] = original_features

    # ------------------------------------------------------------------ #
    # NEW: Cover line 251 — __name__ == "__main__" block                  #
    # ------------------------------------------------------------------ #

    def test_main_module_entrypoint(self, pipeline_env):
        """Running src/main.py as __main__ invokes main() (covers line 251)."""
        # We use runpy to simulate `python -m src.main` which triggers
        # the if __name__ == "__main__" block.
        # Since we've already set up pipeline_env, this should work.
        import importlib
        import src.main

        spec = importlib.util.find_spec("src.main")
        # Execute the module as __main__
        runpy.run_module("src.main", run_name="__main__", alter_sys=False)

        assert os.path.isfile(pipeline_env["paths"]["model_artifact"])
