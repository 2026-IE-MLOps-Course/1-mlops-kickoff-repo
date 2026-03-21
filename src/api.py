# src/api.py
"""
FastAPI prediction service.

Critical Principle: api.py contains ZERO new Machine Learning logic.
It is a thin wrapper that calls existing src modules:
  clean_dataframe() → validate_dataframe() → run_inference()

Model loading at startup:
  - MODEL_SOURCE=wandb  → downloads the promoted W&B artifact (alias=prod)
  - MODEL_SOURCE=local   → loads models/model.joblib from disk
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# ──────────────────────────────
# Pydantic schemas (data contract)
# ──────────────────────────────


class TravelRecord(BaseModel):
    """Schema for a single travel prediction request."""
    model_config = ConfigDict(extra="forbid")

    destination: str
    start_date: str
    end_date: str
    duration_days: float
    traveler_age: float
    traveler_gender: str
    traveler_nationality: str
    accommodation_type: str
    accommodation_cost: float
    transportation_type: str
    transportation_cost: float


class PredictRequest(BaseModel):
    """Wrapper for batch prediction requests."""
    records: List[TravelRecord]


class PredictResponse(BaseModel):
    """Response with predictions."""
    predictions: List[float]


# ──────────────────────────────
# Config helpers
# ──────────────────────────────

def _load_serving_config() -> dict:
    """Load config.yaml for serving."""
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _feature_columns(cfg: dict) -> List[str]:
    """Return the feature columns the model expects at inference time."""
    feat = cfg.get("features", {})
    cols = (
        feat.get("numeric_features", [])
        + feat.get("date_derived_features", [])
        + feat.get("categorical_features", [])
    )
    return cols


# ──────────────────────────────
# Model loading
# ──────────────────────────────

def _load_model_from_wandb(cfg: dict):
    """Download the promoted W&B model artifact and load it."""
    import wandb
    import joblib

    wandb_cfg = cfg.get("wandb", {})
    project = wandb_cfg.get("project", "")
    artifact_name = wandb_cfg.get("model_artifact_name", "voyageiq_model")
    alias = os.getenv("WANDB_MODEL_ALIAS", "prod")
    entity = os.getenv("WANDB_ENTITY", "")

    # Construct the artifact path
    if entity:
        artifact_path = f"{entity}/{project}/{artifact_name}:{alias}"
    else:
        artifact_path = f"{project}/{artifact_name}:{alias}"

    logger.info("Downloading W&B artifact: %s", artifact_path)

    run = wandb.init(project=project, job_type="inference", settings=wandb.Settings(silent=True))
    artifact = run.use_artifact(artifact_path, type="model")
    artifact_dir = artifact.download()
    wandb.finish()

    # Find the .joblib file in the downloaded directory
    joblib_files = list(Path(artifact_dir).glob("*.joblib"))
    if not joblib_files:
        raise FileNotFoundError(f"No .joblib file found in artifact directory: {artifact_dir}")

    model = joblib.load(joblib_files[0])
    logger.info("Model loaded from W&B artifact: %s", artifact_path)
    return model


def _load_model_local(cfg: dict):
    """Load model from local disk."""
    from src.utils import load_model

    project_root = Path(__file__).resolve().parents[1]
    model_path = project_root / cfg["paths"]["model_artifact"]
    return load_model(model_path)


# ──────────────────────────────
# Lifespan (startup / shutdown)
# ──────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load config and model once at startup."""
    cfg = _load_serving_config()
    app.state.config = cfg
    app.state.feature_columns = _feature_columns(cfg)

    model_source = os.getenv("MODEL_SOURCE", "local").strip().lower()
    logger.info("MODEL_SOURCE=%s", model_source)

    if model_source == "wandb":
        app.state.model = _load_model_from_wandb(cfg)
    else:
        app.state.model = _load_model_local(cfg)

    logger.info("Model loaded and ready for serving.")
    yield


# ──────────────────────────────
# App
# ──────────────────────────────

app = FastAPI(
    title="VoyageIQ Trip Cost Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    """Welcome endpoint."""
    return {"message": "Welcome to VoyageIQ Trip Cost Prediction API"}


@app.get("/health")
def health():
    """Health check — confirms server is alive and model is loaded."""
    model_loaded = hasattr(app.state, "model") and app.state.model is not None
    return {"status": "ok", "model_loaded": model_loaded}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict total trip cost for one or more travel records.

    Pipeline: JSON → DataFrame → clean_dataframe() → validate_dataframe() → run_inference()
    Zero new ML logic — only calls existing src modules.
    """
    from src.clean_data import clean_dataframe
    from src.validate import validate_dataframe
    from src.infer import run_inference

    try:
        # 1. Convert Pydantic records to DataFrame
        records_dicts = [r.model_dump() for r in request.records]
        df_input = pd.DataFrame(records_dicts)

        # 2. Clean (inference mode — no target column)
        df_clean = clean_dataframe(df_input, target_column=None)

        # 3. Validate feature columns exist
        feature_cols = app.state.feature_columns
        required_cols = [c for c in feature_cols if c in df_clean.columns]
        validate_dataframe(df_clean, required_cols)

        # 4. Select only feature columns for prediction
        X_infer = df_clean[required_cols]

        # 5. Run inference
        preds_df = run_inference(app.state.model, X_infer)

        predictions = preds_df["prediction"].tolist()
        logger.info("Prediction served for %d records", len(predictions))

        return PredictResponse(predictions=predictions)

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
