# VoyageIQ – Total Trip Cost Forecasting Pipeline

> **MLOps Engineering · Final Group Assignment**
> MsC in Business Analytics and Data Science · IE University · Madrid, 2026

---

## Business Case

| Section | Details |
|---|---|
| **Client** | **VoyageIQ** — Online Travel Agency / Corporate Travel Desk |
| **Business Unit** | Pricing & Revenue Optimisation |
| **Client Maturity** | *Data:* Structured booking records in CSV. *Tools:* Basic Excel reporting. *Processes:* Manual quote generation. *People:* Small analytics team. *Strategy:* Moving toward data-driven pricing. |
| **Goal of Project** | Predict total trip cost (accommodation + transportation) before a booking is confirmed, reducing quote turnaround time by ≥ 60 % and quote error rate to < 10 % MAPE. |
| **Problem Statement** | VoyageIQ agents currently build quotes manually by researching destination prices, leading to slow response times (~24 h per quote) and inconsistent pricing. The baseline error rate on manual quotes is approximately 20–25 %. Measured by: Mean Absolute Percentage Error (MAPE) of predicted vs. actual trip cost. |
| **Solution Description** | An end-to-end ML regression pipeline that ingests traveller and trip attributes (destination, duration, accommodation type, transport mode, demographics) and outputs a predicted total trip cost. The pipeline is fully modular, config-driven, and production-ready, served via a FastAPI REST API deployed on Render. |
| **Solution Scalability** | The architecture generalises to any tabular regression/classification task. Additional data sources (hotel APIs, flight aggregators) can be plugged into `load_data.py`. New models (XGBoost, neural nets) are added by extending the model factory in `train.py`. The same pipeline can serve B2B corporate bulk-quoting and B2C instant pricing. |
| **Client Benefit** | *Short-term:* Instant cost estimates → faster sales cycle. *Long-term:* Dynamic pricing optimisation, personalised packages. *Competitiveness:* Data-driven quoting vs. competitors' manual processes. Measured by: reduction in MAPE, quote turnaround time, and agent productivity. |
| **Cost Estimation ($000)** | Talent: AI specialist ($15 k), Product Mgr. ($10 k), ML/SW Engineer ($20 k), Data Engineer ($15 k), SME ($5 k). Client covers: data infrastructure, cloud compute, licences. Timeline: 12+ weeks. |
| **Risks & Challenges** | Data quality (small dataset, potential PII leakage), model generalisation to unseen destinations, currency fluctuations, seasonal price swings. *Mitigations:* robust validation gate, .gitignore for PII, periodic retraining schedule, config-driven feature selection. |

---

## Repository Structure

```
.
├── README.md                 # This file
├── config.yaml               # Central configuration (all non-secret runtime settings)
├── environment.yml           # Conda dependency management
├── conda-lock.yml            # Locked dependencies for reproducible Docker builds
├── .env                      # Secrets (WANDB_API_KEY, etc.) — git-ignored, never committed
├── .env.example              # Template for .env
├── .gitignore                # Git exclusion rules
├── .dockerignore             # Docker build exclusions (lean image)
├── Dockerfile                # Container definition for serving
├── pytest.ini                # Pytest configuration
├── .github/
│   └── workflows/
│       ├── ci.yml            # CI — runs tests + validates Docker build on PRs to main
│       └── deploy.yml        # CD — triggers Render deploy on GitHub Release
├── data/
│   ├── raw/
│   │   └── travel_raw.csv    # Original dataset (git-ignored, tracked via DVC)
│   ├── processed/
│   │   └── clean.csv         # Cleaned output (git-ignored)
│   └── inference/
│       └── .gitkeep
├── models/
│   └── model.joblib          # Trained pipeline artifact (git-ignored; production uses W&B registry)
├── reports/
│   └── predictions.csv       # Inference output (git-ignored)
├── logs/
│   └── pipeline.log          # Dual-output log file (git-ignored)
├── notebooks/
│   ├── 01_voyageiq_analysis_Legacy.ipynb   # Original monolithic notebook
│   └── 01_voyageiq_analysis_vExp.ipynb     # Modular notebook (imports from src/)
├── src/
│   ├── __init__.py           # Marks src as a Python package
│   ├── logger.py             # Dual-output logging (console + file); zero print() policy
│   ├── utils.py              # I/O plumbing: load_csv, save_csv, save_model, load_model
│   ├── load_data.py          # Data ingestion (load_raw_data)
│   ├── clean_data.py         # Data cleaning / stabilisation (clean_dataframe)
│   ├── validate.py           # Schema & range validation (validate_dataframe)
│   ├── features.py           # Feature preprocessor recipe (get_feature_preprocessor)
│   ├── train.py              # Model training (train_model)
│   ├── evaluate.py           # Metric computation (evaluate_model → dict)
│   ├── infer.py              # Inference / prediction (run_inference → DataFrame)
│   ├── main.py               # Pipeline orchestrator (config-driven, W&B integrated)
│   └── api.py                # FastAPI serving layer (/health, /predict)
└── tests/
    ├── __init__.py
    ├── conftest.py            # Shared fixtures
    ├── test_utils.py
    ├── test_load_data.py
    ├── test_clean_data.py
    ├── test_validate.py
    ├── test_features.py
    ├── test_train.py
    ├── test_evaluate.py
    ├── test_infer.py
    ├── test_main.py
    ├── test_api.py
    └── test_logger.py
```

---

## How to Run

### 1. Set up the environment

```bash
conda env create -f environment.yml
conda activate mlops-student-env
```

For reproducible builds (Docker and CI use this):

```bash
conda install -n base -c conda-forge conda-lock -y
conda-lock install --name mlops-student-env conda-lock.yml
```

### 2. Configure secrets

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required variables:

```
WANDB_API_KEY=<your-key>
WANDB_ENTITY=<your-entity>
WANDB_MODE=online
MODEL_SOURCE=local        # "local" for training, "wandb" for serving from registry
WANDB_MODEL_ALIAS=prod    # used when MODEL_SOURCE=wandb
```

### 3. Place the dataset

Copy `Travel_details_dataset.csv` into `data/raw/travel_raw.csv`.

### 4. Run the full pipeline

```bash
python -m src.main
```

This single command executes: load → clean → validate → 3-way split → build recipe → train → evaluate (validation + test) → inference → save artifacts. All metrics and artifacts are logged to W&B when `WANDB_MODE=online`.

### 5. Promote the model in W&B

After a successful training run:
1. Open the W&B Dashboard → Project → Artifacts
2. Find the latest model version
3. Click **"Add Alias"** and type `prod`

This promoted model is used by the API for live inference.

### 6. Run the test suite

```bash
python -m pytest -v
```

With coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

### 7. Start the API locally

```bash
uvicorn src.api:app --reload
```

Test endpoints:
- Health check: `GET http://127.0.0.1:8000/health`
- Interactive docs: `http://127.0.0.1:8000/docs`
- Predict:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      {
        "destination_country": "France",
        "duration_days": 7,
        "traveler_age": 35,
        "traveler_gender": "Female",
        "traveler_nationality": "United States",
        "accommodation_type": "Hotel",
        "transportation_type": "Flight"
      }
    ]
  }'
```

### 8. Build and run with Docker

```bash
docker build -t voyageiq-api:latest .
docker run -p 8000:8000 --env-file .env voyageiq-api:latest
```

---

## Live Deployment

| Resource | URL |
|---|---|
| **Live API** | [https://voyageiq-api.onrender.com](https://voyageiq-api.onrender.com) |
| **Health Check** | [https://voyageiq-api.onrender.com/health](https://voyageiq-api.onrender.com/health) |
| **API Docs (Swagger)** | [https://voyageiq-api.onrender.com/docs](https://voyageiq-api.onrender.com/docs) |
| **W&B Project** | [https://wandb.ai/mohammad-alkhan-ie-university/voyageiq-trip-cost](https://wandb.ai/mohammad-alkhan-ie-university/voyageiq-trip-cost) |
| **GitHub Release** | See [Releases](../../releases) page — production deploys are tied to formal GitHub Releases |

---

## Pipeline Architecture

```
config.yaml + .env
    │
    ▼
┌──────────┐    ┌────────────┐    ┌────────────┐
│ load_data │───▶│ clean_data │───▶│  validate  │
└──────────┘    └────────────┘    └────────────┘
                                        │
                              ┌─────────▼─────────┐
                              │  train / val /     │  ◀── train_test_split (BEFORE recipe)
                              │   test split       │
                              └─────────┬──────────┘
                                        │
                              ┌─────────▼─────────┐
                              │  features.py       │  build unfitted ColumnTransformer
                              └─────────┬──────────┘
                                        │
                              ┌─────────▼─────────┐
                              │    train.py        │  fit(preprocessor + model) on TRAIN only
                              └─────────┬──────────┘
                                        │
                              ┌─────────▼─────────┐
                              │   evaluate.py      │  score on VAL then TEST → metrics dict
                              └─────────┬──────────┘
                                        │
                         ┌──────────────┼──────────────┐
                         ▼              ▼              ▼
                   ┌──────────┐  ┌───────────┐  ┌──────────┐
                   │  W&B     │  │  infer.py │  │ api.py   │
                   │  logging │  │  (batch)  │  │ (live)   │
                   └──────────┘  └───────────┘  └──────────┘
```

---

## Key Design Decisions

1. **Config-driven pipeline**: All non-secret runtime settings live in `config.yaml`. Secrets live in `.env`. Only code-level constants (endpoints, schemas, logging formats) remain in code.
2. **Three-way split**: Data is partitioned into Train (70 %) / Validation (15 %) / Test (15 %) BEFORE building the feature recipe to prevent data leakage.
3. **Unfitted recipe pattern**: `features.py` returns a `ColumnTransformer` that has never seen data. Only `train.py` calls `.fit()` on the training split.
4. **Unified Pipeline artifact**: The preprocessor and model are bundled in a single `sklearn.pipeline.Pipeline`, saved as `model.joblib`. New data goes through `pipeline.predict(df)` with no manual preprocessing.
5. **Fail-fast validation**: `validate.py` catches schema violations, missing columns, and out-of-range values before expensive compute.
6. **W&B experiment tracking**: `main.py` owns all W&B tracking — row counts, metrics, model artifact, processed data artifact. The production model is promoted with `alias=prod` and served from the W&B registry.
7. **Zero `print()` policy**: All production code uses `logging` via `src/logger.py` with dual output (console + local file).
8. **API contains no ML logic**: `api.py` is a thin wrapper that calls `clean_dataframe()`, `validate_dataframe()`, and `run_inference()`. Pydantic enforces the JSON request contract.
9. **Lean Docker image**: Built from `conda-lock.yml` for zero environment drift. `.dockerignore` excludes tests, notebooks, data, reports, wandb cache, and dev artifacts.
10. **Release-driven deployment**: `deploy.yml` triggers only when a human publishes a GitHub Release, ensuring deliberate production deploys. `ci.yml` validates all PRs (tests + Docker build).

---

## Monitoring & Observability

Operational traceability is achieved through three layers:

- **Local logs**: `src/logger.py` writes to both console and `logs/pipeline.log` with timestamps, log levels, and module names. Zero `print()` in production code.
- **W&B run logs**: Each pipeline run logs row counts, validation/test metrics, artifacts, and config snapshots to the [W&B project](https://wandb.ai/mohammad-alkhan-ie-university/voyageiq-trip-cost).
- **Render service logs**: The deployed API streams logs to Render's dashboard for real-time debugging and health monitoring.

---

## Model Card

| Field | Details |
|---|---|
| **Model Type** | RandomForestRegressor wrapped in an sklearn `Pipeline` (preprocessor + estimator) |
| **Input Features** | `duration_days`, `traveler_age`, `travel_month`, `day_of_week`, `destination_country`, `traveler_gender`, `traveler_nationality`, `accommodation_type`, `transportation_type` |
| **Target** | `total_cost` (accommodation_cost + transportation_cost) |
| **Training Data** | 716 cleaned rows from 739 raw synthetic travel booking records |
| **Preprocessing** | `StandardScaler` for numeric, `OneHotEncoder` for categorical, `SimpleImputer` for missing values |
| **Validation RMSE** | 2255.56 |
| **Test RMSE** | 988.27 |
| **Test R²** | 0.103 |
| **Test MAPE** | Tracked in W&B — target is < 10 % |
| **W&B Artifact** | Model artifact stored in W&B, promoted with `alias=prod` for production inference |
| **Limitations** | Small dataset (739 rows); may not generalise well to unseen destinations or seasonal patterns. Model predicts combined total cost only — separate accommodation vs. flight breakdowns would require two models. |
| **Intended Use** | Internal cost estimation for travel agency quoting. Not intended for consumer-facing pricing without human review. |

---

## Changelog

| Version | Date | Changes |
|---|---|---|
| **v2.0.0** | Mar 2026 | W&B experiment tracking and model registry (`prod` alias). FastAPI serving (`api.py` with Pydantic contracts, `/health` and `/predict` endpoints). Docker containerisation with `conda-lock.yml`. CI/CD with GitHub Actions (`ci.yml` for PR validation, `deploy.yml` for release-triggered Render deploy). Dual-output logging via `src/logger.py` (zero `print()` policy). Monitoring via local logs, W&B, and Render. Live deployment on Render. |
| **v1.0.0** | Mar 2026 | Initial modular pipeline — `load_data`, `clean_data`, `validate`, `features`, `train`, `evaluate`, `infer`, `main`. Config-driven via `config.yaml` + `.env`. Three-way split (70/15/15). Unified `sklearn.Pipeline` artifact. 39+ tests covering core modules and edge cases. Dataset expanded from ~139 to ~739 rows with synthetic data generation. |

---

## CI/CD Workflows

| Workflow | Trigger | Actions |
|---|---|---|
| **`ci.yml`** | Pull Request to `main` | Set up Miniconda → install from `conda-lock.yml` → run `pytest` → validate `docker build` |
| **`deploy.yml`** | GitHub Release published | Trigger Render deploy hook → redeploy live API |

Environment variables in CI: `WANDB_MODE=disabled`, `MODEL_SOURCE=local` — tests validate code logic, not external services.

---

## GitHub Workflow

| Branch | Purpose |
|---|---|
| `main` | Protected. Stable, reviewed code only. CI must pass before merge. |
| `dev` | Integration branch for features. |
| `feature/<name>` | One branch per feature or module. |

**Workflow**: feature branch → PR into `dev` → review → merge → PR into `main` → CI passes → merge.

**Release workflow**: Merge to `main` → GitHub Release from `main` (e.g. `v2.0.0`) → triggers `deploy.yml` → Render redeploys.

---

## Team Module Ownership

| Member              | Files |
|---------------------|-------|
| MOHAMMAD ALKHAN     | `config.yaml`, `environment.yml`, `README.md`, `01_voyageiq_analysis_vExp.ipynb`, `test_utils.py`, `test_validate.py` |
| MICHAEL CONCEPCION  | `test_clean_data.py`, `test_evaluate.py`, `test_features.py` |
| CLAUDIA MOLINER     | `infer.py`, `load_data.py`, `main.py`, `utils.py`, `01_voyageiq_analysis_Legacy.ipynb` |
| NICKLAS URBAN       | `train.py`, `validate.py`, `conftest.py` |
| NICOLE ZLOTCHEVSKY  | `clean_data.py`, `evaluate.py`, `features.py` |
| JAUME BALAGUER      | `test_infer.py`, `test_load_data.py`, `test_main.py`, `test_train.py` |

---

## License

This project is developed for academic purposes as part of the MLOps Engineering course at IE University.
