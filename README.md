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
│       ├── ci.yml            # CI — runs tests + linting on PRs to main
│       └── deploy.yml        # CD — triggers Render deploy on GitHub Release
├── data/
│   ├── raw/
│   │   └── travel_raw.csv    # Original dataset (git-ignored)
│   ├── processed/
│   │   └── clean.csv         # Cleaned output (git-ignored)
│   └── inference/
│       └── .gitkeep
├── models/
│   └── model.joblib          # Trained pipeline artifact (git-ignored)
├── reports/
│   └── predictions.csv       # Inference output (git-ignored)
├── logs/
│   └── pipeline.log          # Dual-output log file (git-ignored)
├── notebooks/
│   ├── 01_voyageiq_analysis_Legacy.ipynb
│   └── 01_voyageiq_analysis_vExp.ipynb
├── src/
│   ├── __init__.py           # Marks src as a Python package
│   ├── logger.py             # Dual-output logging (console + file)
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

### 2. Configure secrets

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required secrets: `WANDB_API_KEY`, `WANDB_ENTITY`.

### 3. Place the dataset

Copy `Travel_details_dataset.csv` into `data/raw/travel_raw.csv`.

### 4. Run the full pipeline

```bash
python -m src.main
```

This single command executes: load → clean → validate → 3-way split → build recipe → train → evaluate (validation + test) → inference → save artifacts. All metrics and artifacts are logged to W&B.

### 5. Run the test suite

```bash
python -m pytest -v
```

### 6. Start the API locally

```bash
uvicorn src.api:app --reload
```

Test endpoints:
- Health: http://127.0.0.1:8000/health
- Docs: http://127.0.0.1:8000/docs
- Predict: `POST /predict` with JSON payload

### 7. Build and run with Docker

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
| **API Docs** | [https://voyageiq-api.onrender.com/docs](https://voyageiq-api.onrender.com/docs) |
| **W&B Project** | [https://wandb.ai/mohammad-alkhan-ie-university/voyageiq-trip-cost](https://wandb.ai/mohammad-alkhan-ie-university/voyageiq-trip-cost) |

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
                              │  train / val /    │  ◀── train_test_split (BEFORE recipe)
                              │   test split      │
                              └─────────┬─────────┘
                                        │
                              ┌─────────▼─────────┐
                              │  features.py      │  build unfitted ColumnTransformer
                              └─────────┬─────────┘
                                        │
                              ┌─────────▼─────────┐
                              │    train.py       │  fit(preprocessor + model) on TRAIN only
                              └─────────┬─────────┘
                                        │
                              ┌─────────▼─────────┐
                              │   evaluate.py     │  score on VAL then TEST → metrics dict
                              └─────────┬─────────┘
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

1. **Config-driven pipeline**: All non-secret runtime settings live in `config.yaml`. Secrets live in `.env`. Only code-level constants remain in code.
2. **Three-way split**: Data is partitioned into Train / Validation / Test BEFORE building the feature recipe to prevent leakage.
3. **Unfitted recipe pattern**: `features.py` returns a `ColumnTransformer` that has never seen data. Only `train.py` calls `.fit()` on the training split.
4. **Unified Pipeline artifact**: The preprocessor and model are bundled in a single `sklearn.pipeline.Pipeline` so inference is identical to training.
5. **Fail-fast validation**: `validate.py` catches schema violations, missing columns, and out-of-range values before expensive compute.
6. **W&B experiment tracking**: `main.py` owns all W&B tracking — metrics, artifacts, and model promotion with `prod` alias.
7. **Zero print() policy**: All production code uses `logging` with dual output (console + file). No `print()` statements in `src/`.
8. **API contains no ML logic**: `api.py` is a thin wrapper that calls `clean_dataframe()`, `validate_dataframe()`, and `run_inference()`.
9. **Lean Docker image**: `.dockerignore` excludes tests, notebooks, data, reports, wandb cache, and dev artifacts.
10. **Release-driven deployment**: `deploy.yml` triggers only when a human publishes a GitHub Release, ensuring deliberate production deploys.

---

## Model Card

| Field | Details |
|---|---|
| **Model Type** | RandomForestRegressor (sklearn Pipeline) |
| **Input Features** | duration_days, traveler_age, travel_month, day_of_week, destination_country, traveler_gender, traveler_nationality, accommodation_type, transportation_type |
| **Target** | total_cost (accommodation_cost + transportation_cost) |
| **Training Data** | 716 cleaned rows from 739 raw travel booking records |
| **Preprocessing** | StandardScaler for numeric, OneHotEncoder for categorical, SimpleImputer for missing values |
| **Validation RMSE** | 2255.56 |
| **Test RMSE** | 988.27 |
| **Test R²** | 0.103 |
| **W&B Project** | [voyageiq-trip-cost](https://wandb.ai/mohammad-alkhan-ie-university/voyageiq-trip-cost) |
| **Limitations** | Small dataset, may not generalise well to unseen destinations or seasonal patterns. Model predicts combined total cost only — separate accommodation vs. flight breakdowns would require two models. |

---

## Changelog

| Version | Date | Changes |
|---|---|---|
| v1.0.0 | Mar 2026 | Initial modular pipeline — load, clean, validate, features, train, evaluate, infer. SETTINGS dictionary bridge. Three-way split. 39+ tests. |
| v2.0.0 | Mar 2026 | Config-driven pipeline (config.yaml + .env). W&B experiment tracking and model registry (prod alias). FastAPI serving (api.py with Pydantic). Docker containerisation. CI/CD with GitHub Actions (ci.yml + deploy.yml). Render deployment. Dual-output logging via src/logger.py (zero print statements). |

---

## GitHub Workflow

| Branch | Purpose |
|---|---|
| `main` | Protected. Stable, reviewed code only. |
| `dev` | Integration branch for features. |
| `feature/<name>` | One branch per feature or module. |

**Workflow**: feature branch → PR into `dev` → review → merge → PR into `main` → CI passes → merge.

**Release workflow**: GitHub Release from `main` → triggers `deploy.yml` → Render redeploys.

---

## Team Module Ownership

| Member              | Files |
|---------------------|-------|
| MOHAMMAD ALKHAN     | `config.yaml`, `environment.yml`, `01_voyageiq_analysis_vExp.ipynb`, `test_utils.py`, `test_validate.py` |
| MICHAEL CONCEPCION  | `test_clean_data.py`, `test_evaluate.py`, `test_features.py` |
| CLAUDIA ARANGUREN   | `infer.py`, `load_data.py`, `main.py`, `utils.py`, `01_voyageiq_analysis_Legacy.ipynb` |
| NICKLAS URBAN       | `train.py`, `validate.py`, `conftest.py`, `README.md` |
| NICOLE ZLOTCHEVSKY  | `clean_data.py`, `evaluate.py`, `features.py` |
| JAUME BALAGUER      | `test_infer.py`, `test_load_data.py`, `test_main.py`, `test_train.py` |

---

## License

This project is developed for academic purposes as part of the MLOps Engineering course at IE University.
