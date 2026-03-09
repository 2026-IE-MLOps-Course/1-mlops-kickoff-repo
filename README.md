# VoyageIQ – Total Trip Cost Forecasting Pipeline

> **MLOps Engineering · 1st Group Assignment**
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
| **Solution Description** | An end-to-end ML regression pipeline that ingests traveller and trip attributes (destination, duration, accommodation type, transport mode, demographics) and outputs a predicted total trip cost. The pipeline is fully modular, config-driven, and production-ready for deployment behind a REST API. |
| **Solution Scalability** | The architecture generalises to any tabular regression/classification task. Additional data sources (hotel APIs, flight aggregators) can be plugged into `load_data.py`. New models (XGBoost, neural nets) are added by extending the model factory in `train.py`. The same pipeline can serve B2B corporate bulk-quoting and B2C instant pricing. |
| **Client Benefit** | *Short-term:* Instant cost estimates → faster sales cycle. *Long-term:* Dynamic pricing optimisation, personalised packages. *Competitiveness:* Data-driven quoting vs. competitors' manual processes. Measured by: reduction in MAPE, quote turnaround time, and agent productivity. |
| **Cost Estimation ($000)** | Talent: AI specialist ($15 k), Product Mgr. ($10 k), ML/SW Engineer ($20 k), Data Engineer ($15 k), SME ($5 k). Client covers: data infrastructure, cloud compute, licences. Timeline: 12+ weeks. |
| **Risks & Challenges** | Data quality (small dataset, potential PII leakage), model generalisation to unseen destinations, currency fluctuations, seasonal price swings. *Mitigations:* robust validation gate, .gitignore for PII, periodic retraining schedule, config-driven feature selection. |

---

## Repository Structure

```
.
├── README.md                 # This file
├── config.yaml               # Reserved for future YAML migration
├── environment.yml           # Conda dependency management
├── .gitignore                # Git exclusion rules
├── pytest.ini                # Pytest configuration
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
├── notebooks/
│   └── .gitkeep
├── src/
│   ├── __init__.py           # Empty — marks src as a Python package
│   ├── utils.py              # I/O plumbing: load_csv, save_csv, save_model, load_model
│   ├── load_data.py          # Data ingestion (load_raw_data)
│   ├── clean_data.py         # Data cleaning / stabilisation (clean_dataframe)
│   ├── validate.py           # Schema & range validation (validate_dataframe)
│   ├── features.py           # Feature preprocessor recipe (get_feature_preprocessor)
│   ├── train.py              # Model training (train_model)
│   ├── evaluate.py           # Metric computation (evaluate_model → float)
│   ├── infer.py              # Inference / prediction (run_inference → DataFrame)
│   └── main.py               # Pipeline orchestrator with SETTINGS dictionary
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
    └── test_main.py
```

---

## Public API Contracts

All modules expose functions with fixed signatures to guarantee interoperability:

| Module | Function | Signature |
|---|---|---|
| `utils.py` | `load_csv` | `(filepath: Path) → pd.DataFrame` |
| `utils.py` | `save_csv` | `(df: pd.DataFrame, filepath: Path) → None` |
| `utils.py` | `save_model` | `(model, filepath: Path) → None` |
| `utils.py` | `load_model` | `(filepath: Path) → model` |
| `load_data.py` | `load_raw_data` | `(raw_data_path: Path) → pd.DataFrame` |
| `clean_data.py` | `clean_dataframe` | `(df_raw: pd.DataFrame, target_column: str) → pd.DataFrame` |
| `validate.py` | `validate_dataframe` | `(df: pd.DataFrame, required_columns: list) → bool` |
| `features.py` | `get_feature_preprocessor` | `(quantile_bin_cols, categorical_onehot_cols, numeric_passthrough_cols, n_bins) → ColumnTransformer` |
| `train.py` | `train_model` | `(X_train, y_train, preprocessor, problem_type: str) → Pipeline` |
| `evaluate.py` | `evaluate_model` | `(model, X_test, y_test, problem_type: str) → float` |
| `infer.py` | `run_inference` | `(model, X_infer: pd.DataFrame) → pd.DataFrame` |
| `main.py` | `main` | `() → None` (+ `if __name__ == "__main__": main()`) |

---

## How to Run

### 1. Set up the environment

```bash
conda env create -f environment.yml
conda activate mlops-student-env
```

### 2. Place the dataset

Copy `Travel_details_dataset.csv` into `data/raw/travel_raw.csv`.

### 3. Run the full pipeline

```bash
python -m src.main
```

This single command executes: load → clean → validate → 3-way split → build recipe → train → evaluate (validation + test) → inference → save artifacts.

### 4. Run the test suite

```bash
python -m pytest -q
```

> You should see 100 % passing tests!

---

## Outputs Generated

1. `data/processed/clean.csv` — The deterministically cleaned input data
2. `models/model.joblib` — The deployable pipeline artifact (preprocessor + model)
3. `reports/predictions.csv` — The inference log containing predictions

---

## Pipeline Architecture

```
SETTINGS (main.py)
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
                              │   evaluate.py     │  score on VAL then TEST → single float
                              └─────────┬─────────┘
                                        │
                              ┌─────────▼─────────┐
                              │    infer.py       │  predict → DataFrame("prediction")
                              └───────────────────┘
```

---

## Key Design Decisions

1. **SETTINGS dictionary bridge**: All configuration lives in a central `SETTINGS` dict in `main.py`, acting as a bridge to future `config.yaml` migration.
2. **Three-way split**: Data is partitioned into Train / Validation / Test BEFORE building the feature recipe to prevent leakage.
3. **Unfitted recipe pattern**: `features.py` returns a `ColumnTransformer` that has never seen data. Only `train.py` calls `.fit()` on the training split.
4. **Unified Pipeline artifact**: The preprocessor and model are bundled in a single `sklearn.pipeline.Pipeline` so inference is identical to training.
5. **Fail-fast validation**: `validate.py` catches schema violations, missing columns, and out-of-range values before expensive compute.
6. **Dual problem-type support**: The pipeline supports both regression (RMSE) and classification (F1 weighted) via the `problem_type` setting.
7. **Student Action Blocks**: Every module contains clearly marked `START/END STUDENT CODE` blocks where domain-specific logic is implemented.

---

## Future Roadmap (Upcoming Sessions)

* Move `SETTINGS` into `config.yaml` and add environment-based secrets via `.env`
* Replace `print()` statements with standard library `logging` and structured logs
* Add MLflow for experiment tracking and model registry
* Containerise and serve predictions via a FastAPI application

---

## GitHub Workflow

| Branch | Purpose |
|---|---|
| `main` | Protected. Stable, reviewed code only. |
| `dev` | Protected. Integration branch for features. |
| `feature/<module-name>` | One branch per module per team member. |

**Workflow**: feature branch → PR into `dev` → review → merge → PR into `main`.

Each team member owns ~2 modules + their tests. Commit early, commit often, and push.

---

## Team Module Ownership 
| Member              | File |
|---------------------|------|

| MOHAMMAD ALKHAN     |  `config.yaml`, `environment.yml`, `01_voyageiq_analysis_vExp.iynb`, `test_utils.py`, `test_validate.py` |
| MICHAEL CONCEPCION  | `test_clean_data.py`, `test_evaluate.py` , `test_features.py` |
| CLAUDIA ARANGUREN     | `infer.py`, `load_data.py` , `main.py` , `utils.py`, `01_voyageiq_analysis_Legacy.iynb` |
| NICKLAS URBAN       | `train.py`, `validate.py`, `confest.py`, `README.md` |
| NICOLE ZLOTCHEVSKY  | `clean_data.py`, `evaluate.py`, `features.py`  |
| JAUME BALAGUER      | `test_infer.py`, `test_load_data.py`, `test_main.py`, `test_train.py` |

---

## License

This project is developed for academic purposes as part of the MLOps Engineering course at IE University.