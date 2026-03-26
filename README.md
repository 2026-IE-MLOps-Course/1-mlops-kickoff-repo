# Project Name: Telco Customer Churn Prediction

**Author:** Group 6
**Course:** MLOps: Master in Business Analytics and Data Science
**Status:** Experiment Tracking

---

## 1. Business Objective

* **The Goal:**
The goal of this project is to develop a machine learning model capable of predicting customer churn in a telecommunications company. By identifying customers who are likely to discontinue their service, the company can proactively implement retention strategies, reduce revenue loss, and improve overall customer lifetime value. The model is designed to generate churn probability scores that support data-driven decision-making and resource allocation.

* **The User:**
The primary users of the model's output are marketing and customer retention teams. These stakeholders will use churn probability predictions to prioritize high-risk customers and design targeted intervention campaigns. In a more advanced stage, the model could also support business analysts through reporting dashboards or be integrated into automated systems for operational decision-making.

---

## 2. Success Metrics

* **Business KPI (The "Why"):**
From a business perspective, the project will be considered successful if it contributes to a measurable reduction in customer churn and supports improved retention performance. The ultimate objective is to decrease churn-related revenue loss and enhance customer lifetime value. Precise financial targets will be defined in alignment with business stakeholders.

* **Technical Metric (The "How"):**
From a technical perspective, the primary evaluation metric is the Area Under the Receiver Operating Characteristic Curve (AUC-ROC), which measures the model's ability to discriminate between churners and non-churners across different thresholds. The baseline implementation achieves an AUC of approximately 0.839 and an accuracy of approximately 0.795. Additional evaluation metrics include precision and recall to ensure balanced performance.

* **Acceptance Criteria:**
The model will be accepted if it demonstrates performance superior to a random classifier and outperforms a simple baseline approach such as majority class prediction. In addition, the training and evaluation process must be fully reproducible through the production pipeline implemented in the src/ directory.

---

## 3. The Data

* **Source:**
The project uses the publicly available Telco Customer Churn dataset, provided in CSV format. The dataset contains 7,043 customer records and 21 variables, including demographic information, subscription characteristics, service usage features, and billing details.

* **Target Variable:**
The target variable is Churn, which indicates whether a customer has discontinued their service. This variable is treated as a binary classification outcome, where "Yes" is encoded as 1 and "No" is encoded as 0.

* **Sensitive Info:**
The dataset does not contain direct personally identifiable information such as email addresses, credit card numbers, or national identification numbers. Although it includes a customerID field, this variable serves only as a unique identifier and is excluded from the modeling process. All raw data files are stored locally and excluded from version control to ensure compliance with good data governance practices.

---

## 4. Repository Structure

This project follows a strict separation between "Sandbox" (Notebooks) and "Production" (Src).

```text
.
├── README.md                 # Project overview and documentation
├── environment.yml           # Conda environment dependencies
├── config.yaml               # Global configuration (paths, parameters, W&B settings)
├── .env                      # Secrets — never committed (see .env.example)
├── .env.example              # Template showing required environment variables

├── notebooks/                # Experimental and exploratory work
│   └── New Final.ipynb       # Notebook from previous analysis

├── src/                      # Production-ready source code (pipeline modules)
│   ├── __init__.py           # Makes src a Python package
│   ├── load_data.py          # Data ingestion from raw sources
│   ├── clean_data.py         # Data preprocessing and cleaning
│   ├── features.py           # Feature engineering logic
│   ├── validate.py           # Data validation and quality checks
│   ├── train.py              # Model training and model persistence
│   ├── evaluate.py           # Model evaluation and metrics generation
│   ├── infer.py              # Inference logic for new/unseen data
│   ├── logger.py             # Centralized logging configuration (console + file)
│   └── main.py               # Pipeline orchestrator (entry point)

├── data/                     # Data storage (not tracked in Git)
│   ├── raw/                  # Original, immutable raw data (place raw dataset here)
│   ├── processed/            # Cleaned and transformed datasets
│   └── inference/            # Data used for prediction/inference

├── models/                   # Serialized trained models (ignored by Git)

├── logs/                     # Pipeline run logs (ignored by Git)

├── reports/                  # Generated reports and predictions

└── tests/                    # Automated unit tests
    ├── test_clean_data.py    # Tests for preprocessing module
    ├── test_evaluate.py      # Tests for evaluation module
    ├── test_features.py      # Tests for feature engineering
    ├── test_infer.py         # Tests for inference logic
    ├── test_load_data.py     # Tests for data loading
    ├── test_main.py          # Tests for pipeline orchestration
    ├── test_train.py         # Tests for training module
    └── test_validate.py      # Tests for validation checks
```

---

## 5. Configuration

All runtime settings are controlled through `config.yaml`. No values are hardcoded in the source code.

| Section | Purpose |
|---|---|
| `paths` | File paths for data, model, predictions, and log file |
| `logging` | Log level and format |
| `problem` | Target column, problem type, identifier column |
| `split` | Train/test split ratio and random seed |
| `training.classification` | Model hyperparameters (max_iter, random_state) |
| `validation` | Required columns and cleaning options |
| `wandb` | Experiment tracking settings (see section 7) |

---

## 6. Execution Model

### Step 0: Add the Raw Data

Before running the pipeline, you must manually place the dataset in the correct location.
1. Download the Telco Customer Churn dataset (CSV format)
2. Create the folder if it does not exist:
```
data/raw/
```
3. Place the dataset inside this folder with the following name:
```
data/raw/telco_chirn.csv
```

If the dataset is missing or incorrectly placed, the pipeline will fail during data loading stage.

### Step 1: Environment Setup

Build and activate the Conda environment:
```
conda env create -f environment.yml
conda activate mlops_project
```

### Step 2: Configure Secrets

Copy the environment template and fill in your credentials:
```
cp .env.example .env
```

Edit `.env` and set your Weights & Biases API key:
```
WANDB_API_KEY=your_key_here
WANDB_ENTITY=your_entity_here
WANDB_PROJECT=telco-churn-logit
```
Your W&B API key can be found at https://wandb.ai/authorize.

### Step 3: Exploratory Sandbox (The Laboratory)

The `notebooks/New Final.ipynb` file represents the experimental baseline implementation of the project. In this notebook, the full analytical workflow was developed and validated before modularization.

The baseline model achieved:
- AUC ≈ 0.839
- Accuracy ≈ 0.795
- Precision ≈ 0.637
- Recall ≈ 0.535

These results confirm strong discriminative performance and provide a benchmark for the production pipeline.

### Step 4: Run the Test Suite

Before executing the full pipeline, verify the integrity of the codebase through automated testing:
```
pytest tests/ -v
```
A fully passing test suite confirms that module contracts are respected and the pipeline is stable.

### Step 5: Execute the Orchestrator

Run the complete machine learning pipeline:
```
python -m src.main
```
This ensures deterministic, reproducible execution and prevents training-serving skew.

---

## 7. Experiment Tracking (Weights & Biases)

The pipeline integrates with [Weights & Biases](https://wandb.ai) for experiment tracking. It is controlled entirely from `config.yaml` under the `wandb:` section.

To enable tracking set `wandb.enabled: true` in `config.yaml`. To disable it set `wandb.enabled: false` — the pipeline runs normally either way.

When enabled, the following is logged automatically per run:

| What | Where in W&B |
|---|---|
| Raw and clean dataset row/column counts | `data/` metrics |
| AUC evaluation metric | `metrics/` |
| ROC curve and Precision-Recall curve | `plots/` |
| Confusion matrix | `plots/` |
| Trained model pipeline | Model artifact |
| Predictions CSV | Predictions artifact |
| Predictions preview table | `tables/` |
| Full `config.yaml` contents | Run config |

---

## 8. Outputs Generated

Upon execution of the orchestrator, the following artifacts are produced:

1. `data/processed/clean.csv` — The deterministically cleaned dataset
2. `models/model.joblib` — The serialized preprocessing and model pipeline
3. `reports/predictions.csv` — Model predictions and churn probabilities
4. `logs/pipeline.log` — Timestamped log of the full pipeline run

These artifacts ensure reproducibility, traceability, and auditability.

---

## 9. Academic Purpose

This repository serves as a practical implementation of Machine Learning Operations (MLOps) principles. The objective is to transition from a monolithic Jupyter Notebook into a modular, testable, and production-oriented architecture.

**Learning Outcomes:**
- Translate exploratory analysis into modular production code
- Enforce separation of concerns across pipeline stages
- Prevent data leakage through structured split boundaries
- Validate data quality before model training
- Centralize all runtime settings in a single configuration file
- Implement structured logging to console and file
- Track experiments and artifacts with Weights & Biases
- Produce reproducible artifacts
- Implement automated tests for behavioral validation
