# Project Name: Telco Customer Churn Prediction
test_change
other_branch_change

**Author:** Group 6 
**Course:** MLOps: Master in Business Analytics and Data Science
**Status:** Orchestration

---

## 1. Business Objective

* **The Goal:** 
The goal of this project is to develop a machine learning model capable of predicting customer churn in a telecommunications company. By identifying customers who are likely to discontinue their service, the company can proactively implement retention strategies, reduce revenue loss, and improve overall customer lifetime value. The model is designed to generate churn probability scores that support data-driven decision-making and resource allocation.

* **The User:** 
The primary users of the model’s output are marketing and customer retention teams. These stakeholders will use churn probability predictions to prioritize high-risk customers and design targeted intervention campaigns. In a more advanced stage, the model could also support business analysts through reporting dashboards or be integrated into automated systems for operational decision-making.

---

## 2. Success Metrics

* **Business KPI (The "Why"):**
From a business perspective, the project will be considered successful if it contributes to a measurable reduction in customer churn and supports improved retention performance. The ultimate objective is to decrease churn-related revenue loss and enhance customer lifetime value. Precise financial targets will be defined in alignment with business stakeholders.

* **Technical Metric (The "How"):**
From a technical perspective, the primary evaluation metric is the Area Under the Receiver Operating Characteristic Curve (AUC-ROC), which measures the model’s ability to discriminate between churners and non-churners across different thresholds. The baseline implementation achieves an AUC of approximately 0.839 and an accuracy of approximately 0.795. Additional evaluation metrics include precision and recall to ensure balanced performance.

* **Acceptance Criteria:**
The model will be accepted if it demonstrates performance superior to a random classifier and outperforms a simple baseline approach such as majority class prediction. In addition, the training and evaluation process must be fully reproducible through the production pipeline implemented in the src/ directory.

---

## 3. The Data

* **Source:** 
The project uses the publicly available Telco Customer Churn dataset, provided in CSV format. The dataset contains 7,043 customer records and 21 variables, including demographic information, subscription characteristics, service usage features, and billing details.

* **Target Variable:** 
The target variable is Churn, which indicates whether a customer has discontinued their service. This variable is treated as a binary classification outcome, where “Yes” is encoded as 1 and “No” is encoded as 0.

* **Sensitive Info:** 
The dataset does not contain direct personally identifiable information such as email addresses, credit card numbers, or national identification numbers. Although it includes a customerID field, this variable serves only as a unique identifier and is excluded from the modeling process. All raw data files are stored locally and excluded from version control to ensure compliance with good data governance practices.

---

## 4. Repository Structure

This project follows a strict separation between "Sandbox" (Notebooks) and "Production" (Src).

```text
.
├── README.md                 # Project overview and documentation
├── environment.yml           # Conda environment dependencies
├── config.yaml               # Global configuration (paths, parameters)

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
│   └── main.py               # Pipeline orchestrator (entry point)

├── data/                     # Data storage (not tracked if large)
│   ├── raw/                  # Original, immutable raw data
│   ├── processed/            # Cleaned and transformed datasets
│   └── inference/            # Data used for prediction/inference

├── models/                   # Serialized trained models (ignored by Git)

├── reports/                  # Generated reports, metrics, and visualizations

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

## 5. Execution Model
 
Build and activate the Conda environment:
```
conda env create -f environment.yml
conda activate mlops-modul
```



