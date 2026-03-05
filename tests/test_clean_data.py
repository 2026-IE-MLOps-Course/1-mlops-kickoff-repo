import pandas as pd
from src.clean_data import load_raw, clean_data, save_df, validate_df

SAMPLE = {
    "customerID": ["0001", "0002", "0003", "0003"],
    "tenure": ["1", "2", "3", "3"],
    "MonthlyCharges": [29.85, 56.95, 53.85, 53.85],
    "TotalCharges": ["29.85", " ", "161.65", "161.65"],
    "Churn": ["No", "Yes", "No", "No"]
}

def test_load_and_clean_happy_path(tmp_path):
    df = pd.DataFrame(SAMPLE)
    raw_file = tmp_path / "raw.csv"
    df.to_csv(raw_file, index=False)
    df_loaded = load_raw(str(raw_file))
    assert df_loaded.shape[0] == 4
    df_clean = clean_data(df_loaded, drop_na=True)
    assert df_clean.shape[0] == 3
    assert "TotalCharges" in df_clean.columns

def test_total_charges_becomes_numeric_after_cleaning():
    df = pd.DataFrame({
        "customerID": ["1", "2"],
        "tenure": ["1", "2"],
        "MonthlyCharges": [10.0, 20.0],
        "TotalCharges": ["10.0", "not_a_number"],
        "Churn": ["No", "Yes"]
    })
    df_clean = clean_data(df, drop_na=False)
    assert pd.api.types.is_numeric_dtype(df_clean["TotalCharges"])

def test_save_and_validate(tmp_path):
    df = pd.DataFrame(SAMPLE).drop_duplicates()
    out_file = tmp_path / "clean.csv"
    save_df(df, str(out_file))
    assert out_file.exists()
    df_read = pd.read_csv(out_file)
    ok, _ = validate_df(df_read)
    assert ok is True
