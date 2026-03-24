import numpy as np
import pandas as pd


def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Clean raw dataframe for Telco churn pipeline.
    Keeps rows unless they are fully empty.
    """

    print("[clean_data.clean_dataframe] Cleaning dataframe")

    if df_raw is None:
        raise ValueError("df_raw is None")

    df_clean = df_raw.copy(deep=True)

    # 1. Standardize column names
    df_clean.columns = [col.strip() for col in df_clean.columns]

    # 2. Trim whitespace and convert blank strings to NaN
    for col in df_clean.select_dtypes(include=["object", "string"]).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        df_clean[col] = df_clean[col].replace(
            {
                "": np.nan,
                "nan": np.nan,
                "<NA>": np.nan,
                "None": np.nan,
                "null": np.nan,
            }
        )

    # 3. Convert TotalCharges to numeric if present
    if "TotalCharges" in df_clean.columns:
        df_clean["TotalCharges"] = pd.to_numeric(
            df_clean["TotalCharges"], errors="coerce"
        )

    # 4. Fill numeric NaNs with median
    for col in df_clean.select_dtypes(include=["number"]).columns:
        if df_clean[col].isna().any():
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)

    # 5. Fill categorical NaNs with mode
    for col in df_clean.select_dtypes(include=["object", "string", "category"]).columns:
        if col != target_column and df_clean[col].isna().any():
            mode_s = df_clean[col].mode()
            if not mode_s.empty:
                df_clean[col] = df_clean[col].fillna(mode_s.iloc[0])
            else:
                df_clean[col] = df_clean[col].fillna("Missing")

    # 6. Drop only obvious ID column
    if "customerID" in df_clean.columns:
        df_clean = df_clean.drop(columns=["customerID"])

    # 7. Encode target for Telco churn
    if target_column in df_clean.columns:
        df_clean[target_column] = df_clean[target_column].replace(
            {"No": 0, "Yes": 1, "False": 0, "True": 1}
        )
        df_clean[target_column] = pd.to_numeric(
            df_clean[target_column], errors="coerce"
        )

    # 8. Drop rows that are entirely empty
    df_clean = df_clean.dropna(how="all")

    print(f"[clean_data.clean_dataframe] Raw shape: {df_raw.shape}")
    print(f"[clean_data.clean_dataframe] Clean shape: {df_clean.shape}")

    return df_clean