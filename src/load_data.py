from pathlib import Path
import pandas as pd

from src.utils import load_csv, save_csv
from src.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Inputs:
    - raw_data_path: Path where the raw CSV should exist.
    Outputs:
    - df_raw: Raw pandas DataFrame.
    Why this contract matters for reliable ML delivery:
    - Makes the pipeline deterministic: same path contract across
      machines and CI.
    """
    logger.info(f"Loading raw data from: {raw_data_path}")

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Notebook-style convenience for this assignment
    # (Telco churn):
    # Why: In class environments, the CSV may exist outside the repo;
    # copying it into data/raw makes the pipeline reproducible.
    # Examples:
    # 1. Read from an alternate path and save to data/raw
    # 2. Replace with your real data ingestion (DB/API) later
    alt_path = Path("/mnt/data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    if (not raw_data_path.exists()) and alt_path.exists():
        logger.info(
            "Found Telco churn CSV at /mnt/data. Copying into "
            "repo data/raw for reproducible runs."
        )
        df_alt = pd.read_csv(alt_path)
        save_csv(df_alt, raw_data_path)
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

def load_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

        logger.warning(
            "\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            "!!!!!!!!\n"
            "Reference data/raw/dataset.csv not found.\n"
            "Dummy dataset created for scaffolding ONLY.\n"
            f"Path: {raw_data_path}\n"
            "Columns are hardcoded: num_feature, cat_feature, target\n"
            "You MUST replace this with your real dataset and update "
            "SETTINGS.\n"
            "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
            "!!!!!!!!\n"
        )

    if path.suffix == ".csv":
        return pd.read_csv(path)

    if path.suffix == ".parquet":
        return pd.read_parquet(path)

if __name__ == "__main__":
    df = load_raw_data(Path("data/raw/telco.csv"))
    logger.debug(f"DataFrame shape: {df.shape}")
    logger.debug(f"DataFrame head:\n{df.head()}")
