# src/load_data.py

import json
import pandas as pd
from config import RAW_DATA_FILE


def load_jsonl_data(file_path: str) -> pd.DataFrame:
    """
    Load a JSONL file and return a pandas DataFrame.
    
    Parameters:
        file_path (str): Path to the JSONL file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    # Quick sanity check
    df = load_jsonl_data(RAW_DATA_FILE)
    print("Dataset loaded successfully")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
