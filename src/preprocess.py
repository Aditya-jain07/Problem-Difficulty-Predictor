# src/preprocess.py

import os
import pandas as pd

from config import (
    RAW_DATA_FILE,
    PROCESSED_DATA_FILE,
    TARGET_CLASS_COLUMN,
    TARGET_SCORE_COLUMN
)
from load_data import load_jsonl_data


TEXT_COLUMNS = [
    "description",
    "input_description",
    "output_description"
]

# Keywords suggested by problem statement / domain knowledge
KEYWORDS = [
    "graph", "dp", "dynamic programming", "recursion",
    "bfs", "dfs", "tree", "segment", "shortest",
    "complexity", "optimize", "modulo"
]


def preprocess_data() -> pd.DataFrame:
    """
    Load raw data, clean text, engineer features,
    and return processed DataFrame.
    """
    # Load raw data
    df = load_jsonl_data(RAW_DATA_FILE)

    # Keep only required columns
    required_columns = TEXT_COLUMNS + [
        TARGET_CLASS_COLUMN,
        TARGET_SCORE_COLUMN
    ]
    df = df[required_columns]

    # Drop rows with missing or empty text
    df = df.dropna(subset=TEXT_COLUMNS)
    for col in TEXT_COLUMNS:
        df = df[df[col].str.strip() != ""]

    # -------------------------
    # Feature Engineering
    # -------------------------

    # 1️⃣ Text length features
    df["len_description"] = df["description"].str.len()
    df["len_input_description"] = df["input_description"].str.len()
    df["len_output_description"] = df["output_description"].str.len()

    # Combine text fields
    df["full_text"] = (
        df["description"] + " " +
        df["input_description"] + " " +
        df["output_description"]
    )

    # 2️⃣ Keyword frequency count
    def keyword_count(text: str) -> int:
        text = text.lower()
        return sum(text.count(keyword) for keyword in KEYWORDS)

    df["keyword_count"] = df["full_text"].apply(keyword_count)

    return df


if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)

    df = preprocess_data()

    # Save processed data
    df.to_csv(PROCESSED_DATA_FILE, index=False)

    print("Preprocessing completed successfully")
    print("Final dataset shape:", df.shape)
    print("Columns:")
    for col in df.columns:
        print(" -", col)
