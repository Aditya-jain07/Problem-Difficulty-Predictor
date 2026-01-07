# src/config.py

import os

# ===============================
# Reproducibility
# ===============================
RANDOM_STATE = 42

# ===============================
# Base Directories
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.path.join(BASE_DIR, "models")

# ===============================
# Data Files
# ===============================
RAW_DATA_FILE = os.path.join(
    RAW_DATA_DIR,
    "problems_data.jsonl"
)

PROCESSED_DATA_FILE = os.path.join(
    PROCESSED_DATA_DIR,
    "dataset.csv"
)

# ===============================
# Column Names
# ===============================
TEXT_COLUMNS = [
    "description",
    "input_description",
    "output_description"
]

TARGET_CLASS_COLUMN = "problem_class"
TARGET_SCORE_COLUMN = "problem_score"

# ===============================
# Train / Test Split
# ===============================
TEST_SIZE = 0.2
