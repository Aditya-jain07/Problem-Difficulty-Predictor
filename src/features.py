# src/features.py

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

from config import (
    PROCESSED_DATA_FILE,
    MODELS_DIR,
    TARGET_CLASS_COLUMN,
    TARGET_SCORE_COLUMN,
    TEST_SIZE,
    RANDOM_STATE
)

# Numeric features kept after preprocessing decision
NUMERIC_FEATURES = [
    "len_description",
    "len_input_description",
    "len_output_description",
    "keyword_count"
]


def build_features():
    """
    Build TF-IDF + numeric features and split into train/test sets.
    """
    # Load processed data
    df = pd.read_csv(PROCESSED_DATA_FILE)

    # -------------------------
    # Text features (TF-IDF)
    # -------------------------
    X_text = df["full_text"]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_tfidf = vectorizer.fit_transform(X_text)

    # Save vectorizer
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf.pkl"))

    # -------------------------
    # Numeric features
    # -------------------------
    X_num = df[NUMERIC_FEATURES].values

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Save scaler
    joblib.dump(scaler, os.path.join(MODELS_DIR, "numeric_scaler.pkl"))

    # Stack sparse TF-IDF with dense numeric features
    X = hstack([X_tfidf, X_num_scaled])

    # -------------------------
    # Targets
    # -------------------------
    y_class = df[TARGET_CLASS_COLUMN]
    y_score = df[TARGET_SCORE_COLUMN]

    # -------------------------
    # Train-test split
    # -------------------------
    X_train, X_test, y_class_train, y_class_test, y_score_train, y_score_test = train_test_split(
        X,
        y_class,
        y_score,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_class
    )

    return (
        X_train,
        X_test,
        y_class_train,
        y_class_test,
        y_score_train,
        y_score_test
    )


if __name__ == "__main__":
    data = build_features()

    print("Feature extraction completed")
    print("Training samples:", data[0].shape[0])
    print("Test samples:", data[1].shape[0])
    print("Total features:", data[0].shape[1])
