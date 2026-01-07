# src/predict.py

import joblib
import numpy as np
from scipy.sparse import hstack

from config import MODELS_DIR
from preprocess import KEYWORDS


def extract_numeric_features(description, input_desc, output_desc):
    """
    Extract numeric features for a single problem instance.
    Must match training-time feature order.
    """
    len_description = len(description)
    len_input = len(input_desc)
    len_output = len(output_desc)

    full_text = f"{description} {input_desc} {output_desc}".lower()
    keyword_count = sum(full_text.count(k) for k in KEYWORDS)

    return np.array([[len_description, len_input, len_output, keyword_count]])


def predict_difficulty(description, input_desc, output_desc):
    """
    FINAL user-facing prediction function.

    Returns:
    - difficulty label (easy / medium / hard)
    - difficulty score (float)
    """

    # ===============================
    # Load trained artifacts
    # ===============================
    clf = joblib.load(f"{MODELS_DIR}/classifier_logreg_best.pkl")
    reg = joblib.load(f"{MODELS_DIR}/regressor_svr_best.pkl")
    vectorizer = joblib.load(f"{MODELS_DIR}/tfidf.pkl")
    scaler = joblib.load(f"{MODELS_DIR}/numeric_scaler.pkl")

    # ===============================
    # Build features
    # ===============================
    full_text = f"{description} {input_desc} {output_desc}"

    # TF-IDF features
    X_text = vectorizer.transform([full_text])

    # Numeric features
    X_num = extract_numeric_features(
        description,
        input_desc,
        output_desc
    )
    X_num_scaled = scaler.transform(X_num)

    # Combine
    X = hstack([X_text, X_num_scaled])

    # ===============================
    # Predictions
    # ===============================
    difficulty_label = clf.predict(X)[0]
    difficulty_score = float(reg.predict(X)[0])

    return difficulty_label, round(difficulty_score, 2)


# ===============================
# CLI testing (optional)
# ===============================
if __name__ == "__main__":
    label, score = predict_difficulty(
        description="Given a graph with N nodes and M edges.",
        input_desc="The first line contains N and M.",
        output_desc="Print the length of the shortest path."
    )

    print("Predicted Difficulty:", label)
    print("Predicted Difficulty Score:", score)
