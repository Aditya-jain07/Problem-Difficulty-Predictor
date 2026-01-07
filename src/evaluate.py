# src/evaluate.py

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    f1_score
)

from features import build_features
from config import MODELS_DIR


# ===============================
# Analysis Parameters
# ===============================
TOP_K_WORDS = 15


def score_to_class(score):
    """Convert regression score to difficulty class."""
    if score < 3.5:
        return "easy"
    elif score < 6.5:
        return "medium"
    else:
        return "hard"


def show_top_words_per_class(model, vectorizer, top_k=15):
    """
    Explainability: show top contributing words
    for each difficulty class.
    """
    feature_names = vectorizer.get_feature_names_out()
    classes = model.classes_

    print("\n" + "=" * 60)
    print("TOP WORDS PER DIFFICULTY CLASS (EXPLAINABILITY)")
    print("=" * 60)

    for idx, class_label in enumerate(classes):
        coef = model.coef_[idx]
        top_indices = coef.argsort()[-top_k:][::-1]

        print(f"\nClass: {class_label}")
        for i in top_indices:
            print(f"  {feature_names[i]}")


def evaluate_models():
    (
        X_train,
        X_test,
        y_class_train,
        y_class_test,
        y_score_train,
        y_score_test
    ) = build_features()

    # ===============================
    # Load Final Models
    # ===============================
    clf = joblib.load(f"{MODELS_DIR}/classifier_logreg_best.pkl")
    reg = joblib.load(f"{MODELS_DIR}/regressor_svr_best.pkl")
    vectorizer = joblib.load(f"{MODELS_DIR}/tfidf.pkl")

    # ===============================
    # CLASSIFICATION EVALUATION
    # ===============================
    print("\n" + "=" * 70)
    print("CLASSIFICATION EVALUATION (FINAL MODEL)")
    print("=" * 70)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)

    acc = accuracy_score(y_class_test, y_pred)
    f1 = f1_score(y_class_test, y_pred, average="macro")

    print("Accuracy:", round(acc, 4))
    print("F1-macro:", round(f1, 4))
    print("\nClassification Report:\n")
    print(classification_report(y_class_test, y_pred))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_class_test, y_pred))

    # ===============================
    # PROBABILITY-BASED PREDICTIONS (INTERNAL)
    # ===============================
    print("\n" + "-" * 60)
    print("SAMPLE PROBABILITY-BASED PREDICTIONS (INTERNAL)")
    print("-" * 60)

    for i in range(5):
        probs = y_prob[i]
        prob_str = ", ".join(
            f"{cls}: {probs[j]:.2f}"
            for j, cls in enumerate(clf.classes_)
        )
        print(
            f"True: {y_class_test.iloc[i]} | "
            f"Pred: {y_pred[i]} | {prob_str}"
        )

    # ===============================
    # REGRESSION EVALUATION
    # ===============================
    print("\n" + "=" * 70)
    print("REGRESSION EVALUATION (FINAL MODEL)")
    print("=" * 70)

    y_score_pred = reg.predict(X_test)

    mae = mean_absolute_error(y_score_test, y_score_pred)
    rmse = np.sqrt(mean_squared_error(y_score_test, y_score_pred))

    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))

    # ===============================
    # REGRESSION → CLASS CALIBRATION
    # ===============================
    print("\n" + "-" * 60)
    print("REGRESSION → CLASS CALIBRATION (CONSISTENCY CHECK)")
    print("-" * 60)

    calibrated = [score_to_class(s) for s in y_score_pred]

    for i in range(5):
        print(
            f"True: {y_class_test.iloc[i]} | "
            f"Classifier: {y_pred[i]} | "
            f"Reg-derived: {calibrated[i]} | "
            f"Score: {round(y_score_pred[i], 2)}"
        )

    # ===============================
    # EXPLAINABILITY
    # ===============================
    show_top_words_per_class(clf, vectorizer, TOP_K_WORDS)


if __name__ == "__main__":
    evaluate_models()
