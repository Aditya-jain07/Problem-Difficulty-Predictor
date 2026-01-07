# src/train_classifier_logreg.py

import os
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from features import build_features
from config import MODELS_DIR


def train_logreg_with_gridsearch():
    """
    Train Logistic Regression with GridSearchCV
    using TF-IDF + numeric features.
    """

    (
        X_train,
        X_test,
        y_class_train,
        y_class_test,
        _,
        _
    ) = build_features()

    # Base Logistic Regression (correct for sparse text)
    logreg = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        n_jobs=-1,
        solver="saga"   # REQUIRED for sparse + l1/l2
    )

    # Hyperparameter grid (tight & meaningful)
    param_grid = {
        "C": [0.01, 0.1, 1.0, 5.0],
        "penalty": ["l2"]  
    }

    # Grid Search
    grid_search = GridSearchCV(
        estimator=logreg,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    # Train
    grid_search.fit(X_train, y_class_train)

    print("\nBest parameters found:")
    print(grid_search.best_params_)

    print("\nBest CV F1-macro score:")
    print(round(grid_search.best_score_, 4))

    # Best model
    best_logreg = grid_search.best_estimator_

    # Test evaluation
    y_pred = best_logreg.predict(X_test)

    acc = accuracy_score(y_class_test, y_pred)
    f1 = f1_score(y_class_test, y_pred, average="macro")

    print("\nTest Set Accuracy:", round(acc, 4))
    print("Test Set F1-macro:", round(f1, 4))

    print("\nClassification Report:\n")
    print(classification_report(y_class_test, y_pred))

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_class_test, y_pred))

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_logreg, os.path.join(MODELS_DIR, "classifier_logreg_best.pkl"))

    print("\nBest Logistic Regression model saved successfully.")


if __name__ == "__main__":
    train_logreg_with_gridsearch()
