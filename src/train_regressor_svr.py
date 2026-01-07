# src/train_regressor_svr.py

import os
import joblib
import numpy as np

from sklearn.svm import LinearSVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import build_features
from config import MODELS_DIR


def train_svr_regressor():
    """
    Train Linear SVR with GridSearchCV
    for difficulty score prediction.
    """

    (
        X_train,
        X_test,
        _,
        _,
        y_train,
        y_test
    ) = build_features()

    svr = LinearSVR(
        max_iter=5000,
        random_state=42
    )

    param_grid = {
        "C": [0.01, 0.1, 1.0, 5.0],
        "epsilon": [0.01, 0.1, 0.5]
    }

    grid_search = GridSearchCV(
        estimator=svr,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("\nBest parameters found:")
    print(grid_search.best_params_)

    print("\nBest CV MAE:")
    print(round(-grid_search.best_score_, 4))

    best_svr = grid_search.best_estimator_

    # Test evaluation
    y_pred = best_svr.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\nTest MAE:", round(mae, 4))
    print("Test RMSE:", round(rmse, 4))

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_svr, os.path.join(MODELS_DIR, "regressor_svr_best.pkl"))

    print("\nBest SVR regressor saved successfully.")


if __name__ == "__main__":
    train_svr_regressor()
