# src/train_regressor_ridge.py

import os
import joblib
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

from features import build_features
from config import MODELS_DIR


def train_ridge_regressor():
    """
    Train Ridge Regression with GridSearchCV
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

    ridge = Ridge()

    param_grid = {
        "alpha": [0.01, 0.1, 1.0, 10.0, 50.0]
    }

    grid_search = GridSearchCV(
        estimator=ridge,
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

    best_ridge = grid_search.best_estimator_

    # Test evaluation
    y_pred = best_ridge.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\nTest MAE:", round(mae, 4))
    print("Test RMSE:", round(rmse, 4))

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_ridge, os.path.join(MODELS_DIR, "regressor_ridge_best.pkl"))

    print("\nBest Ridge regressor saved successfully.")


if __name__ == "__main__":
    train_ridge_regressor()
