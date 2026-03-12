from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

from ml_teaching_studio.core.model_factory import create_model


def test_create_logistic_regression_model() -> None:
    model = create_model(
        "Logistic Regression",
        {"C": 1.5, "penalty": "l2", "solver": "lbfgs"},
        random_state=42,
    )
    assert isinstance(model, LogisticRegression)
    assert model.C == 1.5


def test_create_random_forest_regressor_model() -> None:
    model = create_model(
        "Random Forest Regressor",
        {"n_estimators": 50, "max_depth": 5, "max_features": "sqrt", "min_samples_leaf": 1},
        random_state=42,
    )
    assert isinstance(model, RandomForestRegressor)
    assert model.n_estimators == 50
