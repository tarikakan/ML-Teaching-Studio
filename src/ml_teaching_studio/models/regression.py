"""Regression model constructors."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def _optional_int(value: Any) -> int | None:
    if value in ("", None, 0, "0"):
        return None
    return int(value)


def linear_regression(random_state: int | None = None, **params: Any) -> LinearRegression:
    del random_state, params
    return LinearRegression()


def ridge_regression(random_state: int | None = None, **params: Any) -> Ridge:
    return Ridge(alpha=float(params.get("alpha", 1.0)), random_state=random_state)


def lasso_regression(random_state: int | None = None, **params: Any) -> Lasso:
    return Lasso(alpha=float(params.get("alpha", 0.1)), random_state=random_state, max_iter=5000)


def decision_tree_regressor(
    random_state: int | None = None, **params: Any
) -> DecisionTreeRegressor:
    return DecisionTreeRegressor(
        criterion=params.get("criterion", "squared_error"),
        max_depth=_optional_int(params.get("max_depth", 5)),
        min_samples_split=int(params.get("min_samples_split", 2)),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        random_state=random_state,
    )


def random_forest_regressor(
    random_state: int | None = None, **params: Any
) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=int(params.get("n_estimators", 200)),
        max_depth=_optional_int(params.get("max_depth", 8)),
        max_features=params.get("max_features", "sqrt"),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        random_state=random_state,
        n_jobs=-1,
    )


def svr_regressor(random_state: int | None = None, **params: Any) -> SVR:
    del random_state
    gamma = params.get("gamma", "scale")
    if isinstance(gamma, str) and gamma not in {"scale", "auto"}:
        try:
            gamma = float(gamma)
        except ValueError:
            gamma = "scale"
    return SVR(
        C=float(params.get("C", 1.0)),
        kernel=params.get("kernel", "rbf"),
        gamma=gamma,
        epsilon=float(params.get("epsilon", 0.1)),
    )


def gradient_boosting_regressor(
    random_state: int | None = None, **params: Any
) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=int(params.get("n_estimators", 150)),
        learning_rate=float(params.get("learning_rate", 0.05)),
        max_depth=int(params.get("max_depth", 3)),
        random_state=random_state,
    )


def mlp_regressor(random_state: int | None = None, **params: Any) -> MLPRegressor:
    hidden_layers = params.get("hidden_layer_sizes", (100, 50))
    if isinstance(hidden_layers, str):
        hidden_layers = tuple(int(part.strip()) for part in hidden_layers.split(",") if part.strip())
    return MLPRegressor(
        hidden_layer_sizes=tuple(hidden_layers),
        activation=params.get("activation", "relu"),
        learning_rate_init=float(params.get("learning_rate_init", 0.001)),
        alpha=float(params.get("alpha", 0.0001)),
        batch_size=int(params.get("batch_size", 32)),
        max_iter=int(params.get("max_iter", 500)),
        random_state=random_state,
    )


REGRESSION_BUILDERS = {
    "Linear Regression": linear_regression,
    "Ridge Regression": ridge_regression,
    "Lasso Regression": lasso_regression,
    "Decision Tree Regressor": decision_tree_regressor,
    "Random Forest Regressor": random_forest_regressor,
    "Support Vector Regressor": svr_regressor,
    "Gradient Boosting Regressor": gradient_boosting_regressor,
    "MLP Regressor": mlp_regressor,
}
