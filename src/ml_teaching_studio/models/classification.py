"""Classification model constructors."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def _optional_int(value: Any) -> int | None:
    if value in ("", None, 0, "0"):
        return None
    return int(value)


def logistic_regression(random_state: int | None = None, **params: Any) -> LogisticRegression:
    penalty = params.get("penalty", "l2")
    solver = params.get("solver", "lbfgs")
    if penalty == "l1" and solver not in {"liblinear", "saga"}:
        solver = "liblinear"
    kwargs = dict(
        C=float(params.get("C", 1.0)),
        solver=solver,
        max_iter=int(params.get("max_iter", 1000)),
        random_state=random_state,
    )
    if penalty != "l2":
        kwargs["penalty"] = penalty
    return LogisticRegression(**kwargs)


def knn_classifier(random_state: int | None = None, **params: Any) -> KNeighborsClassifier:
    del random_state
    return KNeighborsClassifier(
        n_neighbors=int(params.get("n_neighbors", 5)),
        weights=params.get("weights", "uniform"),
        metric=params.get("metric", "minkowski"),
    )


def decision_tree_classifier(
    random_state: int | None = None, **params: Any
) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(
        criterion=params.get("criterion", "gini"),
        max_depth=_optional_int(params.get("max_depth", 5)),
        min_samples_split=int(params.get("min_samples_split", 2)),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        random_state=random_state,
    )


def random_forest_classifier(
    random_state: int | None = None, **params: Any
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 200)),
        max_depth=_optional_int(params.get("max_depth", 8)),
        max_features=params.get("max_features", "sqrt"),
        min_samples_leaf=int(params.get("min_samples_leaf", 1)),
        random_state=random_state,
        n_jobs=-1,
    )


def svm_classifier(random_state: int | None = None, **params: Any) -> SVC:
    gamma = params.get("gamma", "scale")
    if isinstance(gamma, str) and gamma not in {"scale", "auto"}:
        try:
            gamma = float(gamma)
        except ValueError:
            gamma = "scale"
    return SVC(
        C=float(params.get("C", 1.0)),
        kernel=params.get("kernel", "rbf"),
        gamma=gamma,
        probability=True,
        random_state=random_state,
    )


def naive_bayes_classifier(random_state: int | None = None, **params: Any) -> GaussianNB:
    del random_state
    return GaussianNB(var_smoothing=float(params.get("var_smoothing", 1e-9)))


def gradient_boosting_classifier(
    random_state: int | None = None, **params: Any
) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=int(params.get("n_estimators", 150)),
        learning_rate=float(params.get("learning_rate", 0.1)),
        max_depth=int(params.get("max_depth", 3)),
        random_state=random_state,
    )


def adaboost_classifier(random_state: int | None = None, **params: Any) -> AdaBoostClassifier:
    return AdaBoostClassifier(
        n_estimators=int(params.get("n_estimators", 100)),
        learning_rate=float(params.get("learning_rate", 1.0)),
        random_state=random_state,
    )


def mlp_classifier(random_state: int | None = None, **params: Any) -> MLPClassifier:
    hidden_layers = params.get("hidden_layer_sizes", (100,))
    if isinstance(hidden_layers, str):
        hidden_layers = tuple(int(part.strip()) for part in hidden_layers.split(",") if part.strip())
    return MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layers),
        activation=params.get("activation", "relu"),
        learning_rate_init=float(params.get("learning_rate_init", 0.001)),
        alpha=float(params.get("alpha", 0.0001)),
        batch_size=int(params.get("batch_size", 32)),
        max_iter=int(params.get("max_iter", 400)),
        random_state=random_state,
    )


CLASSIFICATION_BUILDERS = {
    "Logistic Regression": logistic_regression,
    "K-Nearest Neighbors": knn_classifier,
    "Decision Tree Classifier": decision_tree_classifier,
    "Random Forest Classifier": random_forest_classifier,
    "Support Vector Machine": svm_classifier,
    "Naive Bayes": naive_bayes_classifier,
    "Gradient Boosting Classifier": gradient_boosting_classifier,
    "AdaBoost Classifier": adaboost_classifier,
    "MLP Classifier": mlp_classifier,
}
