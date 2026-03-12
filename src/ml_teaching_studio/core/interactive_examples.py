"""Interactive example engine for live model demonstrations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split

from ml_teaching_studio.educational.hyperparameter_help import get_hyperparameter_help
from ml_teaching_studio.models.model_registry import default_model_for_task, get_model_spec, list_model_names

from .metrics import compute_metrics
from .model_factory import create_model
from .preprocessing import PreprocessingOptions, build_training_pipeline


@dataclass(frozen=True)
class InteractiveScenario:
    name: str
    task_type: str
    description: str


@dataclass
class InteractiveExampleResult:
    task_type: str
    model_name: str
    scenario_name: str
    scenario_description: str
    hyperparameters: dict[str, Any]
    feature_columns: list[str]
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    grid_frame: pd.DataFrame | None = None
    grid_predictions: np.ndarray | None = None
    grid_scores: np.ndarray | None = None
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: list[Any] = field(default_factory=list)
    y_test: list[Any] = field(default_factory=list)
    y_train_pred: list[Any] = field(default_factory=list)
    y_test_pred: list[Any] = field(default_factory=list)
    curve_frame: pd.DataFrame | None = None
    curve_predictions: np.ndarray | None = None


@dataclass
class InteractiveComparisonResult:
    task_type: str
    scenario_name: str
    scenario_description: str
    primary: InteractiveExampleResult
    comparison: InteractiveExampleResult


INTERACTIVE_SCENARIOS = {
    "classification": [
        InteractiveScenario(
            name="Curved Moons",
            task_type="classification",
            description="Two intertwined classes. Flexible models can curve around the moons, while simpler models stay smoother.",
        ),
        InteractiveScenario(
            name="Separated Clusters",
            task_type="classification",
            description="Two cleaner clusters. Simpler models can already perform well, so extra flexibility may not help much.",
        ),
    ],
    "regression": [
        InteractiveScenario(
            name="Smooth Curve",
            task_type="regression",
            description="A moderately noisy nonlinear curve. Good for seeing when a model is too rigid or appropriately smooth.",
        ),
        InteractiveScenario(
            name="Noisy Curve",
            task_type="regression",
            description="A noisier regression pattern. Overly flexible settings tend to chase local noise instead of the underlying shape.",
        ),
    ],
}


def interactive_scenarios(task_type: str) -> list[InteractiveScenario]:
    return INTERACTIVE_SCENARIOS.get(task_type, [])


def comparison_candidates(task_type: str, active_model_name: str) -> list[str]:
    names = list_model_names(task_type)
    if active_model_name in names:
        return [active_model_name] + [name for name in names if name != active_model_name]
    return names


def suggested_comparison_model(task_type: str, active_model_name: str) -> str:
    names = comparison_candidates(task_type, active_model_name)
    baseline = default_model_for_task(task_type)
    if baseline != active_model_name and baseline in names:
        return baseline
    fallbacks = {
        "classification": "K-Nearest Neighbors",
        "regression": "Decision Tree Regressor",
    }
    preferred = fallbacks.get(task_type)
    if preferred and preferred != active_model_name and preferred in names:
        return preferred
    return names[0] if names else active_model_name


def default_demo_hyperparameters(model_name: str, limit: int = 3) -> list[str]:
    spec = get_model_spec(model_name)
    if spec.important_hyperparameters:
        return list(spec.important_hyperparameters[:limit])
    return list(spec.hyperparameters.keys())[:limit]


def _classification_frame(name: str, random_state: int) -> tuple[pd.DataFrame, str]:
    if name == "Separated Clusters":
        X, y = make_blobs(
            n_samples=260,
            n_features=2,
            centers=[(-2.0, -1.2), (2.2, 1.6)],
            cluster_std=[1.1, 1.25],
            random_state=random_state,
        )
        description = next(
            scenario.description for scenario in INTERACTIVE_SCENARIOS["classification"] if scenario.name == name
        )
    else:
        X, y = make_moons(n_samples=260, noise=0.26, random_state=random_state)
        description = next(
            scenario.description for scenario in INTERACTIVE_SCENARIOS["classification"] if scenario.name == "Curved Moons"
        )
    frame = pd.DataFrame(X, columns=["x1", "x2"])
    frame["target"] = np.where(y == 1, "Class B", "Class A")
    return frame, description


def _regression_frame(name: str, random_state: int) -> tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(random_state)
    x = np.linspace(-3.0, 3.0, 220)
    signal = np.sin(1.4 * x) + 0.28 * x
    noise_scale = 0.18 if name == "Smooth Curve" else 0.42
    y = signal + rng.normal(0.0, noise_scale, len(x))
    frame = pd.DataFrame({"x": x, "x_squared": x**2})
    frame["target"] = y
    description = next(
        scenario.description for scenario in INTERACTIVE_SCENARIOS["regression"] if scenario.name == name
    )
    return frame, description


def _example_frame(task_type: str, scenario_name: str, random_state: int) -> tuple[pd.DataFrame, str]:
    if task_type == "classification":
        return _classification_frame(scenario_name, random_state)
    return _regression_frame(scenario_name, random_state)


def _grid_for_classification(frame: pd.DataFrame) -> pd.DataFrame:
    x_min, x_max = frame["x1"].min() - 0.7, frame["x1"].max() + 0.7
    y_min, y_max = frame["x2"].min() - 0.7, frame["x2"].max() + 0.7
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 180), np.linspace(y_min, y_max, 180))
    return pd.DataFrame({"x1": xx.ravel(), "x2": yy.ravel()})


def _curve_for_regression(frame: pd.DataFrame) -> pd.DataFrame:
    x_values = np.linspace(frame["x"].min(), frame["x"].max(), 300)
    return pd.DataFrame({"x": x_values, "x_squared": x_values**2})


def _probability_scores(pipeline: Any, features: pd.DataFrame) -> np.ndarray | None:
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return pipeline.predict_proba(features)
    if hasattr(model, "decision_function"):
        return np.asarray(pipeline.decision_function(features))
    return None


def _primary_metric_name(task_type: str) -> str:
    return "accuracy" if task_type == "classification" else "r2"


def _primary_metric_label(task_type: str) -> str:
    return "accuracy" if task_type == "classification" else "R²"


def _example_split(
    *,
    task_type: str,
    scenario_name: str,
    random_state: int,
) -> tuple[pd.DataFrame, str, list[str], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    frame, description = _example_frame(task_type, scenario_name, random_state)
    feature_columns = [column for column in frame.columns if column != "target"]
    y = frame["target"]
    X = frame[feature_columns]
    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=random_state,
        stratify=stratify,
    )
    return frame, description, feature_columns, X_train, X_test, y_train, y_test


def _fit_interactive_model(
    *,
    frame: pd.DataFrame,
    scenario_name: str,
    scenario_description: str,
    task_type: str,
    model_name: str,
    hyperparameters: dict[str, Any],
    feature_columns: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int,
) -> InteractiveExampleResult:
    estimator = create_model(model_name, hyperparameters, random_state=random_state)
    pipeline = build_training_pipeline(
        frame,
        feature_columns,
        estimator,
        PreprocessingOptions(
            scale_numeric=True,
            normalize_numeric=False,
            impute_missing=True,
            encode_categorical=True,
        ),
    )
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_scores = _probability_scores(pipeline, X_train)
    test_scores = _probability_scores(pipeline, X_test)
    train_metrics = compute_metrics(task_type, y_train, y_train_pred, y_score=train_scores)
    test_metrics = compute_metrics(task_type, y_test, y_test_pred, y_score=test_scores)

    result = InteractiveExampleResult(
        task_type=task_type,
        model_name=model_name,
        scenario_name=scenario_name,
        scenario_description=scenario_description,
        hyperparameters=hyperparameters,
        feature_columns=feature_columns,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.tolist(),
        y_test=y_test.tolist(),
        y_train_pred=np.asarray(y_train_pred).tolist(),
        y_test_pred=np.asarray(y_test_pred).tolist(),
    )

    if task_type == "classification":
        grid_frame = _grid_for_classification(frame)
        grid_predictions = pipeline.predict(grid_frame)
        result.grid_frame = grid_frame
        result.grid_predictions = np.asarray(grid_predictions)
        probability_scores = _probability_scores(pipeline, grid_frame)
        if probability_scores is not None:
            probability_scores = np.asarray(probability_scores)
            if probability_scores.ndim == 2 and probability_scores.shape[1] > 1:
                result.grid_scores = probability_scores[:, 1]
            elif probability_scores.ndim == 1:
                result.grid_scores = probability_scores
    else:
        curve_frame = _curve_for_regression(frame)
        result.curve_frame = curve_frame
        result.curve_predictions = np.asarray(pipeline.predict(curve_frame))

    return result


def _score_gap(result: InteractiveExampleResult) -> float | None:
    metric_name = _primary_metric_name(result.task_type)
    train_score = result.train_metrics.get(metric_name)
    test_score = result.test_metrics.get(metric_name)
    if train_score is None or test_score is None:
        return None
    return train_score - test_score


def interactive_metric_frame(result: InteractiveExampleResult | InteractiveComparisonResult) -> pd.DataFrame:
    if isinstance(result, InteractiveComparisonResult):
        items = [result.primary, result.comparison]
    else:
        items = [result]

    rows: list[dict[str, Any]] = []
    for item in items:
        if item.task_type == "classification":
            rows.append(
                {
                    "Model": item.model_name,
                    "Train accuracy": round(item.train_metrics.get("accuracy", 0.0), 3),
                    "Test accuracy": round(item.test_metrics.get("accuracy", 0.0), 3),
                    "Test f1": round(item.test_metrics.get("f1_weighted", 0.0), 3),
                    "Gap": round((_score_gap(item) or 0.0), 3),
                }
            )
        else:
            rows.append(
                {
                    "Model": item.model_name,
                    "Train R²": round(item.train_metrics.get("r2", 0.0), 3),
                    "Test R²": round(item.test_metrics.get("r2", 0.0), 3),
                    "Test RMSE": round(item.test_metrics.get("rmse", 0.0), 3),
                    "Gap": round((_score_gap(item) or 0.0), 3),
                }
            )
    return pd.DataFrame(rows)


def build_interactive_explanation(
    result: InteractiveExampleResult,
    *,
    focus_parameter: str | None = None,
) -> str:
    primary_metric = _primary_metric_name(result.task_type)
    metric_label = _primary_metric_label(result.task_type)
    train_score = result.train_metrics.get(primary_metric)
    test_score = result.test_metrics.get(primary_metric)
    gap = _score_gap(result)
    parameter_help = get_hyperparameter_help(result.model_name)

    parts = [
        f"<h3>{result.model_name}: live example</h3>",
        f"<p><b>Scenario:</b> {result.scenario_name}. {result.scenario_description}</p>",
    ]
    if result.hyperparameters:
        summary = ", ".join(f"{name}={value}" for name, value in result.hyperparameters.items())
        parts.append(f"<p><b>Current settings:</b> {summary}</p>")
    if train_score is not None and test_score is not None:
        parts.append(f"<p><b>Train {metric_label}:</b> {train_score:.3f} | <b>Test {metric_label}:</b> {test_score:.3f}</p>")
        if gap is not None:
            if gap > 0.12:
                parts.append(
                    "<p>The training score is noticeably ahead of the test score, so the current setting is likely too flexible for this example.</p>"
                )
            elif result.task_type == "classification" and test_score < 0.75:
                parts.append(
                    "<p>The boundary is still fairly simple relative to the pattern. This looks closer to underfitting than overfitting.</p>"
                )
            elif result.task_type == "regression" and test_score < 0.45:
                parts.append(
                    "<p>The fitted curve is missing part of the underlying shape. That usually means the model or current setting is still too constrained.</p>"
                )
            else:
                parts.append(
                    "<p>The train and test scores are relatively close, so this region of the hyperparameter space looks more stable.</p>"
                )

    highlighted = focus_parameter
    if not highlighted and result.hyperparameters:
        highlighted = next(iter(result.hyperparameters))
    if highlighted and highlighted in parameter_help:
        help_entry = parameter_help[highlighted]
        current_value = result.hyperparameters.get(highlighted, "default")
        parts.append(
            f"<p><b>{highlighted}</b> is currently <b>{current_value}</b>. {help_entry['plain_language']} "
            f"{help_entry['algorithmic_role']}</p>"
        )
        parts.append(f"<p><b>Interpretation:</b> {help_entry['impact']} {help_entry['tip']}</p>")

    parts.append(
        "<p><b>How to use this view:</b> change one setting at a time and watch whether the boundary or fitted curve becomes smoother, more jagged, or less stable on the test split.</p>"
    )
    return "".join(parts)


def build_interactive_comparison_explanation(
    result: InteractiveComparisonResult,
    *,
    focus_parameter: str | None = None,
) -> str:
    metric_name = _primary_metric_name(result.task_type)
    metric_label = _primary_metric_label(result.task_type)
    primary_score = result.primary.test_metrics.get(metric_name, 0.0)
    comparison_score = result.comparison.test_metrics.get(metric_name, 0.0)
    score_delta = primary_score - comparison_score
    primary_gap = _score_gap(result.primary) or 0.0
    comparison_gap = _score_gap(result.comparison) or 0.0
    same_model = result.primary.model_name == result.comparison.model_name

    parts = [
        f"<h3>Comparison: {result.primary.model_name} vs {result.comparison.model_name}</h3>",
        f"<p><b>Scenario:</b> {result.scenario_name}. Both models use the same synthetic dataset and the same train/test split, so the comparison is fair.</p>",
    ]
    if result.primary.hyperparameters:
        summary = ", ".join(f"{name}={value}" for name, value in result.primary.hyperparameters.items())
        parts.append(f"<p><b>Active settings on {result.primary.model_name}:</b> {summary}</p>")

    if same_model:
        parts.append(
            "<p>You are comparing the current hyperparameter setting against the same model's default settings. This isolates the effect of tuning more clearly than switching model families.</p>"
        )
    else:
        parts.append(
            f"<p>{result.comparison.model_name} is acting as the reference model. This helps you separate model-family differences from hyperparameter changes inside the active model.</p>"
        )

    parts.append(
        f"<p><b>Test {metric_label}:</b> {result.primary.model_name} = {primary_score:.3f}, {result.comparison.model_name} = {comparison_score:.3f}</p>"
    )
    if abs(score_delta) < 0.02:
        parts.append(
            "<p>The test scores are very close, so the main lesson is likely about decision shape, smoothness, or stability rather than raw performance.</p>"
        )
    elif score_delta > 0:
        parts.append(
            f"<p>{result.primary.model_name} is ahead on unseen data. The visual difference is worth studying because the better score came from a more suitable bias-variance balance on this scenario.</p>"
        )
    else:
        parts.append(
            f"<p>{result.comparison.model_name} is ahead on unseen data. The active setting may be too rigid, too flexible, or simply less suitable for the geometry of this scenario.</p>"
        )

    if primary_gap - comparison_gap > 0.08:
        parts.append(
            f"<p>{result.primary.model_name} shows the larger train-test gap, which suggests it is fitting the training data more aggressively than the comparison model.</p>"
        )
    elif comparison_gap - primary_gap > 0.08:
        parts.append(
            f"<p>{result.comparison.model_name} shows the larger train-test gap, so it appears to be the more variance-heavy option here.</p>"
        )
    else:
        parts.append(
            "<p>The train-test gaps are similar, so the main difference is probably model inductive bias rather than a dramatic overfitting gap.</p>"
        )

    if focus_parameter and focus_parameter in result.primary.hyperparameters:
        help_entry = get_hyperparameter_help(result.primary.model_name).get(focus_parameter)
        if help_entry:
            parts.append(
                f"<p><b>Focused parameter:</b> {focus_parameter}={result.primary.hyperparameters[focus_parameter]}. {help_entry['impact']}</p>"
            )

    parts.append(
        "<p><b>How to use this comparison:</b> change the active hyperparameters, keep the scenario fixed, and ask whether the performance gap comes from smoother generalization, extra flexibility, or a closer match to the underlying pattern.</p>"
    )
    return "".join(parts)


def run_interactive_example(
    *,
    task_type: str,
    model_name: str,
    scenario_name: str,
    hyperparameters: dict[str, Any],
    random_state: int = 42,
) -> InteractiveExampleResult:
    frame, description, feature_columns, X_train, X_test, y_train, y_test = _example_split(
        task_type=task_type,
        scenario_name=scenario_name,
        random_state=random_state,
    )
    return _fit_interactive_model(
        frame=frame,
        scenario_name=scenario_name,
        scenario_description=description,
        task_type=task_type,
        model_name=model_name,
        hyperparameters=hyperparameters,
        feature_columns=feature_columns,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        random_state=random_state,
    )


def run_interactive_comparison(
    *,
    task_type: str,
    primary_model_name: str,
    comparison_model_name: str,
    scenario_name: str,
    primary_hyperparameters: dict[str, Any],
    comparison_hyperparameters: dict[str, Any] | None = None,
    random_state: int = 42,
) -> InteractiveComparisonResult:
    frame, description, feature_columns, X_train, X_test, y_train, y_test = _example_split(
        task_type=task_type,
        scenario_name=scenario_name,
        random_state=random_state,
    )
    primary = _fit_interactive_model(
        frame=frame,
        scenario_name=scenario_name,
        scenario_description=description,
        task_type=task_type,
        model_name=primary_model_name,
        hyperparameters=primary_hyperparameters,
        feature_columns=feature_columns,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        random_state=random_state,
    )
    comparison = _fit_interactive_model(
        frame=frame,
        scenario_name=scenario_name,
        scenario_description=description,
        task_type=task_type,
        model_name=comparison_model_name,
        hyperparameters=comparison_hyperparameters or {},
        feature_columns=feature_columns,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        random_state=random_state,
    )
    return InteractiveComparisonResult(
        task_type=task_type,
        scenario_name=scenario_name,
        scenario_description=description,
        primary=primary,
        comparison=comparison,
    )
