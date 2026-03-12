from __future__ import annotations

from ml_teaching_studio.core.interactive_examples import (
    interactive_metric_frame,
    run_interactive_comparison,
    run_interactive_example,
)


def test_interactive_classification_example_generates_boundary_data() -> None:
    result = run_interactive_example(
        task_type="classification",
        model_name="K-Nearest Neighbors",
        scenario_name="Curved Moons",
        hyperparameters={"n_neighbors": 5, "weights": "uniform", "metric": "minkowski"},
    )
    assert result.grid_frame is not None
    assert result.grid_predictions is not None
    assert "accuracy" in result.test_metrics


def test_interactive_regression_example_generates_curve() -> None:
    result = run_interactive_example(
        task_type="regression",
        model_name="Ridge Regression",
        scenario_name="Smooth Curve",
        hyperparameters={"alpha": 1.0},
    )
    assert result.curve_frame is not None
    assert result.curve_predictions is not None
    assert "r2" in result.test_metrics


def test_interactive_comparison_uses_same_scenario_for_two_models() -> None:
    comparison = run_interactive_comparison(
        task_type="classification",
        primary_model_name="K-Nearest Neighbors",
        comparison_model_name="Logistic Regression",
        scenario_name="Curved Moons",
        primary_hyperparameters={"n_neighbors": 3, "weights": "uniform", "metric": "minkowski"},
    )
    assert comparison.primary.scenario_name == comparison.comparison.scenario_name
    assert comparison.primary.X_test is not None
    assert comparison.comparison.X_test is not None
    assert comparison.primary.X_test.equals(comparison.comparison.X_test)
    frame = interactive_metric_frame(comparison)
    assert frame.shape[0] == 2
