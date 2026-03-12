from __future__ import annotations

from ml_teaching_studio.core.metrics import compute_classification_metrics, compute_regression_metrics


def test_classification_metrics_include_accuracy() -> None:
    metrics = compute_classification_metrics(
        ["yes", "no", "yes", "no"],
        ["yes", "no", "no", "no"],
        y_score=[[0.1, 0.9], [0.8, 0.2], [0.6, 0.4], [0.7, 0.3]],
    )
    assert "accuracy" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_regression_metrics_include_r2_and_rmse() -> None:
    metrics = compute_regression_metrics([1.0, 2.0, 3.0], [1.1, 2.2, 2.8])
    assert "r2" in metrics
    assert "rmse" in metrics
