from __future__ import annotations

from ml_teaching_studio.core.datasets import load_builtin_dataset
from ml_teaching_studio.core.preprocessing import PreprocessingOptions
from ml_teaching_studio.core.trainers import TrainingRequest, train_and_evaluate


def test_training_pipeline_runs_for_classification() -> None:
    bundle = load_builtin_dataset("Iris")
    request = TrainingRequest(
        dataset=bundle,
        task_type="classification",
        target_column=bundle.target_column,
        feature_columns=bundle.feature_columns,
        model_name="Logistic Regression",
        hyperparameters={"C": 1.0, "penalty": "l2", "solver": "lbfgs"},
        preprocessing=PreprocessingOptions(),
    )
    result = train_and_evaluate(request)
    assert "accuracy" in result.test_metrics
    assert result.model_name == "Logistic Regression"


def test_training_pipeline_runs_for_regression() -> None:
    bundle = load_builtin_dataset("Synthetic Regression")
    request = TrainingRequest(
        dataset=bundle,
        task_type="regression",
        target_column=bundle.target_column,
        feature_columns=bundle.feature_columns,
        model_name="Linear Regression",
        hyperparameters={},
        preprocessing=PreprocessingOptions(),
    )
    result = train_and_evaluate(request)
    assert "r2" in result.test_metrics
