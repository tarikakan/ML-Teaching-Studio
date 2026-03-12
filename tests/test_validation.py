from __future__ import annotations

import pytest

from ml_teaching_studio.core.datasets import load_builtin_dataset
from ml_teaching_studio.core.preprocessing import PreprocessingOptions
from ml_teaching_studio.core.trainers import TrainingRequest, train_and_evaluate
from ml_teaching_studio.core.validation import collect_training_warnings, validate_training_request, ValidationError


def test_collect_training_warnings_flags_scale_sensitive_model() -> None:
    bundle = load_builtin_dataset("Iris")
    request = TrainingRequest(
        dataset=bundle,
        task_type="classification",
        target_column=bundle.target_column,
        feature_columns=bundle.feature_columns,
        model_name="Logistic Regression",
        hyperparameters={},
        preprocessing=PreprocessingOptions(scale_numeric=False),
    )
    warnings = collect_training_warnings(request)
    assert any("sensitive to feature scale" in warning for warning in warnings)
    assert any("multiclass" in warning.lower() for warning in warnings)


def test_validate_training_request_rejects_too_many_pca_components() -> None:
    bundle = load_builtin_dataset("Iris")
    request = TrainingRequest(
        dataset=bundle,
        task_type="classification",
        target_column=bundle.target_column,
        feature_columns=bundle.feature_columns,
        model_name="Logistic Regression",
        hyperparameters={},
        preprocessing=PreprocessingOptions(use_pca=True, pca_components=10),
    )
    with pytest.raises(ValidationError):
        validate_training_request(request)


def test_training_result_includes_preflight_warnings() -> None:
    bundle = load_builtin_dataset("Iris")
    request = TrainingRequest(
        dataset=bundle,
        task_type="classification",
        target_column=bundle.target_column,
        feature_columns=bundle.feature_columns,
        model_name="Logistic Regression",
        hyperparameters={},
        preprocessing=PreprocessingOptions(scale_numeric=False),
    )
    result = train_and_evaluate(request)
    assert any("feature scale" in warning for warning in result.warnings)
