"""Factory for creating estimators from the model registry."""

from __future__ import annotations

from typing import Any

from ml_teaching_studio.models.model_registry import get_model_spec, list_model_names

from .validation import validate_hyperparameters


def create_model(
    model_name: str,
    hyperparameters: dict[str, Any] | None = None,
    *,
    random_state: int | None = None,
) -> Any:
    spec = get_model_spec(model_name)
    cleaned_params = validate_hyperparameters(model_name, hyperparameters or {})
    return spec.constructor(random_state=random_state, **cleaned_params)


def available_models(task_type: str | None = None) -> list[str]:
    return list_model_names(task_type)
