"""Validation helpers for user inputs."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from ml_teaching_studio.models.model_registry import get_model_spec
from ml_teaching_studio.utils.helpers import parse_literal


class ValidationError(ValueError):
    """Raised when user input is not valid for training or sweeps."""


SCALE_SENSITIVE_MODELS = {
    "K-Nearest Neighbors",
    "Logistic Regression",
    "Support Vector Machine",
    "Support Vector Regressor",
    "MLP Classifier",
    "MLP Regressor",
    "Ridge Regression",
    "Lasso Regression",
}


def coerce_hyperparameter_value(name: str, value: Any) -> Any:
    if name == "max_features" and value == "1.0":
        return 1.0
    if isinstance(value, str):
        parsed = parse_literal(value)
        return parsed if parsed is not None else value
    return value


def validate_hyperparameters(model_name: str, hyperparameters: dict[str, Any]) -> dict[str, Any]:
    spec = get_model_spec(model_name)
    cleaned: dict[str, Any] = {}
    for key, value in hyperparameters.items():
        if key not in spec.hyperparameters:
            continue
        field_spec = spec.hyperparameters[key]
        coerced = coerce_hyperparameter_value(key, value)
        if field_spec.param_type == "choice" and str(coerced) not in {
            str(choice) for choice in field_spec.choices
        }:
            raise ValidationError(
                f"'{coerced}' is not a valid value for {key}. Allowed values: {field_spec.choices}"
            )
        cleaned[key] = coerced
    return cleaned


def validate_feature_selection(feature_columns: list[str]) -> None:
    if not feature_columns:
        raise ValidationError("Select at least one feature column before training.")


def validate_training_request(request: Any) -> None:
    if request.dataset is None:
        raise ValidationError("Load a dataset before starting training.")
    if not request.model_name:
        raise ValidationError("Choose a model before training.")
    if request.target_column not in request.dataset.dataframe.columns:
        raise ValidationError("The chosen target column does not exist in the dataset.")
    missing_features = [column for column in request.feature_columns if column not in request.dataset.dataframe.columns]
    if missing_features:
        raise ValidationError("Some selected features are no longer present in the active dataset.")
    validate_feature_selection(request.feature_columns)
    if request.target_column in request.feature_columns:
        raise ValidationError("The target column cannot also be used as a feature.")
    has_categorical = any(
        not request.dataset.dataframe[column].dtype.kind in {"i", "u", "f", "b"}
        for column in request.feature_columns
    )
    if has_categorical and not request.preprocessing.encode_categorical:
        raise ValidationError(
            "Categorical features are selected, so one-hot encoding should stay enabled."
        )
    if not 0.05 <= float(request.test_size) <= 0.5:
        raise ValidationError("Train/test split must keep between 5% and 50% of rows for testing.")
    model_spec = get_model_spec(request.model_name)
    if model_spec.task_type != request.task_type:
        raise ValidationError(
            f"{request.model_name} is a {model_spec.task_type} model, not a {request.task_type} model."
        )
    target = request.dataset.dataframe[request.target_column]
    if request.task_type == "regression" and not pd.api.types.is_numeric_dtype(target):
        raise ValidationError("Regression requires a numeric target column.")
    if request.task_type == "classification":
        class_counts = target.astype(str).value_counts()
        if class_counts.empty or len(class_counts) < 2:
            raise ValidationError("Classification requires at least two target classes.")
        if int(class_counts.min()) < 2:
            raise ValidationError(
                "At least one class has fewer than two examples, so a stratified train/test split is not possible."
            )
        n_rows = len(target)
        n_test = math.ceil(n_rows * float(request.test_size))
        if n_test < len(class_counts):
            raise ValidationError(
                "The chosen test split is too small for the number of classes. Increase test size or reduce the class count."
            )
    if request.preprocessing.use_pca and request.preprocessing.pca_components:
        numeric_feature_count = sum(
            1 for column in request.feature_columns if pd.api.types.is_numeric_dtype(request.dataset.dataframe[column])
        )
        categorical_feature_count = len(request.feature_columns) - numeric_feature_count
        effective_feature_count = numeric_feature_count
        if request.preprocessing.encode_categorical:
            effective_feature_count += categorical_feature_count
        if effective_feature_count < 2:
            raise ValidationError("PCA requires at least two usable feature dimensions after preprocessing.")
        if request.preprocessing.pca_components > effective_feature_count:
            raise ValidationError(
                f"PCA components ({request.preprocessing.pca_components}) exceed the usable feature count after preprocessing ({effective_feature_count})."
            )


def collect_training_warnings(request: Any) -> list[str]:
    warnings: list[str] = []
    frame = request.dataset.dataframe
    numeric_columns = [
        column for column in request.feature_columns if pd.api.types.is_numeric_dtype(frame[column])
    ]
    target = frame[request.target_column]

    if request.model_name in SCALE_SENSITIVE_MODELS and numeric_columns and not request.preprocessing.scale_numeric:
        warnings.append(
            f"{request.model_name} is sensitive to feature scale. Turning scaling off may make the model depend more on units than on true signal."
        )

    if request.task_type == "classification":
        class_counts = target.astype(str).value_counts()
        if len(class_counts) > 2:
            warnings.append(
                "This is a multiclass classification problem. ROC and precision-recall plots will be shown in one-vs-rest form, so read them class by class."
            )
        if not class_counts.empty and int(class_counts.min()) < 5:
            warnings.append(
                "At least one class has very few examples. Train/test metrics may move noticeably with different random seeds."
            )
    else:
        if target.nunique(dropna=True) < 20:
            warnings.append(
                "The regression target has relatively few unique values. Double-check that regression is really the intended task type."
            )

    if request.preprocessing.use_pca:
        warnings.append(
            "PCA changes the meaning of the features. Coefficient and feature-importance views will describe principal components instead of the original columns."
        )
        if numeric_columns and not request.preprocessing.scale_numeric:
            warnings.append(
                "PCA is enabled without scaling. Large-scale features can dominate the components and make the projection less informative."
            )

    if len(numeric_columns) < 2:
        warnings.append(
            "Some behavior plots need at least two numeric features. Decision-boundary and 2D geometric views may be unavailable or simplified."
        )

    return warnings


def humanize_training_exception(exc: Exception) -> str:
    message = str(exc).strip()
    lowered = message.lower()

    if "n_components" in lowered and "pca" in lowered:
        return (
            "PCA could not be applied with the current settings. Reduce the number of PCA components "
            "or choose more numeric features."
        )
    if "n_components=" in lowered and "must be between" in lowered:
        return (
            "The chosen PCA component count is too high for this dataset. Reduce the component count and try again."
        )
    if "n_neighbors" in lowered and "n_samples_fit" in lowered:
        return (
            "K-Nearest Neighbors needs more training samples than the current n_neighbors value. "
            "Lower n_neighbors or use a larger training split."
        )
    if "at least 2 classes" in lowered:
        return "The selected training split ended up with only one class. Use a larger dataset or adjust the split."
    if "could not convert string to float" in lowered:
        return (
            "The selected features still contain non-numeric values after preprocessing. "
            "Enable categorical encoding or remove the text columns from the feature set."
        )
    if "test_size" in lowered and "number of classes" in lowered:
        return (
            "The test split is too small for the number of classes. Increase test size so each class can appear in the test set."
        )
    return message or exc.__class__.__name__
