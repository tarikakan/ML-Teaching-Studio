"""Unified model training logic."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, StratifiedKFold, learning_curve, train_test_split

from ml_teaching_studio.utils.helpers import utc_timestamp

from .explainers import explain_training_outcome
from .metrics import compute_metrics, primary_metric_for_task
from .model_factory import create_model
from .preprocessing import PreprocessingOptions, build_training_pipeline, extract_feature_names
from .validation import (
    ValidationError,
    collect_training_warnings,
    humanize_training_exception,
    validate_training_request,
)


@dataclass
class TrainingRequest:
    dataset: Any
    task_type: str
    target_column: str
    feature_columns: list[str]
    model_name: str
    hyperparameters: dict[str, Any]
    preprocessing: PreprocessingOptions
    test_size: float = 0.2
    random_seed: int = 42
    notes: str = ""


@dataclass
class TrainingResult:
    run_id: str
    created_at: str
    dataset_name: str
    task_type: str
    target_column: str
    feature_columns: list[str]
    preprocessing: dict[str, Any]
    model_name: str
    hyperparameters: dict[str, Any]
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    training_time: float
    inference_time: float
    explanation: str
    classes_: list[Any] = field(default_factory=list)
    y_true: list[Any] = field(default_factory=list)
    y_pred: list[Any] = field(default_factory=list)
    y_score: list[Any] | None = None
    y_train_true: list[Any] = field(default_factory=list)
    y_train_pred: list[Any] = field(default_factory=list)
    feature_importances: dict[str, float] = field(default_factory=dict)
    coefficients: dict[str, float] = field(default_factory=dict)
    learning_curve_data: dict[str, list[float]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: str = ""
    pipeline: Any | None = field(default=None, repr=False)
    X_train: pd.DataFrame | None = field(default=None, repr=False)
    X_test: pd.DataFrame | None = field(default=None, repr=False)

    def to_record(self) -> dict[str, Any]:
        primary_metric = primary_metric_for_task(self.task_type)
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "preprocessing": self.preprocessing,
            "model_name": self.model_name,
            "hyperparameters": self.hyperparameters,
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "metric_primary": self.test_metrics.get(primary_metric),
            "training_time": self.training_time,
            "inference_time": self.inference_time,
            "explanation": self.explanation,
            "classes_": self.classes_,
            "y_true": self.y_true,
            "y_pred": self.y_pred,
            "y_score": self.y_score,
            "feature_importances": self.feature_importances,
            "coefficients": self.coefficients,
            "learning_curve_data": self.learning_curve_data,
            "warnings": self.warnings,
            "notes": self.notes,
            "generated_plots": available_plot_types(self.task_type),
        }


def _prediction_scores(pipeline: Any, features: pd.DataFrame) -> np.ndarray | None:
    model = pipeline.named_steps["model"]
    if hasattr(model, "predict_proba"):
        return pipeline.predict_proba(features)
    if hasattr(model, "decision_function"):
        return pipeline.decision_function(features)
    return None


def _feature_importances(pipeline: Any, feature_names: list[str]) -> dict[str, float]:
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = getattr(model, "feature_importances_")
        return {name: float(value) for name, value in zip(feature_names, values)}
    return {}


def _coefficients(pipeline: Any, feature_names: list[str]) -> dict[str, float]:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "coef_"):
        return {}
    coefficients = getattr(model, "coef_")
    if np.ndim(coefficients) == 2:
        coefficients = np.mean(np.abs(coefficients), axis=0)
    return {name: float(value) for name, value in zip(feature_names, coefficients)}


def _learning_curve_data(request: TrainingRequest, pipeline: Any, X: pd.DataFrame, y: pd.Series) -> dict[str, list[float]]:
    try:
        cv = (
            StratifiedKFold(n_splits=3, shuffle=True, random_state=request.random_seed)
            if request.task_type == "classification"
            else KFold(n_splits=3, shuffle=True, random_state=request.random_seed)
        )
        train_sizes = np.linspace(0.6, 1.0, 4) if request.task_type == "classification" else np.linspace(0.3, 1.0, 4)
        train_sizes, train_scores, test_scores = learning_curve(
            pipeline,
            X,
            y,
            cv=cv,
            train_sizes=train_sizes,
            scoring=primary_metric_for_task(request.task_type),
            n_jobs=1,
        )
        return {
            "train_sizes": train_sizes.tolist(),
            "train_scores_mean": np.mean(train_scores, axis=1).tolist(),
            "test_scores_mean": np.mean(test_scores, axis=1).tolist(),
        }
    except Exception:
        return {}


def available_plot_types(task_type: str) -> list[str]:
    if task_type == "classification":
        return [
            "confusion_matrix",
            "roc_curve",
            "precision_recall_curve",
            "decision_boundary",
            "feature_importance",
            "learning_curve",
        ]
    return [
        "prediction_vs_true",
        "residuals",
        "error_distribution",
        "feature_importance",
        "coefficient_plot",
        "learning_curve",
    ]


def _warning_messages(caught_warnings: list[warnings.WarningMessage]) -> list[str]:
    messages: list[str] = []
    for caught in caught_warnings:
        raw_message = str(caught.message).strip()
        lowered = raw_message.lower()
        if not raw_message:
            continue
        if "physical cores" in lowered:
            continue
        if issubclass(caught.category, ConvergenceWarning):
            messages.append(
                "The optimizer reached its iteration limit before full convergence. The result is still usable for learning, but increasing max_iter or enabling scaling may improve stability."
            )
            continue
        messages.append(raw_message)
    return list(dict.fromkeys(messages))


def train_and_evaluate(request: TrainingRequest) -> TrainingResult:
    validate_training_request(request)
    warning_notes = collect_training_warnings(request)

    try:
        dataframe = request.dataset.dataframe.copy()
        X = dataframe[request.feature_columns]
        y = dataframe[request.target_column]
        stratify = y if request.task_type == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=request.test_size,
            random_state=request.random_seed,
            stratify=stratify,
        )

        estimator = create_model(
            request.model_name,
            request.hyperparameters,
            random_state=request.random_seed,
        )
        pipeline = build_training_pipeline(
            dataframe,
            request.feature_columns,
            estimator,
            request.preprocessing,
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            training_started = perf_counter()
            pipeline.fit(X_train, y_train)
            training_time = perf_counter() - training_started

            inference_started = perf_counter()
            y_pred = pipeline.predict(X_test)
            inference_time = perf_counter() - inference_started
            y_train_pred = pipeline.predict(X_train)

            y_score = _prediction_scores(pipeline, X_test)
            y_train_score = _prediction_scores(pipeline, X_train)

            test_metrics = compute_metrics(request.task_type, y_test, y_pred, y_score=y_score)
            train_metrics = compute_metrics(request.task_type, y_train, y_train_pred, y_score=y_train_score)
            feature_names = extract_feature_names(pipeline, request.feature_columns, request.preprocessing)
            importances = _feature_importances(pipeline, feature_names)
            coefficients = _coefficients(pipeline, feature_names)
            learning_data = _learning_curve_data(request, pipeline, X, y)

        warning_notes.extend(_warning_messages(caught_warnings))

        explanation = explain_training_outcome(
            task_type=request.task_type,
            model_name=request.model_name,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            feature_importances=importances,
            coefficients=coefficients,
        )

        classes = []
        model = pipeline.named_steps["model"]
        if hasattr(model, "classes_"):
            classes = [str(label) for label in getattr(model, "classes_")]

        return TrainingResult(
            run_id=str(uuid4()),
            created_at=utc_timestamp(),
            dataset_name=request.dataset.name,
            task_type=request.task_type,
            target_column=request.target_column,
            feature_columns=request.feature_columns,
            preprocessing=request.preprocessing.to_dict(),
            model_name=request.model_name,
            hyperparameters=request.hyperparameters,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            training_time=training_time,
            inference_time=inference_time,
            explanation=explanation,
            classes_=classes,
            y_true=y_test.tolist(),
            y_pred=np.asarray(y_pred).tolist(),
            y_score=np.asarray(y_score).tolist() if y_score is not None else None,
            y_train_true=y_train.tolist(),
            y_train_pred=np.asarray(y_train_pred).tolist(),
            feature_importances=importances,
            coefficients=coefficients,
            learning_curve_data=learning_data,
            warnings=list(dict.fromkeys(warning_notes)),
            notes=request.notes,
            pipeline=pipeline,
            X_train=X_train.reset_index(drop=True),
            X_test=X_test.reset_index(drop=True),
        )
    except ValidationError:
        raise
    except Exception as exc:
        raise ValidationError(humanize_training_exception(exc)) from exc
