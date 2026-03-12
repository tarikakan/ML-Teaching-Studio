"""Metric calculation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import LabelBinarizer


def primary_metric_for_task(task_type: str) -> str:
    return "accuracy" if task_type == "classification" else "r2"


def metric_higher_is_better(metric_name: str) -> bool:
    return metric_name not in {"mae", "rmse", "mape", "log_loss"}


def compute_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_score: Any | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_score is not None:
        score_array = np.asarray(y_score)
        try:
            if score_array.ndim == 1:
                binary_true = LabelBinarizer().fit_transform(y_true).ravel()
                metrics["roc_auc"] = float(roc_auc_score(binary_true, score_array))
                metrics["average_precision"] = float(average_precision_score(binary_true, score_array))
            elif score_array.ndim == 2 and score_array.shape[1] == 2:
                binary_true = LabelBinarizer().fit_transform(y_true).ravel()
                metrics["roc_auc"] = float(roc_auc_score(binary_true, score_array[:, 1]))
                metrics["average_precision"] = float(average_precision_score(binary_true, score_array[:, 1]))
                metrics["log_loss"] = float(log_loss(y_true, score_array))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, score_array, multi_class="ovr", average="weighted")
                )
                metrics["log_loss"] = float(log_loss(y_true, score_array))
        except Exception:
            pass
    return metrics


def compute_regression_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_metrics(
    task_type: str,
    y_true: Any,
    y_pred: Any,
    y_score: Any | None = None,
) -> dict[str, float]:
    if task_type == "classification":
        return compute_classification_metrics(y_true, y_pred, y_score=y_score)
    return compute_regression_metrics(y_true, y_pred)


def confusion_matrix_data(y_true: Any, y_pred: Any) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)
