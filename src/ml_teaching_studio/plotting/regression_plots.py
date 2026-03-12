"""Regression plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ml_teaching_studio.utils.config import COLOR_PALETTE


def _figure(title: str, size: tuple[int, int] = (7, 5)) -> tuple[Figure, any]:
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor("white")
    ax.set_title(title, fontsize=13, fontweight="bold", color=COLOR_PALETTE["text"])
    ax.grid(alpha=0.18)
    return fig, ax


def plot_prediction_vs_true(result: any) -> Figure:
    fig, ax = _figure(f"{result.model_name}: predictions vs true")
    y_true = np.asarray(result.y_true)
    y_pred = np.asarray(result.y_pred)
    ax.scatter(y_true, y_pred, alpha=0.75, color=COLOR_PALETTE["primary"])
    low = min(y_true.min(), y_pred.min())
    high = max(y_true.max(), y_pred.max())
    ax.plot([low, high], [low, high], linestyle="--", color=COLOR_PALETTE["danger"])
    ax.set_xlabel("True target")
    ax.set_ylabel("Predicted target")
    fig.tight_layout()
    return fig


def plot_residuals(result: any) -> Figure:
    fig, ax = _figure(f"{result.model_name}: residuals")
    y_true = np.asarray(result.y_true)
    y_pred = np.asarray(result.y_pred)
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.75, color=COLOR_PALETTE["secondary"])
    ax.axhline(0.0, color=COLOR_PALETTE["danger"], linestyle="--")
    ax.set_xlabel("Predicted value")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    return fig


def plot_error_distribution(result: any) -> Figure:
    fig, ax = _figure(f"{result.model_name}: error distribution")
    y_true = np.asarray(result.y_true)
    y_pred = np.asarray(result.y_pred)
    errors = y_true - y_pred
    ax.hist(errors, bins=25, color=COLOR_PALETTE["warning"], alpha=0.9)
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def plot_coefficients(result: any) -> Figure:
    fig, ax = _figure(f"{result.model_name}: coefficients")
    values = result.coefficients or result.feature_importances
    if not values:
        ax.text(0.5, 0.5, "This model does not expose coefficients or feature importance.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    items = sorted(values.items(), key=lambda item: abs(item[1]), reverse=True)[:12]
    labels = [name for name, _ in items]
    scores = [score for _, score in items]
    colors = [COLOR_PALETTE["success"] if score >= 0 else COLOR_PALETTE["danger"] for score in scores]
    ax.barh(labels[::-1], scores[::-1], color=colors[::-1])
    ax.set_xlabel("Coefficient / importance")
    fig.tight_layout()
    return fig
