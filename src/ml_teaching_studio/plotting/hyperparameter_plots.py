"""Plots for hyperparameter sweep results."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from ml_teaching_studio.utils.config import COLOR_PALETTE


def _figure(title: str, size: tuple[int, int] = (8, 5)) -> tuple[Figure, any]:
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor("white")
    ax.set_title(title, fontsize=13, fontweight="bold", color=COLOR_PALETTE["text"])
    ax.grid(alpha=0.18)
    return fig, ax


def plot_validation_curve(sweep_result: any, param_name: str | None = None) -> Figure:
    frame = sweep_result.to_frame()
    selected_param = param_name or sweep_result.param_names[0]
    fig, ax = _figure(f"{sweep_result.model_name}: validation curve")
    if selected_param not in frame.columns:
        ax.text(0.5, 0.5, "No one-parameter sweep data is available for this view.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    frame = frame.sort_values(selected_param)
    ax.plot(frame[selected_param], frame["train_score_mean"], marker="o", label="Train score", color=COLOR_PALETTE["secondary"])
    ax.plot(frame[selected_param], frame["test_score_mean"], marker="o", label="Test score", color=COLOR_PALETTE["primary"])
    ax.set_xlabel(selected_param)
    ax.set_ylabel(sweep_result.metric_name)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_train_test_gap(sweep_result: any, param_name: str | None = None) -> Figure:
    frame = sweep_result.to_frame()
    selected_param = param_name or sweep_result.param_names[0]
    fig, ax = _figure(f"{sweep_result.model_name}: train-test gap")
    if selected_param not in frame.columns:
        ax.text(0.5, 0.5, "No matching sweep data is available for the selected parameter.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    frame = frame.sort_values(selected_param)
    ax.plot(frame[selected_param], frame["gap_mean"], marker="o", color=COLOR_PALETTE["danger"])
    ax.axhline(0.0, linestyle="--", color=COLOR_PALETTE["muted_text"])
    ax.set_xlabel(selected_param)
    ax.set_ylabel("Train score - test score")
    fig.tight_layout()
    return fig


def plot_heatmap(sweep_result: any, x_param: str | None = None, y_param: str | None = None) -> Figure:
    frame = sweep_result.to_frame()
    if len(sweep_result.param_names) < 2:
        fig, ax = _figure("Two-parameter heatmap unavailable")
        ax.text(0.5, 0.5, "Run a two-parameter sweep to generate a heatmap.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    x_name = x_param or sweep_result.param_names[0]
    y_name = y_param or sweep_result.param_names[1]
    pivot = frame.pivot(index=y_name, columns=x_name, values="test_score_mean")
    fig, ax = _figure(f"{sweep_result.model_name}: {x_name} vs {y_name}", size=(7, 6))
    image = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(value) for value in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(value) for value in pivot.index])
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    fig.colorbar(image, ax=ax, shrink=0.85, label=sweep_result.metric_name)
    fig.tight_layout()
    return fig


def sweep_table(sweep_result: any) -> pd.DataFrame:
    return sweep_result.to_frame()
