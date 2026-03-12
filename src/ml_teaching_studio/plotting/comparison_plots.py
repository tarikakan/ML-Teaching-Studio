"""Plots for comparing runs."""

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


def comparison_table(run_records: list[dict]) -> pd.DataFrame:
    rows = []
    for record in run_records:
        row = {
            "run_id": record.get("run_id"),
            "created_at": record.get("created_at"),
            "dataset_name": record.get("dataset_name"),
            "model_name": record.get("model_name"),
            "training_time": record.get("training_time"),
            "inference_time": record.get("inference_time"),
        }
        for metric_name, metric_value in record.get("test_metrics", {}).items():
            row[metric_name] = metric_value
        rows.append(row)
    return pd.DataFrame(rows)


def plot_metric_bars(run_records: list[dict], metric_name: str) -> Figure:
    frame = comparison_table(run_records)
    size = (9.5, max(4.5, 0.65 * max(len(frame.index), 1) + 1.8))
    fig, ax = _figure(f"Saved runs: {metric_name}", size=size)
    if frame.empty or metric_name not in frame.columns:
        ax.text(0.5, 0.5, "No comparable run data is available for this metric.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    frame = frame.sort_values(metric_name, ascending=True)
    labels = [f"{row.model_name}\n{row.dataset_name}" for row in frame.itertuples()]
    ax.barh(labels, frame[metric_name], color=COLOR_PALETTE["primary"])
    ax.set_xlabel(metric_name)
    fig.tight_layout()
    return fig


def plot_training_time(run_records: list[dict]) -> Figure:
    frame = comparison_table(run_records)
    size = (9.5, max(4.5, 0.65 * max(len(frame.index), 1) + 1.8))
    fig, ax = _figure("Saved runs: training time", size=size)
    if frame.empty:
        ax.text(0.5, 0.5, "No saved runs are available yet.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    labels = [f"{row.model_name}\n{row.dataset_name}" for row in frame.itertuples()]
    ax.barh(labels, frame["training_time"], color=COLOR_PALETTE["accent"])
    ax.set_xlabel("Seconds")
    fig.tight_layout()
    return fig


def plot_inference_time(run_records: list[dict]) -> Figure:
    frame = comparison_table(run_records)
    size = (9.5, max(4.5, 0.65 * max(len(frame.index), 1) + 1.8))
    fig, ax = _figure("Saved runs: inference time", size=size)
    if frame.empty:
        ax.text(0.5, 0.5, "No saved runs are available yet.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    labels = [f"{row.model_name}\n{row.dataset_name}" for row in frame.itertuples()]
    ax.barh(labels, frame["inference_time"], color=COLOR_PALETTE["secondary"])
    ax.set_xlabel("Seconds")
    fig.tight_layout()
    return fig


def plot_learning_curve_data(title: str, learning_curve_data: dict) -> Figure:
    fig, ax = _figure(title)
    if not learning_curve_data:
        ax.text(0.5, 0.5, "No learning-curve data is available for this run.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    ax.plot(
        learning_curve_data["train_sizes"],
        learning_curve_data["train_scores_mean"],
        marker="o",
        label="Train score",
        color=COLOR_PALETTE["secondary"],
    )
    ax.plot(
        learning_curve_data["train_sizes"],
        learning_curve_data["test_scores_mean"],
        marker="o",
        label="Test score",
        color=COLOR_PALETTE["primary"],
    )
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
