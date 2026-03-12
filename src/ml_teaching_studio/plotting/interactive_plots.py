"""Plots for interactive example demos."""

from __future__ import annotations

from textwrap import fill
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from ml_teaching_studio.core.interactive_examples import InteractiveComparisonResult
from ml_teaching_studio.utils.config import COLOR_PALETTE


def _wrapped_title(title: str, width: int = 18) -> str:
    return fill(title, width=width)


def _single_figure(title: str, size: tuple[int, int] = (7, 5)) -> tuple[Figure, Any]:
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor("white")
    ax.set_title(title, fontsize=13, fontweight="bold", color=COLOR_PALETTE["text"])
    ax.grid(alpha=0.18)
    return fig, ax


def _plot_classification_axis(
    ax: Any,
    result: Any,
    title: str,
    *,
    color_map: dict[str, Any] | None = None,
    include_legend_labels: bool = True,
    show_x_label: bool = True,
    show_y_label: bool = True,
    title_font_size: int = 12,
) -> None:
    ax.set_title(_wrapped_title(title), fontsize=title_font_size, fontweight="bold", color=COLOR_PALETTE["text"], pad=8)
    ax.grid(alpha=0.18)
    if result.grid_frame is None or result.grid_predictions is None:
        ax.text(0.5, 0.5, "No grid predictions are available for this example.", ha="center", va="center")
        ax.axis("off")
        return

    x1 = result.grid_frame["x1"].to_numpy()
    x2 = result.grid_frame["x2"].to_numpy()
    xx = x1.reshape(180, 180)
    yy = x2.reshape(180, 180)
    labels = sorted({str(value) for value in result.y_train + result.y_test})
    mapping = {label: index for index, label in enumerate(labels)}
    zz = np.vectorize(lambda item: mapping[str(item)])(result.grid_predictions).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=len(labels), cmap="coolwarm", alpha=0.22)

    if result.grid_scores is not None:
        scores = result.grid_scores.reshape(xx.shape)
        ax.contour(xx, yy, scores, levels=6, linewidths=0.8, cmap="viridis", alpha=0.45)

    train_frame = result.X_train.copy()
    train_frame["target"] = result.y_train
    test_frame = result.X_test.copy()
    test_frame["target"] = result.y_test
    if color_map is None:
        palette = plt.cm.get_cmap("tab10", max(len(labels), 2))
        color_map = {label: palette(index) for index, label in enumerate(labels)}
    for label in labels:
        train_group = train_frame[train_frame["target"].astype(str) == label]
        test_group = test_frame[test_frame["target"].astype(str) == label]
        ax.scatter(
            train_group["x1"],
            train_group["x2"],
            color=color_map[label],
            label=str(label) if include_legend_labels else None,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.35,
            s=28,
        )
        ax.scatter(
            test_group["x1"],
            test_group["x2"],
            color=color_map[label],
            label=None,
            marker="x",
            linewidth=1.0,
            s=36,
        )
    ax.set_xlabel("x1" if show_x_label else "")
    ax.set_ylabel("x2" if show_y_label else "")


def _plot_regression_axis(
    ax: Any,
    result: Any,
    title: str,
    *,
    show_x_label: bool = True,
    show_y_label: bool = True,
    title_font_size: int = 12,
) -> None:
    ax.set_title(_wrapped_title(title), fontsize=title_font_size, fontweight="bold", color=COLOR_PALETTE["text"], pad=8)
    ax.grid(alpha=0.18)
    train_frame = result.X_train.copy()
    train_frame["target"] = result.y_train
    test_frame = result.X_test.copy()
    test_frame["target"] = result.y_test

    ax.scatter(train_frame["x"], train_frame["target"], label="Train points", alpha=0.7, color=COLOR_PALETTE["secondary"])
    ax.scatter(test_frame["x"], test_frame["target"], label="Test points", alpha=0.7, color=COLOR_PALETTE["accent"])

    if result.curve_frame is not None and result.curve_predictions is not None:
        order = np.argsort(result.curve_frame["x"].to_numpy())
        x_sorted = result.curve_frame["x"].to_numpy()[order]
        y_sorted = result.curve_predictions[order]
        ax.plot(x_sorted, y_sorted, color=COLOR_PALETTE["primary"], linewidth=2.5, label="Model curve")

    ax.set_xlabel("x" if show_x_label else "")
    ax.set_ylabel("target" if show_y_label else "")


def plot_interactive_classification(result: Any, *, size: tuple[float, float] = (7, 6)) -> Figure:
    fig, ax = _single_figure(f"{result.model_name}: interactive boundary", size=size)
    _plot_classification_axis(ax, result, f"{result.model_name}: decision shape", title_font_size=11)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def plot_interactive_regression(result: Any, *, size: tuple[float, float] = (7, 5)) -> Figure:
    fig, ax = _single_figure(f"{result.model_name}: interactive regression", size=size)
    _plot_regression_axis(ax, result, f"{result.model_name}: fitted curve", title_font_size=11)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


def plot_interactive_comparison(
    result: InteractiveComparisonResult,
    *,
    size_scale: float = 1.0,
) -> Figure:
    if result.task_type == "classification":
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(13.2 * size_scale, 6.4 * size_scale),
            sharex=True,
            sharey=True,
        )
        fig.patch.set_facecolor("white")
        labels = sorted({str(value) for value in result.primary.y_train + result.primary.y_test})
        palette = plt.cm.get_cmap("tab10", max(len(labels), 2))
        color_map = {label: palette(index) for index, label in enumerate(labels)}
        _plot_classification_axis(
            axes[0],
            result.primary,
            result.primary.model_name,
            color_map=color_map,
            include_legend_labels=False,
            show_x_label=False,
            show_y_label=False,
            title_font_size=10,
        )
        _plot_classification_axis(
            axes[1],
            result.comparison,
            result.comparison.model_name,
            color_map=color_map,
            include_legend_labels=False,
            show_x_label=False,
            show_y_label=False,
            title_font_size=10,
        )
        legend_handles = [
            Line2D([0], [0], marker="o", linestyle="", color=color_map[label], markersize=6, label=label)
            for label in labels
        ]
        fig.legend(
            legend_handles,
            labels,
            frameon=False,
            ncol=min(4, max(1, len(labels))),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.925),
            fontsize=8,
        )
        fig.suptitle("Model Comparison", fontsize=12, fontweight="bold", color=COLOR_PALETTE["text"], y=0.975)
        fig.supxlabel("x1")
        fig.supylabel("x2")
        fig.text(
            0.5,
            0.03,
            "Marker key: filled circles = train points, X markers = test points",
            ha="center",
            fontsize=8,
            color=COLOR_PALETTE["muted_text"],
        )
        fig.tight_layout(rect=(0.035, 0.085, 0.98, 0.84))
        return fig

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.0 * size_scale, 6.0 * size_scale),
        sharex=True,
        sharey=True,
    )
    fig.patch.set_facecolor("white")
    _plot_regression_axis(
        axes[0],
        result.primary,
        result.primary.model_name,
        show_x_label=False,
        show_y_label=False,
        title_font_size=10,
    )
    _plot_regression_axis(
        axes[1],
        result.comparison,
        result.comparison.model_name,
        show_x_label=False,
        show_y_label=False,
        title_font_size=10,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=3, fontsize=8)
    fig.suptitle("Model Comparison", fontsize=12, fontweight="bold", color=COLOR_PALETTE["text"], y=0.975)
    fig.supxlabel("x")
    fig.supylabel("target")
    fig.tight_layout(rect=(0.035, 0.065, 0.98, 0.855))
    return fig
