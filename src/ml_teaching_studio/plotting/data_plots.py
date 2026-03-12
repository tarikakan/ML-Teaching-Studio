"""Plots for dataset understanding."""

from __future__ import annotations

from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from ml_teaching_studio.utils.config import COLOR_PALETTE


def _figure(title: str, size: tuple[int, int] = (8, 5)) -> tuple[Figure, any]:
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor("white")
    ax.set_title(title, fontsize=13, fontweight="bold", color=COLOR_PALETTE["text"])
    ax.grid(alpha=0.18)
    return fig, ax


def plot_class_distribution(bundle: any) -> Figure:
    fig, ax = _figure(f"{bundle.name}: class distribution")
    counts = bundle.dataframe[bundle.target_column].value_counts()
    ax.bar(counts.index.astype(str), counts.values, color=COLOR_PALETTE["primary"])
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    fig.tight_layout()
    return fig


def plot_feature_histograms(bundle: any, max_features: int = 4) -> Figure:
    numeric_columns = bundle.numeric_columns[:max_features]
    if not numeric_columns:
        fig, ax = _figure("No numeric features available")
        ax.text(0.5, 0.5, "Histogram view requires numeric features.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    rows = ceil(len(numeric_columns) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(10, 3.2 * rows))
    axes = np.atleast_1d(axes).flatten()
    fig.patch.set_facecolor("white")
    for axis, column in zip(axes, numeric_columns):
        bundle.dataframe[column].hist(ax=axis, color=COLOR_PALETTE["secondary"], bins=20)
        axis.set_title(column)
        axis.grid(alpha=0.15)
    for axis in axes[len(numeric_columns):]:
        axis.axis("off")
    fig.suptitle(f"{bundle.name}: feature distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(bundle: any, max_features: int = 10) -> Figure:
    numeric_columns = bundle.numeric_columns[:max_features]
    fig, ax = _figure(f"{bundle.name}: correlation heatmap", size=(8, 6))
    if len(numeric_columns) < 2:
        ax.text(0.5, 0.5, "At least two numeric features are needed.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    corr = bundle.dataframe[numeric_columns].corr(numeric_only=True)
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(numeric_columns)))
    ax.set_xticklabels(numeric_columns, rotation=45, ha="right")
    ax.set_yticks(range(len(numeric_columns)))
    ax.set_yticklabels(numeric_columns)
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_missing_values(bundle: any) -> Figure:
    fig, ax = _figure(f"{bundle.name}: missing values")
    missing = bundle.dataframe.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values were found in this dataset.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    ax.barh(missing.index.astype(str), missing.values, color=COLOR_PALETTE["warning"])
    ax.set_xlabel("Missing values")
    fig.tight_layout()
    return fig


def plot_scatter_preview(bundle: any) -> Figure:
    numeric_columns = bundle.numeric_columns[:2]
    fig, ax = _figure(f"{bundle.name}: 2D preview")
    if len(numeric_columns) < 2:
        ax.text(0.5, 0.5, "At least two numeric features are needed for a scatter preview.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    x_col, y_col = numeric_columns
    if bundle.task_type == "classification":
        for label, group in bundle.dataframe.groupby(bundle.target_column):
            ax.scatter(group[x_col], group[y_col], label=str(label), alpha=0.7)
        ax.legend(frameon=False)
    else:
        scatter = ax.scatter(
            bundle.dataframe[x_col],
            bundle.dataframe[y_col],
            c=bundle.dataframe[bundle.target_column],
            cmap="viridis",
            alpha=0.75,
        )
        fig.colorbar(scatter, ax=ax, shrink=0.8, label=bundle.target_column)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()
    return fig


def plot_pca_projection(bundle: any) -> Figure:
    fig, ax = _figure(f"{bundle.name}: PCA projection")
    numeric = bundle.dataframe[bundle.numeric_columns]
    if numeric.shape[1] < 2:
        ax.text(0.5, 0.5, "PCA requires at least two numeric features.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    components = PCA(n_components=2).fit_transform(numeric.fillna(numeric.median()))
    projection = pd.DataFrame(components, columns=["PC1", "PC2"])
    if bundle.task_type == "classification":
        projection["target"] = bundle.dataframe[bundle.target_column].astype(str).values
        for label, group in projection.groupby("target"):
            ax.scatter(group["PC1"], group["PC2"], label=label, alpha=0.75)
        ax.legend(frameon=False)
    else:
        scatter = ax.scatter(
            projection["PC1"],
            projection["PC2"],
            c=bundle.dataframe[bundle.target_column],
            cmap="viridis",
            alpha=0.75,
        )
        fig.colorbar(scatter, ax=ax, shrink=0.8, label=bundle.target_column)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    return fig
