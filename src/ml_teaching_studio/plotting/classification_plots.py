"""Classification plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer

from ml_teaching_studio.utils.config import COLOR_PALETTE


def _figure(title: str, size: tuple[int, int] = (7, 5)) -> tuple[Figure, any]:
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_facecolor("white")
    ax.set_title(title, fontsize=13, fontweight="bold", color=COLOR_PALETTE["text"])
    ax.grid(alpha=0.18)
    return fig, ax


def plot_confusion_matrix(result: any) -> Figure:
    fig, ax = _figure(f"{result.model_name}: confusion matrix")
    ConfusionMatrixDisplay.from_predictions(
        result.y_true,
        result.y_pred,
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )
    fig.tight_layout()
    return fig


def plot_roc_curve(result: any) -> Figure:
    fig, ax = _figure(f"{result.model_name}: ROC curve")
    if result.y_score is None:
        ax.text(0.5, 0.5, "ROC curve requires probability or score outputs.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    y_score = np.asarray(result.y_score)
    if y_score.ndim == 1:
        binary_true = LabelBinarizer().fit(result.y_true).transform(result.y_true).ravel()
        RocCurveDisplay.from_predictions(binary_true, y_score, ax=ax)
    elif y_score.ndim == 2 and y_score.shape[1] == 2:
        binary_true = LabelBinarizer().fit(result.y_true).transform(result.y_true).ravel()
        RocCurveDisplay.from_predictions(binary_true, y_score[:, 1], ax=ax)
    else:
        lb = LabelBinarizer().fit(result.y_true)
        y_true = lb.transform(result.y_true)
        for index, label in enumerate(lb.classes_):
            RocCurveDisplay.from_predictions(
                y_true[:, index],
                y_score[:, index],
                ax=ax,
                name=str(label),
            )
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_precision_recall_curve(result: any) -> Figure:
    fig, ax = _figure(f"{result.model_name}: precision-recall curve")
    if result.y_score is None:
        ax.text(0.5, 0.5, "Precision-recall curve requires probability or score outputs.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    y_score = np.asarray(result.y_score)
    if y_score.ndim == 1:
        binary_true = LabelBinarizer().fit(result.y_true).transform(result.y_true).ravel()
        PrecisionRecallDisplay.from_predictions(binary_true, y_score, ax=ax)
    elif y_score.ndim == 2 and y_score.shape[1] == 2:
        binary_true = LabelBinarizer().fit(result.y_true).transform(result.y_true).ravel()
        PrecisionRecallDisplay.from_predictions(binary_true, y_score[:, 1], ax=ax)
    else:
        lb = LabelBinarizer().fit(result.y_true)
        y_true = lb.transform(result.y_true)
        for index, label in enumerate(lb.classes_):
            PrecisionRecallDisplay.from_predictions(
                y_true[:, index],
                y_score[:, index],
                ax=ax,
                name=str(label),
            )
        ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_decision_boundary(result: any) -> Figure:
    fig, ax = _figure(f"{result.model_name}: decision boundary", size=(7, 6))
    if result.X_train is None or result.X_train.shape[1] < 2:
        ax.text(0.5, 0.5, "Decision boundary view requires at least two input features.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    numeric_columns = [
        column for column in result.X_train.columns if pd.api.types.is_numeric_dtype(result.X_train[column])
    ]
    if len(numeric_columns) < 2:
        ax.text(0.5, 0.5, "Decision boundary view requires two numeric features.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig

    x_col, y_col = numeric_columns[:2]
    x_min, x_max = result.X_train[x_col].min() - 0.5, result.X_train[x_col].max() + 0.5
    y_min, y_max = result.X_train[y_col].min() - 0.5, result.X_train[y_col].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 160), np.linspace(y_min, y_max, 160))
    defaults = {}
    for column in result.X_train.columns:
        series = result.X_train[column]
        defaults[column] = series.median() if pd.api.types.is_numeric_dtype(series) else series.mode().iloc[0]
    grid = pd.DataFrame(defaults, index=range(xx.size))
    grid[x_col] = xx.ravel()
    grid[y_col] = yy.ravel()
    predictions = result.pipeline.predict(grid).reshape(xx.shape)
    unique_labels = sorted({str(label) for label in result.y_train_true})
    mapping = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_predictions = np.vectorize(lambda item: mapping[str(item)])(predictions)
    ax.contourf(xx, yy, numeric_predictions, alpha=0.28, cmap="coolwarm")
    train_frame = result.X_train.copy()
    train_frame["target"] = result.y_train_true
    for label, group in train_frame.groupby("target"):
        ax.scatter(group[x_col], group[y_col], label=str(label), edgecolor="black", linewidth=0.3, alpha=0.75)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_probability_surface(result: any) -> Figure:
    fig, ax = _figure(f"{result.model_name}: probability surface", size=(7, 6))
    if result.X_train is None or result.X_train.shape[1] < 2 or result.y_score is None:
        ax.text(0.5, 0.5, "Probability surface requires a probabilistic classifier and two features.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    model = result.pipeline.named_steps["model"]
    if not hasattr(model, "predict_proba"):
        ax.text(0.5, 0.5, "The selected model does not expose class probabilities.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    numeric_columns = [
        column for column in result.X_train.columns if pd.api.types.is_numeric_dtype(result.X_train[column])
    ]
    if len(numeric_columns) < 2:
        ax.text(0.5, 0.5, "Probability surface requires two numeric features.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    x_col, y_col = numeric_columns[:2]
    x_min, x_max = result.X_train[x_col].min() - 0.5, result.X_train[x_col].max() + 0.5
    y_min, y_max = result.X_train[y_col].min() - 0.5, result.X_train[y_col].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 160), np.linspace(y_min, y_max, 160))
    defaults = {}
    for column in result.X_train.columns:
        series = result.X_train[column]
        defaults[column] = series.median() if pd.api.types.is_numeric_dtype(series) else series.mode().iloc[0]
    grid = pd.DataFrame(defaults, index=range(xx.size))
    grid[x_col] = xx.ravel()
    grid[y_col] = yy.ravel()
    probabilities = result.pipeline.predict_proba(grid)
    positive_surface = probabilities[:, min(1, probabilities.shape[1] - 1)].reshape(xx.shape)
    surface = ax.contourf(xx, yy, positive_surface, levels=12, cmap="viridis", alpha=0.75)
    fig.colorbar(surface, ax=ax, shrink=0.8, label="Predicted probability")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    fig.tight_layout()
    return fig


def plot_feature_importance(result: any) -> Figure:
    fig, ax = _figure(f"{result.model_name}: feature importance")
    values = result.feature_importances or result.coefficients
    if not values:
        ax.text(0.5, 0.5, "This model does not expose feature importance or coefficients.", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        return fig
    items = sorted(values.items(), key=lambda item: abs(item[1]), reverse=True)[:10]
    labels = [name for name, _ in items]
    scores = [score for _, score in items]
    ax.barh(labels[::-1], scores[::-1], color=COLOR_PALETTE["accent"])
    ax.set_xlabel("Importance / coefficient magnitude")
    fig.tight_layout()
    return fig
