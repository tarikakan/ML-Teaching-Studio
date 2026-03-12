"""Dataset loading and summarization utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_iris,
    load_wine,
    make_classification,
    make_moons,
    make_regression,
)

from ml_teaching_studio.utils.helpers import infer_task_type


@dataclass
class DatasetBundle:
    name: str
    description: str
    dataframe: pd.DataFrame
    target_column: str
    task_type: str
    source: str
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    notes: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return self.dataframe.shape


@dataclass
class DatasetSummary:
    rows: int
    columns: int
    numeric_features: list[str]
    categorical_features: list[str]
    missing_values: dict[str, int]
    class_balance: dict[str, int]
    descriptive_statistics: pd.DataFrame

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "columns": self.columns,
            "numeric_features": self.numeric_features,
            "categorical_features": self.categorical_features,
            "missing_values": self.missing_values,
            "class_balance": self.class_balance,
            "descriptive_statistics": self.descriptive_statistics.to_dict(),
        }


BUILTIN_DATASETS = [
    "Iris",
    "Wine",
    "Breast Cancer",
    "California Housing",
    "Synthetic Classification",
    "Synthetic Regression",
    "Synthetic Noisy Classification",
    "Synthetic Noisy Regression",
]


def available_builtin_datasets() -> list[str]:
    return BUILTIN_DATASETS.copy()


def _bundle_from_frame(
    name: str,
    description: str,
    frame: pd.DataFrame,
    target_column: str,
    task_type: str,
    source: str,
    notes: list[str] | None = None,
) -> DatasetBundle:
    feature_columns = [column for column in frame.columns if column != target_column]
    numeric_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(frame[column])
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    return DatasetBundle(
        name=name,
        description=description,
        dataframe=frame,
        target_column=target_column,
        task_type=task_type,
        source=source,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        notes=notes or [],
    )


def _load_classification_bunch(name: str, description: str, loader: Any) -> DatasetBundle:
    bunch = loader(as_frame=True)
    frame = bunch.frame.copy()
    target_column = bunch.target.name if hasattr(bunch.target, "name") else "target"
    frame = frame.rename(columns={target_column: "target"})
    if hasattr(bunch, "target_names"):
        mapping = {index: label for index, label in enumerate(bunch.target_names)}
        frame["target"] = frame["target"].map(mapping).astype(str)
    return _bundle_from_frame(
        name=name,
        description=description,
        frame=frame,
        target_column="target",
        task_type="classification",
        source="scikit-learn",
    )


def _california_housing_bundle() -> DatasetBundle:
    try:
        bunch = fetch_california_housing(as_frame=True)
        frame = bunch.frame.copy()
        frame = frame.rename(columns={"MedHouseVal": "target"})
        return _bundle_from_frame(
            name="California Housing",
            description=(
                "Classic tabular regression dataset for housing-value prediction."
            ),
            frame=frame,
            target_column="target",
            task_type="regression",
            source="scikit-learn",
        )
    except Exception:
        rng = np.random.default_rng(42)
        features = pd.DataFrame(
            {
                "MedInc": rng.normal(4.0, 1.2, 1200),
                "HouseAge": rng.integers(1, 40, 1200),
                "AveRooms": rng.normal(5.0, 1.1, 1200),
                "AveBedrms": rng.normal(1.1, 0.2, 1200),
                "Population": rng.normal(1400, 500, 1200),
                "AveOccup": rng.normal(3.2, 0.8, 1200),
                "Latitude": rng.uniform(32.0, 42.0, 1200),
                "Longitude": rng.uniform(-124.0, -114.0, 1200),
            }
        )
        target = (
            0.7 * features["MedInc"]
            - 0.03 * features["HouseAge"]
            + 0.4 * features["AveRooms"]
            - 0.25 * features["AveBedrms"]
            + 0.0002 * features["Population"]
            + rng.normal(0.0, 0.8, 1200)
        )
        frame = features.copy()
        frame["target"] = target
        return _bundle_from_frame(
            name="California Housing",
            description=(
                "California Housing-style regression dataset. A synthetic fallback is being "
                "used because the original scikit-learn dataset was not available locally."
            ),
            frame=frame,
            target_column="target",
            task_type="regression",
            source="synthetic fallback",
            notes=[
                "The canonical California Housing dataset could not be fetched, so the app loaded a compatible offline surrogate."
            ],
        )


def _synthetic_classification(seed: int = 42) -> DatasetBundle:
    features, target = make_classification(
        n_samples=600,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        n_clusters_per_class=1,
        class_sep=1.4,
        random_state=seed,
    )
    frame = pd.DataFrame(features, columns=[f"feature_{idx}" for idx in range(1, 7)])
    frame["target"] = np.where(target == 1, "Class B", "Class A")
    return _bundle_from_frame(
        name="Synthetic Classification",
        description="Balanced synthetic classification dataset with a reasonably learnable boundary.",
        frame=frame,
        target_column="target",
        task_type="classification",
        source="generated",
    )


def _synthetic_regression(seed: int = 42) -> DatasetBundle:
    features, target = make_regression(
        n_samples=600,
        n_features=6,
        n_informative=5,
        noise=12.0,
        random_state=seed,
    )
    frame = pd.DataFrame(features, columns=[f"feature_{idx}" for idx in range(1, 7)])
    frame["target"] = target
    return _bundle_from_frame(
        name="Synthetic Regression",
        description="Synthetic regression dataset with moderate noise for baseline comparisons.",
        frame=frame,
        target_column="target",
        task_type="regression",
        source="generated",
    )


def _synthetic_noisy_classification(seed: int = 42) -> DatasetBundle:
    features, target = make_moons(n_samples=450, noise=0.32, random_state=seed)
    frame = pd.DataFrame(features, columns=["x1", "x2"])
    rng = np.random.default_rng(seed)
    frame["noise_feature"] = rng.normal(0, 1, len(frame))
    frame["target"] = np.where(target == 1, "Moon 2", "Moon 1")
    return _bundle_from_frame(
        name="Synthetic Noisy Classification",
        description="Curved classification problem with noise, useful for overfitting demonstrations.",
        frame=frame,
        target_column="target",
        task_type="classification",
        source="generated",
    )


def _synthetic_noisy_regression(seed: int = 42) -> DatasetBundle:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 6.0, 450)
    frame = pd.DataFrame({"x": x, "x_squared": x**2})
    target = np.sin(x) + 0.15 * x + rng.normal(0.0, 0.28, len(x))
    frame["target"] = target
    return _bundle_from_frame(
        name="Synthetic Noisy Regression",
        description="Noisy nonlinear regression dataset designed for underfitting and overfitting lessons.",
        frame=frame,
        target_column="target",
        task_type="regression",
        source="generated",
    )


def load_builtin_dataset(name: str, random_seed: int = 42) -> DatasetBundle:
    if name == "Iris":
        return _load_classification_bunch(
            "Iris",
            "Small flower classification dataset with three species.",
            load_iris,
        )
    if name == "Wine":
        return _load_classification_bunch(
            "Wine",
            "Chemical analysis features for classifying wine cultivars.",
            load_wine,
        )
    if name == "Breast Cancer":
        return _load_classification_bunch(
            "Breast Cancer",
            "Binary classification dataset for malignant versus benign cases.",
            load_breast_cancer,
        )
    if name == "California Housing":
        return _california_housing_bundle()
    if name == "Synthetic Classification":
        return _synthetic_classification(random_seed)
    if name == "Synthetic Regression":
        return _synthetic_regression(random_seed)
    if name == "Synthetic Noisy Classification":
        return _synthetic_noisy_classification(random_seed)
    if name == "Synthetic Noisy Regression":
        return _synthetic_noisy_regression(random_seed)
    raise ValueError(f"Unknown built-in dataset: {name}")


def load_csv_dataset(
    path: str | Path,
    *,
    target_column: str | None = None,
    task_type: str | None = None,
    name: str | None = None,
) -> DatasetBundle:
    file_path = Path(path)
    frame = pd.read_csv(file_path)
    if frame.empty:
        raise ValueError("The selected CSV file is empty.")
    chosen_target = target_column or frame.columns[-1]
    if chosen_target not in frame.columns:
        raise ValueError(f"Target column '{chosen_target}' is not present in the CSV file.")
    inferred_task = task_type or infer_task_type(frame[chosen_target])
    return _bundle_from_frame(
        name=name or file_path.stem,
        description=f"User-provided CSV dataset loaded from {file_path.name}.",
        frame=frame,
        target_column=chosen_target,
        task_type=inferred_task,
        source=str(file_path),
    )


def summarize_dataset(bundle: DatasetBundle) -> DatasetSummary:
    target = bundle.dataframe[bundle.target_column]
    class_balance: dict[str, int] = {}
    if bundle.task_type == "classification":
        class_balance = {str(key): int(value) for key, value in target.value_counts().items()}
    return DatasetSummary(
        rows=bundle.dataframe.shape[0],
        columns=bundle.dataframe.shape[1],
        numeric_features=bundle.numeric_columns,
        categorical_features=bundle.categorical_columns,
        missing_values={column: int(value) for column, value in bundle.dataframe.isna().sum().items()},
        class_balance=class_balance,
        descriptive_statistics=bundle.dataframe.describe(include="all").fillna(""),
    )


def descriptive_statistics(bundle: DatasetBundle) -> pd.DataFrame:
    return bundle.dataframe.describe(include="all").fillna("")


def dataset_preview(bundle: DatasetBundle, rows: int = 20) -> pd.DataFrame:
    return bundle.dataframe.head(rows).copy()


def target_overview(bundle: DatasetBundle) -> dict[str, Any]:
    target = bundle.dataframe[bundle.target_column]
    if bundle.task_type == "classification":
        counts = target.value_counts()
        return {
            "classes": int(counts.size),
            "balance": {str(key): int(value) for key, value in counts.items()},
        }
    return {
        "mean": float(target.mean()),
        "std": float(target.std()),
        "min": float(target.min()),
        "max": float(target.max()),
    }
