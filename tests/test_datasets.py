from __future__ import annotations

import pandas as pd

from ml_teaching_studio.core.datasets import load_builtin_dataset, load_csv_dataset, summarize_dataset


def test_builtin_iris_dataset_loads() -> None:
    bundle = load_builtin_dataset("Iris")
    assert bundle.task_type == "classification"
    assert bundle.target_column == "target"
    assert len(bundle.feature_columns) == 4


def test_csv_dataset_loads_with_target(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4],
            "feature_b": [10, 11, 12, 13],
            "label": ["low", "low", "high", "high"],
        }
    )
    path = tmp_path / "sample.csv"
    frame.to_csv(path, index=False)
    bundle = load_csv_dataset(path, target_column="label", task_type="classification")
    summary = summarize_dataset(bundle)
    assert summary.rows == 4
    assert summary.class_balance["low"] == 2
