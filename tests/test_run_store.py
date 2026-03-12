from __future__ import annotations

from pathlib import Path

from ml_teaching_studio.core.run_store import RunStore


def test_run_store_saves_and_reads_runs(tmp_path: Path) -> None:
    store = RunStore(storage_path=tmp_path / "runs.json")
    record = {
        "run_id": "abc",
        "created_at": "2026-03-10T12:00:00+00:00",
        "dataset_name": "Iris",
        "task_type": "classification",
        "model_name": "Logistic Regression",
        "training_time": 0.1,
        "inference_time": 0.01,
        "test_metrics": {"accuracy": 0.95},
    }
    store.save_run(record)
    loaded = store.get_run("abc")
    assert loaded is not None
    assert loaded["model_name"] == "Logistic Regression"
