"""Storage and retrieval for saved experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ml_teaching_studio.utils.config import RUN_STORE_PATH
from ml_teaching_studio.utils.io import load_json, save_json


@dataclass
class RunStore:
    storage_path: Path = RUN_STORE_PATH

    def __post_init__(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            save_json(self.storage_path, {"runs": [], "sweeps": []})

    def _load(self) -> dict[str, list[dict[str, Any]]]:
        return load_json(self.storage_path, default={"runs": [], "sweeps": []})

    def _save(self, payload: dict[str, list[dict[str, Any]]]) -> None:
        save_json(self.storage_path, payload)

    def save_run(self, record: dict[str, Any]) -> None:
        payload = self._load()
        payload.setdefault("runs", [])
        payload["runs"] = [item for item in payload["runs"] if item["run_id"] != record["run_id"]]
        payload["runs"].append(record)
        self._save(payload)

    def save_sweep(self, record: dict[str, Any]) -> None:
        payload = self._load()
        payload.setdefault("sweeps", [])
        payload["sweeps"].append(record)
        self._save(payload)

    def list_runs(self) -> list[dict[str, Any]]:
        payload = self._load()
        runs = payload.get("runs", [])
        return sorted(runs, key=lambda item: item.get("created_at", ""), reverse=True)

    def list_sweeps(self) -> list[dict[str, Any]]:
        payload = self._load()
        sweeps = payload.get("sweeps", [])
        return sorted(sweeps, key=lambda item: item.get("created_at", ""), reverse=True)

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        return next((item for item in self.list_runs() if item["run_id"] == run_id), None)

    def delete_run(self, run_id: str) -> None:
        payload = self._load()
        payload["runs"] = [item for item in payload.get("runs", []) if item["run_id"] != run_id]
        self._save(payload)

    def comparison_frame(self) -> pd.DataFrame:
        rows = []
        for record in self.list_runs():
            row = {
                "run_id": record["run_id"],
                "created_at": record.get("created_at"),
                "dataset_name": record.get("dataset_name"),
                "task_type": record.get("task_type"),
                "model_name": record.get("model_name"),
                "training_time": record.get("training_time"),
                "inference_time": record.get("inference_time"),
            }
            for metric_name, metric_value in record.get("test_metrics", {}).items():
                row[f"metric_{metric_name}"] = metric_value
            rows.append(row)
        return pd.DataFrame(rows)
