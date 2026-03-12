"""Input and output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import APP_DATA_DIR, EXPORT_DIR, RUN_STORE_PATH
from .helpers import json_default


def ensure_app_directories() -> None:
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    RUN_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: str | Path, default: Any | None = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return default if default is not None else {}
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Any) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, default=json_default)
    return file_path


def export_dataframe_csv(dataframe: pd.DataFrame, path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(file_path, index=False)
    return file_path


def save_text(path: str | Path, text: str) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(text, encoding="utf-8")
    return file_path
