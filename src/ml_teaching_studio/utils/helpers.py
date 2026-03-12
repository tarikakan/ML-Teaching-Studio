"""Shared helper functions."""

from __future__ import annotations

import ast
import json
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def humanize_key(value: str) -> str:
    return value.replace("_", " ").strip().title()


def safe_json_dumps(payload: Any, indent: int = 2) -> str:
    return json.dumps(payload, indent=indent, ensure_ascii=True, default=json_default)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return value.tolist()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable.")


def parse_literal(value: str) -> Any:
    text = value.strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"none", "null"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def parse_value_list(text: str) -> list[Any]:
    return [parse_literal(chunk) for chunk in text.split(",") if chunk.strip()]


def format_metric(value: Any, decimals: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def infer_task_type(target: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(target):
        unique_count = target.nunique(dropna=True)
        if unique_count <= min(20, max(3, len(target) // 15)):
            return "classification"
        return "regression"
    return "classification"


def list_to_html(items: Iterable[str]) -> str:
    return "".join(f"<li>{item}</li>" for item in items)


def make_html_card(title: str, body: str, subtitle: str | None = None) -> str:
    subtitle_html = f"<p><i>{subtitle}</i></p>" if subtitle else ""
    return (
        f"<div style='padding:10px 2px;'>"
        f"<h2 style='margin-bottom:6px;'>{title}</h2>"
        f"{subtitle_html}"
        f"{body}"
        f"</div>"
    )


def top_n_mapping(mapping: dict[str, float], n: int = 5) -> list[tuple[str, float]]:
    return sorted(mapping.items(), key=lambda item: abs(item[1]), reverse=True)[:n]
