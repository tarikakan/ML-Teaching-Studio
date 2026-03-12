"""Hyperparameter sweep logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable
from uuid import uuid4

import pandas as pd

from ml_teaching_studio.utils.helpers import utc_timestamp

from .explainers import explain_sweep_outcome
from .metrics import metric_higher_is_better, primary_metric_for_task
from .preprocessing import PreprocessingOptions
from .trainers import TrainingRequest, train_and_evaluate


@dataclass
class SweepRequest:
    dataset: Any
    task_type: str
    target_column: str
    feature_columns: list[str]
    model_name: str
    preprocessing: PreprocessingOptions
    base_hyperparameters: dict[str, Any]
    param_grid: dict[str, list[Any]]
    metric_name: str | None = None
    test_size: float = 0.2
    random_seed: int = 42
    repeat_seeds: list[int] = field(default_factory=list)
    notes: str = ""


@dataclass
class SweepResult:
    sweep_id: str
    created_at: str
    dataset_name: str
    task_type: str
    model_name: str
    metric_name: str
    param_names: list[str]
    rows: list[dict[str, Any]]
    explanation: str
    notes: str = ""

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def to_record(self) -> dict[str, Any]:
        return {
            "sweep_id": self.sweep_id,
            "created_at": self.created_at,
            "dataset_name": self.dataset_name,
            "task_type": self.task_type,
            "model_name": self.model_name,
            "metric_name": self.metric_name,
            "param_names": self.param_names,
            "rows": self.rows,
            "explanation": self.explanation,
            "notes": self.notes,
        }


def run_hyperparameter_sweep(
    request: SweepRequest,
    progress_callback: Callable[[str, int], None] | None = None,
) -> SweepResult:
    metric_name = request.metric_name or primary_metric_for_task(request.task_type)
    seeds = request.repeat_seeds or [request.random_seed]
    param_names = list(request.param_grid.keys())
    combinations = list(product(*[request.param_grid[name] for name in param_names]))
    rows: list[dict[str, Any]] = []

    for index, combination in enumerate(combinations, start=1):
        current_params = dict(request.base_hyperparameters)
        current_params.update(dict(zip(param_names, combination)))
        train_scores: list[float] = []
        test_scores: list[float] = []
        for seed in seeds:
            training_request = TrainingRequest(
                dataset=request.dataset,
                task_type=request.task_type,
                target_column=request.target_column,
                feature_columns=request.feature_columns,
                model_name=request.model_name,
                hyperparameters=current_params,
                preprocessing=request.preprocessing,
                test_size=request.test_size,
                random_seed=seed,
            )
            result = train_and_evaluate(training_request)
            train_scores.append(result.train_metrics[metric_name])
            test_scores.append(result.test_metrics[metric_name])
        row = {name: value for name, value in zip(param_names, combination)}
        row["train_score_mean"] = float(pd.Series(train_scores).mean())
        row["test_score_mean"] = float(pd.Series(test_scores).mean())
        row["train_score_std"] = float(pd.Series(train_scores).std(ddof=0))
        row["test_score_std"] = float(pd.Series(test_scores).std(ddof=0))
        row["gap_mean"] = row["train_score_mean"] - row["test_score_mean"]
        rows.append(row)
        if progress_callback:
            progress_callback(
                f"Evaluated {index} of {len(combinations)} hyperparameter settings.",
                int(index / max(len(combinations), 1) * 100),
            )

    frame = pd.DataFrame(rows)
    ascending = not metric_higher_is_better(metric_name)
    frame = frame.sort_values("test_score_mean", ascending=ascending).reset_index(drop=True)
    explanation = explain_sweep_outcome(
        frame,
        model_name=request.model_name,
        metric_name=metric_name,
        task_type=request.task_type,
        param_names=param_names,
    )
    return SweepResult(
        sweep_id=str(uuid4()),
        created_at=utc_timestamp(),
        dataset_name=request.dataset.name,
        task_type=request.task_type,
        model_name=request.model_name,
        metric_name=metric_name,
        param_names=param_names,
        rows=frame.to_dict(orient="records"),
        explanation=explanation,
        notes=request.notes,
    )
