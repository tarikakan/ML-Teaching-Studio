from __future__ import annotations

from ml_teaching_studio.core.datasets import load_builtin_dataset
from ml_teaching_studio.core.hyperparameter_sweeps import SweepRequest, run_hyperparameter_sweep
from ml_teaching_studio.core.preprocessing import PreprocessingOptions


def test_hyperparameter_sweep_runs_for_knn() -> None:
    bundle = load_builtin_dataset("Iris")
    request = SweepRequest(
        dataset=bundle,
        task_type="classification",
        target_column=bundle.target_column,
        feature_columns=bundle.feature_columns,
        model_name="K-Nearest Neighbors",
        preprocessing=PreprocessingOptions(),
        base_hyperparameters={"weights": "uniform", "metric": "minkowski"},
        param_grid={"n_neighbors": [1, 3, 5]},
        metric_name="accuracy",
        repeat_seeds=[42],
    )
    result = run_hyperparameter_sweep(request)
    frame = result.to_frame()
    assert not frame.empty
    assert "test_score_mean" in frame.columns
