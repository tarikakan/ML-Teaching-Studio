"""Human-readable explanations of metrics and model behavior."""

from __future__ import annotations

from typing import Any

import pandas as pd

from ml_teaching_studio.utils.helpers import format_metric, top_n_mapping


def explain_metric(metric_name: str, value: float, task_type: str) -> str:
    readable = format_metric(value)
    if task_type == "classification":
        messages = {
            "accuracy": f"Accuracy of {readable} means this fraction of test predictions were correct.",
            "precision_weighted": f"Weighted precision of {readable} summarizes how reliable positive predictions were across classes.",
            "recall_weighted": f"Weighted recall of {readable} summarizes how many true cases were recovered across classes.",
            "f1_weighted": f"Weighted F1 of {readable} balances precision and recall across classes.",
            "roc_auc": f"ROC AUC of {readable} suggests how well the model ranks positives ahead of negatives across thresholds.",
            "average_precision": f"Average precision of {readable} focuses on ranking quality for the positive class.",
            "log_loss": f"Log loss of {readable} reflects how well calibrated and confident the predicted probabilities were.",
        }
    else:
        messages = {
            "mae": f"MAE of {readable} is the average absolute prediction error.",
            "rmse": f"RMSE of {readable} penalizes large mistakes more strongly than MAE.",
            "mape": f"MAPE of {readable} measures relative error size as a percentage-like quantity.",
            "r2": f"R² of {readable} estimates how much target variance the model explained.",
        }
    return messages.get(metric_name, f"{metric_name} = {readable}")


def explain_training_outcome(
    *,
    task_type: str,
    model_name: str,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    feature_importances: dict[str, float] | None = None,
    coefficients: dict[str, float] | None = None,
) -> str:
    parts: list[str] = [f"<h3>{model_name} interpretation</h3>"]
    primary = "accuracy" if task_type == "classification" else "r2"
    train_primary = train_metrics.get(primary)
    test_primary = test_metrics.get(primary)
    if train_primary is not None and test_primary is not None:
        gap = train_primary - test_primary
        if task_type == "classification":
            if gap > 0.08:
                parts.append(
                    "<p>Training performance is noticeably higher than test performance. That pattern suggests overfitting: the model learned details that did not generalize well.</p>"
                )
            elif train_primary < 0.7 and test_primary < 0.7:
                parts.append(
                    "<p>Both training and test accuracy are modest, which suggests underfitting or weak signal in the current features.</p>"
                )
            else:
                parts.append(
                    "<p>Training and test accuracy are reasonably aligned, so the current complexity looks fairly balanced.</p>"
                )
        else:
            if gap > 0.15:
                parts.append(
                    "<p>The training R² is much higher than the test R². This is a common sign of overfitting in regression.</p>"
                )
            elif train_primary < 0.4 and test_primary < 0.4:
                parts.append(
                    "<p>Both training and test R² are low. The model may be too simple, or the features may not contain enough predictive signal.</p>"
                )
            else:
                parts.append(
                    "<p>The gap between training and test R² is modest, so the model appears to generalize reasonably well.</p>"
                )

    if feature_importances:
        top_features = top_n_mapping(feature_importances, n=3)
        if top_features:
            summary = ", ".join(
                f"{name} ({value:.3f})" for name, value in top_features
            )
            parts.append(
                f"<p>The model relied most on these transformed features: <b>{summary}</b>. Use that as a cue for further feature inspection, not as proof of causality.</p>"
            )
    elif coefficients:
        top_features = top_n_mapping(coefficients, n=3)
        if top_features:
            summary = ", ".join(
                f"{name} ({value:.3f})" for name, value in top_features
            )
            parts.append(
                f"<p>The largest coefficients were <b>{summary}</b>. Coefficients are easiest to interpret when numeric features are scaled consistently.</p>"
            )

    helpful_metric = next(iter(test_metrics.items()))
    parts.append(
        f"<p>{explain_metric(helpful_metric[0], helpful_metric[1], task_type)}</p>"
    )
    return "".join(parts)


def explain_sweep_outcome(
    sweep_frame: pd.DataFrame,
    *,
    model_name: str,
    metric_name: str,
    task_type: str,
    param_names: list[str],
) -> str:
    if sweep_frame.empty:
        return "<p>No sweep results are available yet.</p>"

    score_column = "test_score_mean"
    best_row = sweep_frame.sort_values(score_column, ascending=False).iloc[0]
    parts = [f"<h3>{model_name} sweep interpretation</h3>"]
    params_text = ", ".join(f"{name}={best_row[name]}" for name in param_names)
    parts.append(
        f"<p>The strongest observed test {metric_name} occurred at <b>{params_text}</b>.</p>"
    )

    if len(param_names) == 1:
        series = sweep_frame.sort_values(param_names[0])
        best_index = series.index.get_loc(series[score_column].idxmax())
        if 0 < best_index < len(series) - 1:
            parts.append(
                "<p>The best setting appears in the interior of the tested range. That usually indicates a useful bias-variance balance rather than “bigger is always better.”</p>"
            )
        else:
            parts.append(
                "<p>The best setting appeared at an edge of the tested range. That usually means a wider sweep is worth trying.</p>"
            )
    if "gap_mean" in sweep_frame.columns:
        average_gap = float(sweep_frame["gap_mean"].mean())
        if average_gap > 0.1:
            parts.append(
                "<p>Many sweep points show a noticeable train-test gap. That suggests the model family can overfit under some settings, so interpret the best score together with stability.</p>"
            )
        else:
            parts.append(
                "<p>The train-test gaps remain fairly controlled across the sweep, which suggests the tested settings are relatively stable.</p>"
            )
    if task_type == "classification":
        parts.append(
            "<p>Look for regions where accuracy stays strong without the boundary becoming unnecessarily complex. Stable regions usually teach more than isolated peaks.</p>"
        )
    else:
        parts.append(
            "<p>For regression, prefer settings that improve test R² while keeping residual behavior smooth and the train-test gap moderate.</p>"
        )
    return "".join(parts)


def explain_run_comparison(run_records: list[dict[str, Any]]) -> str:
    if not run_records:
        return "<p>No saved runs are available for comparison.</p>"
    frame = pd.DataFrame(run_records)
    metric_columns = [column for column in frame.columns if column.startswith("metric_")]
    if not metric_columns:
        return "<p>Saved runs exist, but no comparable metric columns were found.</p>"
    best_metric = metric_columns[0]
    best_row = frame.sort_values(best_metric, ascending=False).iloc[0]
    return (
        f"<p><b>{best_row['model_name']}</b> currently leads the saved comparison on "
        f"<b>{best_metric.replace('metric_', '')}</b>. Use the table to confirm whether that gain is large enough "
        "to justify extra complexity or training time.</p>"
    )
