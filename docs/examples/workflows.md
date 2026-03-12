# Example Workflows

## Workflow 1: Introductory Classification Lesson

1. Load `Iris`.
2. Train `Logistic Regression`.
3. Inspect the confusion matrix and ROC-related metrics.
4. Switch to `K-Nearest Neighbors`.
5. Sweep `n_neighbors` in the Hyperparameter Lab.
6. Compare saved runs.

## Workflow 2: Overfitting Demonstration

1. Load `Synthetic Noisy Classification`.
2. Train `Decision Tree Classifier` with a large `max_depth`.
3. Save the run.
4. Retrain with smaller `max_depth` or larger `min_samples_leaf`.
5. Compare train-test gap and decision-boundary smoothness.

## Workflow 3: Regression Diagnostics

1. Load `Synthetic Noisy Regression` or `California Housing`.
2. Train `Linear Regression`, then `Random Forest Regressor`.
3. Inspect residuals and prediction-vs-true plots.
4. Save both runs and compare timing and performance.

## Screenshot Placeholders

Planned screenshot areas:

- Home dashboard
- Dataset exploration page
- Training page with metrics and plot output
- Hyperparameter Lab validation curve
- Compare Runs table and comparison chart
