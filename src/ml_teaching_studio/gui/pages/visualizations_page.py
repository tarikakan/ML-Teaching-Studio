"""Visualization gallery page."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QSplitter, QVBoxLayout, QWidget

from ml_teaching_studio.plotting import data_plots, hyperparameter_plots
from ml_teaching_studio.plotting.classification_plots import (
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_probability_surface,
    plot_roc_curve,
)
from ml_teaching_studio.plotting.comparison_plots import (
    plot_inference_time,
    plot_learning_curve_data,
    plot_metric_bars,
    plot_training_time,
)
from ml_teaching_studio.plotting.regression_plots import (
    plot_coefficients,
    plot_error_distribution,
    plot_prediction_vs_true,
    plot_residuals,
)
from ml_teaching_studio.gui.widgets.explanation_panel import ExplanationPanel
from ml_teaching_studio.gui.widgets.layout_utils import configure_splitter
from ml_teaching_studio.gui.widgets.plot_canvas import PlotCanvas


class VisualizationsPage(QWidget):
    def __init__(self, studio, parent=None) -> None:
        super().__init__(parent)
        self.studio = studio
        self.category_combo = QComboBox()
        self.category_combo.addItems(
            [
                "Data Understanding",
                "Model Performance",
                "Model Behavior",
                "Hyperparameter Sensitivity",
                "Model Comparison",
            ]
        )
        self.plot_combo = QComboBox()
        self.context_label = QLabel()
        self.plot_canvas = PlotCanvas()
        self.explanation_panel = ExplanationPanel("What This Visual Teaches")

        controls_panel = QWidget()
        controls_panel.setObjectName("ControlPanel")
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(14, 14, 14, 14)
        controls_layout.setSpacing(12)
        form = QFormLayout()
        form.addRow("Visual category", self.category_combo)
        form.addRow("Selected plot", self.plot_combo)
        controls_layout.addLayout(form)
        controls_layout.addWidget(QLabel("Current context"))
        controls_layout.addWidget(self.context_label)
        controls_layout.addStretch(1)

        plot_panel = QWidget()
        plot_panel.setObjectName("PlotSection")
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(14, 14, 14, 14)
        plot_layout.setSpacing(10)
        plot_layout.addWidget(self.plot_canvas, stretch=1)

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(plot_panel)
        right_splitter.addWidget(self.explanation_panel)
        configure_splitter(right_splitter, [640, 240], stretch_factors=[1, 0])

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(controls_panel)
        main_splitter.addWidget(right_splitter)
        configure_splitter(main_splitter, [320, 1120], stretch_factors=[0, 1])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(main_splitter, stretch=1)

        self.category_combo.currentTextChanged.connect(self._populate_plot_options)
        self.plot_combo.currentTextChanged.connect(self._render)
        self._populate_plot_options()

    def _populate_plot_options(self) -> None:
        category = self.category_combo.currentText()
        self.plot_combo.clear()
        if category == "Data Understanding":
            options = ["Class Distribution", "Feature Histograms", "Correlation Heatmap", "Missing Values", "PCA Projection", "2D Preview"]
        elif category == "Model Performance":
            if self.studio.current_training_result and self.studio.current_training_result.task_type == "classification":
                options = ["Confusion Matrix", "ROC Curve", "Precision-Recall Curve", "Learning Curve"]
            else:
                options = ["Prediction vs True", "Residual Plot", "Error Distribution", "Learning Curve"]
        elif category == "Model Behavior":
            if self.studio.current_training_result and self.studio.current_training_result.task_type == "classification":
                options = ["Decision Boundary", "Probability Surface", "Feature Importance"]
            else:
                options = ["Coefficient / Importance Plot"]
        elif category == "Hyperparameter Sensitivity":
            options = ["Validation Curve", "Train-Test Gap"]
            if self.studio.current_sweep_result and len(self.studio.current_sweep_result.param_names) > 1:
                options.append("Heatmap")
        else:
            options = ["Primary Metric Bars", "Training Time", "Inference Time"]
        self.plot_combo.addItems(options)
        self._render()

    def _render(self) -> None:
        category = self.category_combo.currentText()
        plot_name = self.plot_combo.currentText()
        dataset = self.studio.current_dataset
        result = self.studio.current_training_result
        sweep = self.studio.current_sweep_result
        run_records = self.studio.run_store.list_runs()
        self.context_label.setText(
            f"Dataset: {dataset.name} | Model: {result.model_name if result else 'No trained model yet'}"
        )

        if category == "Data Understanding":
            if plot_name == "Class Distribution":
                figure = data_plots.plot_class_distribution(dataset)
                explanation = "<p>Use class distribution to judge whether accuracy alone will be trustworthy.</p>"
            elif plot_name == "Feature Histograms":
                figure = data_plots.plot_feature_histograms(dataset)
                explanation = "<p>Histograms show whether scaling, transformation, or outlier handling might matter.</p>"
            elif plot_name == "Correlation Heatmap":
                figure = data_plots.plot_correlation_heatmap(dataset)
                explanation = "<p>High correlations often affect coefficient-based models and interpretation.</p>"
            elif plot_name == "Missing Values":
                figure = data_plots.plot_missing_values(dataset)
                explanation = "<p>Missing-value patterns help determine whether imputation should be part of the workflow.</p>"
            elif plot_name == "PCA Projection":
                figure = data_plots.plot_pca_projection(dataset)
                explanation = "<p>PCA can reveal broad structure even before a model is trained.</p>"
            else:
                figure = data_plots.plot_scatter_preview(dataset)
                explanation = "<p>Simple 2D previews are especially useful for synthetic datasets and decision-boundary lessons.</p>"
        elif category == "Model Performance":
            if result is None:
                self.plot_canvas.show_placeholder("Train a model first to unlock performance plots.")
                self.explanation_panel.set_text("No training result is available yet.")
                return
            if plot_name == "Confusion Matrix":
                figure = plot_confusion_matrix(result)
                explanation = "<p>The confusion matrix shows which classes are being confused, not just how often the model is correct overall.</p>"
            elif plot_name == "ROC Curve":
                figure = plot_roc_curve(result)
                explanation = "<p>ROC curves show ranking quality across thresholds, which is useful when the decision threshold is not fixed.</p>"
            elif plot_name == "Precision-Recall Curve":
                figure = plot_precision_recall_curve(result)
                explanation = "<p>Precision-recall curves are especially informative when positive cases are rare or costly.</p>"
            elif plot_name == "Prediction vs True":
                figure = plot_prediction_vs_true(result)
                explanation = "<p>Points close to the diagonal represent accurate predictions; systematic drift suggests bias.</p>"
            elif plot_name == "Residual Plot":
                figure = plot_residuals(result)
                explanation = "<p>Residual plots help you see whether errors are random or whether the model is missing structure.</p>"
            elif plot_name == "Error Distribution":
                figure = plot_error_distribution(result)
                explanation = "<p>Error distributions show whether a few large mistakes dominate the regression quality.</p>"
            else:
                figure = plot_learning_curve_data(f"{result.model_name}: learning curve", result.learning_curve_data)
                explanation = "<p>Learning curves show whether more data is likely to help and whether train-test gaps shrink with sample size.</p>"
        elif category == "Model Behavior":
            if result is None:
                self.plot_canvas.show_placeholder("Train a model first to unlock behavior plots.")
                self.explanation_panel.set_text("No training result is available yet.")
                return
            if plot_name == "Decision Boundary":
                figure = plot_decision_boundary(result)
                explanation = "<p>Decision boundaries make flexibility visible. Very jagged boundaries often indicate higher variance.</p>"
            elif plot_name == "Probability Surface":
                figure = plot_probability_surface(result)
                explanation = "<p>Probability surfaces show not only the predicted class but also model confidence across feature space.</p>"
            elif plot_name == "Feature Importance":
                figure = plot_feature_importance(result)
                explanation = "<p>Feature importance highlights what the model depended on most, but it does not prove causality.</p>"
            else:
                figure = plot_coefficients(result)
                explanation = "<p>Coefficient and importance plots help connect model output back to specific inputs.</p>"
        elif category == "Hyperparameter Sensitivity":
            if sweep is None:
                self.plot_canvas.show_placeholder("Run the Hyperparameter Lab first to unlock sweep plots.")
                self.explanation_panel.set_text("No sweep result is available yet.")
                return
            if plot_name == "Validation Curve":
                figure = hyperparameter_plots.plot_validation_curve(sweep)
                explanation = "<p>Validation curves show where extra flexibility stops helping on unseen data.</p>"
            elif plot_name == "Train-Test Gap":
                figure = hyperparameter_plots.plot_train_test_gap(sweep)
                explanation = "<p>The train-test gap is one of the clearest indicators of overfitting pressure during a sweep.</p>"
            else:
                figure = hyperparameter_plots.plot_heatmap(sweep)
                explanation = "<p>Two-parameter heatmaps show whether the best performance lies in a stable region or a fragile corner.</p>"
        else:
            if plot_name == "Primary Metric Bars":
                metric_name = "accuracy" if any(record.get("task_type") == "classification" for record in run_records) else "r2"
                figure = plot_metric_bars(run_records, metric_name)
                explanation = "<p>Metric bars let you compare saved runs side by side, but only runs with comparable setup should be interpreted together.</p>"
            elif plot_name == "Training Time":
                figure = plot_training_time(run_records)
                explanation = "<p>Training time matters when a modest metric gain comes with much higher cost.</p>"
            else:
                figure = plot_inference_time(run_records)
                explanation = "<p>Inference time matters when a model will be used repeatedly or interactively.</p>"
        self.plot_canvas.set_figure(figure)
        self.explanation_panel.set_html(explanation)

    def on_dataset_changed(self) -> None:
        self._populate_plot_options()

    def on_training_result_changed(self) -> None:
        self._populate_plot_options()

    def on_sweep_result_changed(self) -> None:
        self._populate_plot_options()

    def on_run_store_changed(self) -> None:
        self._populate_plot_options()
