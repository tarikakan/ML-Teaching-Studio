"""Training workflow page."""

from __future__ import annotations

from typing import Any

import pandas as pd
from PySide6.QtCore import QThread, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ml_teaching_studio.core.preprocessing import PreprocessingOptions
from ml_teaching_studio.core.trainers import TrainingRequest
from ml_teaching_studio.core.validation import (
    ValidationError,
    collect_training_warnings,
    humanize_training_exception,
    validate_training_request,
)
from ml_teaching_studio.gui.widgets.explanation_panel import ExplanationPanel
from ml_teaching_studio.gui.widgets.layout_utils import configure_splitter, wrap_in_scroll_area
from ml_teaching_studio.gui.widgets.metric_table import MetricTable
from ml_teaching_studio.gui.widgets.plot_canvas import PlotCanvas
from ml_teaching_studio.gui.workers.training_worker import TrainingWorker
from ml_teaching_studio.models.model_registry import get_model_spec, list_model_names
from ml_teaching_studio.plotting.classification_plots import (
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_probability_surface,
    plot_roc_curve,
)
from ml_teaching_studio.plotting.comparison_plots import plot_learning_curve_data
from ml_teaching_studio.plotting.regression_plots import (
    plot_coefficients,
    plot_error_distribution,
    plot_prediction_vs_true,
    plot_residuals,
)


class TrainingPage(QWidget):
    def __init__(self, studio, parent=None) -> None:
        super().__init__(parent)
        self.studio = studio
        self.current_result = None
        self.training_thread = None
        self.training_worker = None
        self.hyperparameter_inputs: dict[str, tuple[Any, Any]] = {}

        self.dataset_label = QLabel()
        self.target_combo = QComboBox()
        self.task_combo = QComboBox()
        self.task_combo.addItems(["classification", "regression"])
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.model_combo = QComboBox()
        self.hyperparameter_form = QFormLayout()
        hyperparameter_container = QWidget()
        hyperparameter_container.setLayout(self.hyperparameter_form)
        self.hyperparameter_scroll = QScrollArea()
        self.hyperparameter_scroll.setWidgetResizable(True)
        self.hyperparameter_scroll.setWidget(hyperparameter_container)

        self.scale_checkbox = QCheckBox("Scale numeric features")
        self.scale_checkbox.setChecked(True)
        self.normalize_checkbox = QCheckBox("Normalize rows")
        self.impute_checkbox = QCheckBox("Impute missing values")
        self.impute_checkbox.setChecked(True)
        self.encode_checkbox = QCheckBox("One-hot encode categoricals")
        self.encode_checkbox.setChecked(True)
        self.pca_checkbox = QCheckBox("Use PCA")
        self.pca_spin = QSpinBox()
        self.pca_spin.setRange(2, 20)
        self.pca_spin.setValue(2)
        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setRange(0.05, 0.5)
        self.test_size_spin.setSingleStep(0.05)
        self.test_size_spin.setValue(0.2)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(1, 999999)
        self.seed_spin.setValue(42)
        self.notes_edit = QLineEdit()

        self.train_button = QPushButton("Train Model")
        self.save_button = QPushButton("Save Run")
        self.reset_button = QPushButton("Reset")
        self.save_button.setEnabled(False)
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Idle")
        self.metric_table = MetricTable()
        self.result_plot_combo = QComboBox()
        self.plot_canvas = PlotCanvas()
        self.compatibility_panel = ExplanationPanel("Warnings and Compatibility Notes")
        self.explanation_panel = ExplanationPanel("Result Interpretation")

        controls = self._build_controls_panel()
        results = self._build_results_panel()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(controls)
        splitter.addWidget(results)
        configure_splitter(splitter, [430, 1010], stretch_factors=[0, 1])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.target_combo.currentTextChanged.connect(self._target_changed)
        self.task_combo.currentTextChanged.connect(self._task_changed)
        self.model_combo.currentTextChanged.connect(self._build_hyperparameter_widgets)
        self.model_combo.currentTextChanged.connect(self._update_compatibility_notes)
        self.result_plot_combo.currentTextChanged.connect(self._render_result_plot)
        self.train_button.clicked.connect(self._start_training)
        self.save_button.clicked.connect(self._save_run)
        self.reset_button.clicked.connect(self._reset_form)
        self.feature_list.itemSelectionChanged.connect(self._update_compatibility_notes)
        self.scale_checkbox.stateChanged.connect(self._update_compatibility_notes)
        self.normalize_checkbox.stateChanged.connect(self._update_compatibility_notes)
        self.impute_checkbox.stateChanged.connect(self._update_compatibility_notes)
        self.encode_checkbox.stateChanged.connect(self._update_compatibility_notes)
        self.pca_checkbox.stateChanged.connect(self._update_compatibility_notes)
        self.pca_spin.valueChanged.connect(self._update_compatibility_notes)
        self.test_size_spin.valueChanged.connect(self._update_compatibility_notes)

        self._populate_from_state()

    def _build_controls_panel(self) -> QWidget:
        dataset_box = QGroupBox("Training Setup")
        dataset_form = QFormLayout(dataset_box)
        dataset_form.addRow("Active dataset", self.dataset_label)
        dataset_form.addRow("Target column", self.target_combo)
        dataset_form.addRow("Task type", self.task_combo)
        dataset_form.addRow("Feature selection", self.feature_list)

        preprocessing_box = QGroupBox("Preprocessing")
        preprocessing_form = QFormLayout(preprocessing_box)
        preprocessing_form.addRow("", self.scale_checkbox)
        preprocessing_form.addRow("", self.normalize_checkbox)
        preprocessing_form.addRow("", self.impute_checkbox)
        preprocessing_form.addRow("", self.encode_checkbox)
        preprocessing_form.addRow("", self.pca_checkbox)
        preprocessing_form.addRow("PCA components", self.pca_spin)
        preprocessing_form.addRow("Test size", self.test_size_spin)
        preprocessing_form.addRow("Random seed", self.seed_spin)
        preprocessing_form.addRow("Notes", self.notes_edit)

        model_box = QGroupBox("Model and Hyperparameters")
        model_layout = QVBoxLayout(model_box)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.hyperparameter_scroll, stretch=1)

        run_row = QHBoxLayout()
        run_row.addWidget(self.train_button)
        run_row.addWidget(self.save_button)
        run_row.addWidget(self.reset_button)

        panel_content = QWidget()
        panel_content.setObjectName("ControlPanel")
        layout = QVBoxLayout(panel_content)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)
        layout.addWidget(dataset_box)
        layout.addWidget(preprocessing_box)
        layout.addWidget(model_box, stretch=1)
        layout.addLayout(run_row)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.compatibility_panel, stretch=1)
        return wrap_in_scroll_area(panel_content)

    def _build_results_panel(self) -> QWidget:
        metrics_panel = QWidget()
        metrics_panel.setObjectName("MetricsSection")
        metrics_layout = QVBoxLayout(metrics_panel)
        metrics_layout.setContentsMargins(14, 14, 14, 14)
        metrics_layout.setSpacing(10)
        metrics_layout.addWidget(QLabel("Evaluation metrics"))
        metrics_layout.addWidget(self.metric_table, stretch=1)

        plot_panel = QWidget()
        plot_panel.setObjectName("PlotSection")
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(14, 14, 14, 14)
        plot_layout.setSpacing(10)
        plot_layout.addWidget(QLabel("Result plot"))
        plot_layout.addWidget(self.result_plot_combo)
        plot_layout.addWidget(self.plot_canvas, stretch=1)

        insight_splitter = QSplitter(Qt.Vertical)
        insight_splitter.addWidget(plot_panel)
        insight_splitter.addWidget(self.explanation_panel)
        configure_splitter(insight_splitter, [620, 240], stretch_factors=[1, 0])

        results_splitter = QSplitter(Qt.Vertical)
        results_splitter.addWidget(metrics_panel)
        results_splitter.addWidget(insight_splitter)
        configure_splitter(results_splitter, [260, 620], stretch_factors=[0, 1])
        return results_splitter

    def _populate_from_state(self) -> None:
        bundle = self.studio.current_dataset
        self.dataset_label.setText(bundle.name)
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItems([str(column) for column in bundle.dataframe.columns])
        self.target_combo.setCurrentText(self.studio.selected_target_column)
        self.target_combo.blockSignals(False)
        self.task_combo.setCurrentText(self.studio.selected_task_type)
        self._populate_feature_list()
        self._populate_models()
        if self.studio.preferred_model_name in list_model_names(self.task_combo.currentText()):
            self.model_combo.setCurrentText(self.studio.preferred_model_name)
        self._build_hyperparameter_widgets()
        self._update_result_plot_choices()
        self._update_compatibility_notes()

    def _populate_feature_list(self) -> None:
        self.feature_list.clear()
        selected = set(self.studio.selected_feature_columns)
        target = self.target_combo.currentText()
        for column in self.studio.current_dataset.dataframe.columns:
            if column == target:
                continue
            item = QListWidgetItem(column)
            self.feature_list.addItem(item)
            item.setSelected(column in selected or not selected)

    def _target_changed(self) -> None:
        self._populate_feature_list()
        self._update_compatibility_notes()

    def _task_changed(self) -> None:
        self._populate_models()
        self._update_compatibility_notes()

    def _populate_models(self) -> None:
        task_type = self.task_combo.currentText()
        current = self.model_combo.currentText()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(list_model_names(task_type))
        if current in list_model_names(task_type):
            self.model_combo.setCurrentText(current)
        elif self.studio.preferred_model_name in list_model_names(task_type):
            self.model_combo.setCurrentText(self.studio.preferred_model_name)
        self.model_combo.blockSignals(False)
        self._build_hyperparameter_widgets()

    def _clear_form_layout(self) -> None:
        while self.hyperparameter_form.rowCount():
            self.hyperparameter_form.removeRow(0)
        self.hyperparameter_inputs.clear()

    def _build_hyperparameter_widgets(self) -> None:
        self._clear_form_layout()
        model_name = self.model_combo.currentText()
        if not model_name:
            return
        spec = get_model_spec(model_name)
        for name, field_spec in spec.hyperparameters.items():
            if field_spec.widget == "spin":
                widget = QSpinBox()
                widget.setRange(int(field_spec.min_value or 0), int(field_spec.max_value or 999999))
                widget.setValue(int(field_spec.default))
            elif field_spec.widget == "double":
                widget = QDoubleSpinBox()
                widget.setDecimals(5)
                widget.setRange(float(field_spec.min_value or 0.0), float(field_spec.max_value or 999999.0))
                widget.setSingleStep(float(field_spec.step or 0.1))
                widget.setValue(float(field_spec.default))
            elif field_spec.widget == "combo":
                widget = QComboBox()
                widget.addItems([str(choice) for choice in field_spec.choices])
                widget.setCurrentText(str(field_spec.default))
            else:
                widget = QLineEdit(str(field_spec.default))
            widget.setToolTip(field_spec.description)
            self.hyperparameter_form.addRow(f"{name}", widget)
            self.hyperparameter_inputs[name] = (field_spec, widget)
        self.studio.set_preferred_model(model_name, refresh=False)
        self._update_compatibility_notes()

    def _collect_hyperparameters(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, (_, widget) in self.hyperparameter_inputs.items():
            if isinstance(widget, QSpinBox):
                params[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                params[name] = widget.value()
            elif isinstance(widget, QComboBox):
                params[name] = widget.currentText()
            else:
                params[name] = widget.text().strip()
        return params

    def _selected_features(self) -> list[str]:
        available_columns = set(self.studio.current_dataset.dataframe.columns)
        return [item.text() for item in self.feature_list.selectedItems() if item.text() in available_columns]

    def _preprocessing_options(self) -> PreprocessingOptions:
        return PreprocessingOptions(
            scale_numeric=self.scale_checkbox.isChecked(),
            normalize_numeric=self.normalize_checkbox.isChecked(),
            impute_missing=self.impute_checkbox.isChecked(),
            encode_categorical=self.encode_checkbox.isChecked(),
            use_pca=self.pca_checkbox.isChecked(),
            pca_components=self.pca_spin.value() if self.pca_checkbox.isChecked() else None,
        )

    def _start_training(self) -> None:
        self.studio.apply_dataset_configuration(
            target_column=self.target_combo.currentText(),
            feature_columns=self._selected_features(),
            task_type=self.task_combo.currentText(),
            refresh=False,
        )
        request = self._build_training_request()
        try:
            validate_training_request(request)
        except ValidationError as exc:
            self._set_compatibility_notes([], validation_message=str(exc))
            self.studio.show_error("Training Setup", str(exc))
            return
        self._set_compatibility_notes(collect_training_warnings(request))
        self.train_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting training...")

        self.training_thread = QThread(self)
        self.training_worker = TrainingWorker(request)
        self.training_worker.moveToThread(self.training_thread)
        self.training_thread.started.connect(self.training_worker.run)
        self.training_worker.progress.connect(self._update_progress)
        self.training_worker.finished.connect(self._training_finished)
        self.training_worker.error.connect(self._training_failed)
        self.training_worker.finished.connect(self.training_thread.quit)
        self.training_worker.error.connect(self.training_thread.quit)
        self.training_worker.finished.connect(self.training_worker.deleteLater)
        self.training_worker.error.connect(self.training_worker.deleteLater)
        self.training_thread.finished.connect(self.training_thread.deleteLater)
        self.training_thread.start()

    def _update_progress(self, message: str, value: int) -> None:
        self.progress_label.setText(message)
        self.progress_bar.setValue(value)

    def _training_finished(self, result) -> None:
        self.current_result = result
        self.studio.set_training_result(result)

    def _display_result(self, result) -> None:
        frame = pd.DataFrame(
            {
                "metric": list(result.test_metrics.keys()),
                "train": [result.train_metrics.get(name) for name in result.test_metrics.keys()],
                "test": [result.test_metrics.get(name) for name in result.test_metrics.keys()],
            }
        )
        self.metric_table.set_dataframe(frame)
        self.explanation_panel.set_html(result.explanation)
        self._set_compatibility_notes(result.warnings)
        self._update_result_plot_choices()
        self._render_result_plot()
        self.train_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.progress_label.setText("Training complete.")
        self.progress_bar.setValue(100)

    def _training_failed(self, message: str) -> None:
        self.train_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Training failed.")
        friendly = humanize_training_exception(Exception(message))
        self._set_compatibility_notes([], validation_message=friendly)
        self.studio.show_error("Training Error", friendly)

    def _update_result_plot_choices(self) -> None:
        self.result_plot_combo.clear()
        task_type = self.task_combo.currentText()
        if task_type == "classification":
            options = [
                "Confusion Matrix",
                "ROC Curve",
                "Precision-Recall Curve",
                "Decision Boundary",
                "Probability Surface",
                "Feature Importance",
                "Learning Curve",
            ]
        else:
            options = [
                "Prediction vs True",
                "Residual Plot",
                "Error Distribution",
                "Coefficient / Importance Plot",
                "Learning Curve",
            ]
        self.result_plot_combo.addItems(options)

    def _render_result_plot(self) -> None:
        if self.current_result is None:
            self.plot_canvas.show_placeholder("Train a model to view evaluation plots.")
            return
        plot_name = self.result_plot_combo.currentText()
        result = self.current_result
        try:
            if plot_name == "Confusion Matrix":
                figure = plot_confusion_matrix(result)
            elif plot_name == "ROC Curve":
                figure = plot_roc_curve(result)
            elif plot_name == "Precision-Recall Curve":
                figure = plot_precision_recall_curve(result)
            elif plot_name == "Decision Boundary":
                figure = plot_decision_boundary(result)
            elif plot_name == "Probability Surface":
                figure = plot_probability_surface(result)
            elif plot_name == "Feature Importance":
                figure = plot_feature_importance(result)
            elif plot_name == "Prediction vs True":
                figure = plot_prediction_vs_true(result)
            elif plot_name == "Residual Plot":
                figure = plot_residuals(result)
            elif plot_name == "Error Distribution":
                figure = plot_error_distribution(result)
            elif plot_name == "Coefficient / Importance Plot":
                figure = plot_coefficients(result)
            else:
                figure = plot_learning_curve_data(
                    f"{result.model_name}: learning curve",
                    result.learning_curve_data,
                )
            self.plot_canvas.set_figure(figure)
        except Exception as exc:
            self.plot_canvas.show_placeholder("This plot is not available for the current model and dataset setup.")
            self._set_compatibility_notes(
                self.current_result.warnings if self.current_result is not None else [],
                validation_message=humanize_training_exception(exc),
            )

    def _save_run(self) -> None:
        if self.current_result is None:
            return
        self.studio.save_training_result(notes=self.notes_edit.text().strip())
        self.save_button.setEnabled(False)

    def _reset_form(self) -> None:
        self.notes_edit.clear()
        self.scale_checkbox.setChecked(True)
        self.normalize_checkbox.setChecked(False)
        self.impute_checkbox.setChecked(True)
        self.encode_checkbox.setChecked(True)
        self.pca_checkbox.setChecked(False)
        self.pca_spin.setValue(2)
        self.test_size_spin.setValue(0.2)
        self.seed_spin.setValue(42)
        self._populate_from_state()

    def on_dataset_changed(self) -> None:
        self.current_result = None
        self._populate_from_state()
        self.plot_canvas.show_placeholder("Train a model to view evaluation plots.")

    def on_training_result_changed(self) -> None:
        self.current_result = self.studio.current_training_result
        if self.current_result is not None:
            self._display_result(self.current_result)

    def _build_training_request(self) -> TrainingRequest:
        return TrainingRequest(
            dataset=self.studio.current_dataset,
            task_type=self.task_combo.currentText(),
            target_column=self.target_combo.currentText(),
            feature_columns=self._selected_features(),
            model_name=self.model_combo.currentText(),
            hyperparameters=self._collect_hyperparameters(),
            preprocessing=self._preprocessing_options(),
            test_size=self.test_size_spin.value(),
            random_seed=self.seed_spin.value(),
            notes=self.notes_edit.text().strip(),
        )

    def _set_compatibility_notes(
        self,
        warnings: list[str],
        *,
        validation_message: str | None = None,
    ) -> None:
        if not warnings and not validation_message:
            self.compatibility_panel.set_html(
                "<p>No compatibility issues are currently detected. If you change the model, preprocessing, or split, new notes may appear here.</p>"
            )
            return
        parts = []
        if validation_message:
            parts.append(f"<p><b>Validation note:</b> {validation_message}</p>")
        if warnings:
            items = "".join(f"<li>{warning}</li>" for warning in warnings)
            parts.append(f"<ul>{items}</ul>")
        self.compatibility_panel.set_html("".join(parts))

    def _update_compatibility_notes(self) -> None:
        try:
            request = self._build_training_request()
            validate_training_request(request)
            self._set_compatibility_notes(collect_training_warnings(request))
        except ValidationError as exc:
            try:
                request = self._build_training_request()
                warnings = collect_training_warnings(request)
            except Exception:
                warnings = []
            self._set_compatibility_notes(warnings, validation_message=str(exc))
        except Exception:
            self._set_compatibility_notes([], validation_message="The form is updating for the current dataset and model selection.")
