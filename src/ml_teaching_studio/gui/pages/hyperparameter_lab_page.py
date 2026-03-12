"""Hyperparameter sweep page."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QThread, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
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

from ml_teaching_studio.core.hyperparameter_sweeps import SweepRequest
from ml_teaching_studio.core.preprocessing import PreprocessingOptions
from ml_teaching_studio.gui.widgets.explanation_panel import ExplanationPanel
from ml_teaching_studio.gui.widgets.layout_utils import configure_splitter, wrap_in_scroll_area
from ml_teaching_studio.gui.widgets.metric_table import MetricTable
from ml_teaching_studio.gui.widgets.plot_canvas import PlotCanvas
from ml_teaching_studio.gui.workers.sweep_worker import SweepWorker
from ml_teaching_studio.models.model_registry import get_model_spec, list_model_names
from ml_teaching_studio.plotting import hyperparameter_plots
from ml_teaching_studio.utils.helpers import parse_value_list


class HyperparameterLabPage(QWidget):
    def __init__(self, studio, parent=None) -> None:
        super().__init__(parent)
        self.studio = studio
        self.current_result = None
        self.sweep_thread = None
        self.sweep_worker = None
        self.base_hyperparameter_inputs: dict[str, tuple[Any, Any]] = {}

        self.dataset_label = QLabel()
        self.target_combo = QComboBox()
        self.task_combo = QComboBox()
        self.task_combo.addItems(["classification", "regression"])
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.model_combo = QComboBox()
        self.param_one_combo = QComboBox()
        self.param_one_values = QLineEdit("1,3,5,7,9")
        self.param_two_combo = QComboBox()
        self.param_two_values = QLineEdit()
        self.metric_combo = QComboBox()
        self.seeds_edit = QLineEdit("42,43,44")
        self.test_size_spin = QDoubleSpinBox()
        self.test_size_spin.setRange(0.05, 0.5)
        self.test_size_spin.setSingleStep(0.05)
        self.test_size_spin.setValue(0.2)
        self.notes_edit = QLineEdit()
        self.scale_checkbox = QPushButton("Using training-page preprocessing defaults")
        self.scale_checkbox.setEnabled(False)

        self.base_form = QFormLayout()
        base_container = QWidget()
        base_container.setLayout(self.base_form)
        base_scroll = QScrollArea()
        base_scroll.setWidgetResizable(True)
        base_scroll.setWidget(base_container)

        self.run_button = QPushButton("Run Sweep")
        self.save_button = QPushButton("Save Sweep")
        self.save_button.setEnabled(False)
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Idle")
        self.result_table = MetricTable()
        self.plot_combo = QComboBox()
        self.plot_canvas = PlotCanvas()
        self.explanation_panel = ExplanationPanel("Sweep Interpretation")

        setup_box = QGroupBox("Sweep Setup")
        setup_form = QFormLayout(setup_box)
        setup_form.addRow("Active dataset", self.dataset_label)
        setup_form.addRow("Target column", self.target_combo)
        setup_form.addRow("Task type", self.task_combo)
        setup_form.addRow("Feature selection", self.feature_list)
        setup_form.addRow("Model", self.model_combo)
        setup_form.addRow("Hyperparameter 1", self.param_one_combo)
        setup_form.addRow("Values 1", self.param_one_values)
        setup_form.addRow("Hyperparameter 2", self.param_two_combo)
        setup_form.addRow("Values 2", self.param_two_values)
        setup_form.addRow("Metric", self.metric_combo)
        setup_form.addRow("Seeds", self.seeds_edit)
        setup_form.addRow("Test size", self.test_size_spin)
        setup_form.addRow("Notes", self.notes_edit)

        left_content = QWidget()
        left_content.setObjectName("ControlPanel")
        left_layout = QVBoxLayout(left_content)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(12)
        left_layout.addWidget(setup_box)
        left_layout.addWidget(QLabel("Base hyperparameters"))
        left_layout.addWidget(base_scroll, stretch=1)
        button_row = QHBoxLayout()
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.save_button)
        left_layout.addLayout(button_row)
        left_layout.addWidget(self.progress_bar)
        left_layout.addWidget(self.progress_label)
        left_layout.addStretch(1)
        left_panel = wrap_in_scroll_area(left_content)

        results_panel = QWidget()
        results_panel.setObjectName("MetricsSection")
        right_layout = QVBoxLayout(results_panel)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)
        right_layout.addWidget(QLabel("Sweep results"))
        right_layout.addWidget(self.result_table, stretch=1)

        plot_panel = QWidget()
        plot_panel.setObjectName("PlotSection")
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(14, 14, 14, 14)
        plot_layout.setSpacing(10)
        plot_layout.addWidget(QLabel("Sweep visualization"))
        plot_layout.addWidget(self.plot_combo)
        plot_layout.addWidget(self.plot_canvas, stretch=1)

        insight_splitter = QSplitter(Qt.Vertical)
        insight_splitter.addWidget(plot_panel)
        insight_splitter.addWidget(self.explanation_panel)
        configure_splitter(insight_splitter, [620, 240], stretch_factors=[1, 0])

        right_panel = QSplitter(Qt.Vertical)
        right_panel.addWidget(results_panel)
        right_panel.addWidget(insight_splitter)
        configure_splitter(right_panel, [280, 620], stretch_factors=[0, 1])

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        configure_splitter(splitter, [430, 1010], stretch_factors=[0, 1])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.task_combo.currentTextChanged.connect(self._populate_models)
        self.target_combo.currentTextChanged.connect(self._target_changed)
        self.model_combo.currentTextChanged.connect(self._build_base_form)
        self.plot_combo.currentTextChanged.connect(self._render_plot)
        self.run_button.clicked.connect(self._run_sweep)
        self.save_button.clicked.connect(self._save_sweep)

        self._populate_from_state()

    def _populate_from_state(self) -> None:
        bundle = self.studio.current_dataset
        self.dataset_label.setText(bundle.name)
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItems([str(column) for column in bundle.dataframe.columns])
        self.target_combo.setCurrentText(self.studio.selected_target_column)
        self.target_combo.blockSignals(False)
        self.task_combo.setCurrentText(self.studio.selected_task_type)
        self._populate_features()
        self._update_metric_choices()
        self._populate_models()
        self._update_plot_options()

    def _populate_features(self) -> None:
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
        self._populate_features()

    def _populate_models(self) -> None:
        task_type = self.task_combo.currentText()
        self._update_metric_choices()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(list_model_names(task_type))
        preferred = self.studio.preferred_model_name
        if preferred in list_model_names(task_type):
            self.model_combo.setCurrentText(preferred)
        self.model_combo.blockSignals(False)
        self._build_base_form()

    def _clear_form_layout(self) -> None:
        while self.base_form.rowCount():
            self.base_form.removeRow(0)
        self.base_hyperparameter_inputs.clear()

    def _build_base_form(self) -> None:
        self._clear_form_layout()
        model_name = self.model_combo.currentText()
        if not model_name:
            return
        spec = get_model_spec(model_name)
        parameter_names = list(spec.hyperparameters.keys())
        self.param_one_combo.clear()
        self.param_one_combo.addItems(parameter_names)
        self.param_two_combo.clear()
        self.param_two_combo.addItem("None")
        self.param_two_combo.addItems(parameter_names)
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
            self.base_form.addRow(name, widget)
            self.base_hyperparameter_inputs[name] = (field_spec, widget)

    def _collect_base_hyperparameters(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for name, (_, widget) in self.base_hyperparameter_inputs.items():
            if isinstance(widget, QSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                values[name] = widget.value()
            elif isinstance(widget, QComboBox):
                values[name] = widget.currentText()
            else:
                values[name] = widget.text().strip()
        return values

    def _selected_features(self) -> list[str]:
        return [item.text() for item in self.feature_list.selectedItems()]

    def _update_plot_options(self) -> None:
        self.plot_combo.clear()
        self.plot_combo.addItems(["Validation Curve", "Train-Test Gap"])
        if self.current_result is not None and len(self.current_result.param_names) > 1:
            self.plot_combo.addItem("Heatmap")

    def _preprocessing(self) -> PreprocessingOptions:
        return PreprocessingOptions(scale_numeric=True, normalize_numeric=False, impute_missing=True, encode_categorical=True)

    def _run_sweep(self) -> None:
        self.studio.apply_dataset_configuration(
            target_column=self.target_combo.currentText(),
            feature_columns=self._selected_features(),
            task_type=self.task_combo.currentText(),
            refresh=False,
        )
        grid = {
            self.param_one_combo.currentText(): parse_value_list(self.param_one_values.text()),
        }
        if self.param_two_combo.currentText() != "None" and self.param_two_values.text().strip():
            grid[self.param_two_combo.currentText()] = parse_value_list(self.param_two_values.text())
        request = SweepRequest(
            dataset=self.studio.current_dataset,
            task_type=self.task_combo.currentText(),
            target_column=self.target_combo.currentText(),
            feature_columns=self._selected_features(),
            model_name=self.model_combo.currentText(),
            preprocessing=self._preprocessing(),
            base_hyperparameters=self._collect_base_hyperparameters(),
            param_grid=grid,
            metric_name=self.metric_combo.currentText(),
            test_size=self.test_size_spin.value(),
            repeat_seeds=[int(value) for value in parse_value_list(self.seeds_edit.text()) or [42]],
            notes=self.notes_edit.text().strip(),
        )
        self.run_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting sweep...")

        self.sweep_thread = QThread(self)
        self.sweep_worker = SweepWorker(request)
        self.sweep_worker.moveToThread(self.sweep_thread)
        self.sweep_thread.started.connect(self.sweep_worker.run)
        self.sweep_worker.progress.connect(self._update_progress)
        self.sweep_worker.finished.connect(self._sweep_finished)
        self.sweep_worker.error.connect(self._sweep_failed)
        self.sweep_worker.finished.connect(self.sweep_thread.quit)
        self.sweep_worker.error.connect(self.sweep_thread.quit)
        self.sweep_worker.finished.connect(self.sweep_worker.deleteLater)
        self.sweep_worker.error.connect(self.sweep_worker.deleteLater)
        self.sweep_thread.finished.connect(self.sweep_thread.deleteLater)
        self.sweep_thread.start()

    def _update_progress(self, message: str, value: int) -> None:
        self.progress_label.setText(message)
        self.progress_bar.setValue(value)

    def _sweep_finished(self, result) -> None:
        self.current_result = result
        self.studio.set_sweep_result(result)

    def _display_result(self, result) -> None:
        self.result_table.set_dataframe(result.to_frame())
        self.explanation_panel.set_html(result.explanation)
        self._update_plot_options()
        self._render_plot()
        self.run_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.progress_label.setText("Sweep complete.")
        self.progress_bar.setValue(100)

    def _sweep_failed(self, message: str) -> None:
        self.run_button.setEnabled(True)
        self.progress_label.setText("Sweep failed.")
        self.studio.show_error("Sweep Error", message)

    def _render_plot(self) -> None:
        if self.current_result is None:
            self.plot_canvas.show_placeholder("Run a hyperparameter sweep to view the results.")
            return
        if self.plot_combo.currentText() == "Validation Curve":
            figure = hyperparameter_plots.plot_validation_curve(self.current_result)
        elif self.plot_combo.currentText() == "Train-Test Gap":
            figure = hyperparameter_plots.plot_train_test_gap(self.current_result)
        else:
            figure = hyperparameter_plots.plot_heatmap(self.current_result)
        self.plot_canvas.set_figure(figure)

    def _save_sweep(self) -> None:
        if self.current_result is None:
            return
        self.studio.save_sweep_result(notes=self.notes_edit.text().strip())
        self.save_button.setEnabled(False)

    def on_dataset_changed(self) -> None:
        self.current_result = None
        self._populate_from_state()

    def on_sweep_result_changed(self) -> None:
        self.current_result = self.studio.current_sweep_result
        if self.current_result is not None:
            self._display_result(self.current_result)

    def _update_metric_choices(self) -> None:
        task_type = self.task_combo.currentText()
        current = self.metric_combo.currentText()
        options = ["accuracy", "f1_weighted", "roc_auc"] if task_type == "classification" else ["r2", "rmse", "mae"]
        self.metric_combo.clear()
        self.metric_combo.addItems(options)
        if current in options:
            self.metric_combo.setCurrentText(current)
