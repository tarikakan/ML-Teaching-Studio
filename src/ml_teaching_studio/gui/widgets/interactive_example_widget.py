"""Reusable live demo widget for interactive model examples."""

from __future__ import annotations

from typing import Any

import pandas as pd
from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ml_teaching_studio.core.interactive_examples import (
    build_interactive_comparison_explanation,
    build_interactive_explanation,
    comparison_candidates,
    default_demo_hyperparameters,
    interactive_metric_frame,
    interactive_scenarios,
    run_interactive_comparison,
    run_interactive_example,
    suggested_comparison_model,
)
from ml_teaching_studio.models.model_registry import get_model_spec
from ml_teaching_studio.plotting.interactive_plots import (
    plot_interactive_classification,
    plot_interactive_comparison,
    plot_interactive_regression,
)

from ..dialogs.interactive_plot_dialog import InteractivePlotDialog
from .explanation_panel import ExplanationPanel
from .layout_utils import configure_splitter
from .metric_table import MetricTable
from .plot_canvas import PlotCanvas


class InteractiveExampleWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.model_name = ""
        self.task_type = "classification"
        self.focus_parameter: str | None = None
        self.input_widgets: dict[str, tuple[Any, Any]] = {}
        self.current_metrics_frame = pd.DataFrame()
        self.current_explanation_html = "<p>No interactive result is available yet.</p>"
        self.current_render_mode = "single"
        self.current_render_payload: Any = None
        self.plot_dialog: InteractivePlotDialog | None = None
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.setInterval(140)
        self.refresh_timer.timeout.connect(self.refresh_example)
        self.setObjectName("ContentPanel")

        self.heading = QLabel("Interactive Example")
        self.heading.setProperty("role", "sectionTitle")
        self.status_label = QLabel("Choose a model to activate the live example.")
        self.status_label.setWordWrap(True)
        self.scenario_combo = QComboBox()
        self.scenario_combo.currentTextChanged.connect(self.schedule_refresh)
        self.view_mode_combo = QComboBox()
        self.view_mode_combo.addItems(["Single Model", "Compare Models"])
        self.view_mode_combo.currentTextChanged.connect(self._comparison_mode_changed)
        self.comparison_model_combo = QComboBox()
        self.comparison_model_combo.currentTextChanged.connect(self.schedule_refresh)

        self.controls_form = QFormLayout()
        controls_container = QWidget()
        controls_container.setLayout(self.controls_form)
        controls_container.setObjectName("ControlPanel")
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(controls_container)
        controls_scroll.setMinimumWidth(250)

        self.metric_table = MetricTable()
        self.plot_canvas = PlotCanvas()
        self.explanation_panel = ExplanationPanel("Live Interpretation")
        self.open_large_view_button = QPushButton("Open Full Window")
        self.open_large_view_button.setProperty("variant", "secondary")
        self.open_large_view_button.setEnabled(False)
        self.open_large_view_button.clicked.connect(self._open_large_view)

        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)
        controls_layout.addWidget(self.heading)
        controls_layout.addWidget(self.status_label)
        controls_layout.addWidget(QLabel("Example scenario"))
        controls_layout.addWidget(self.scenario_combo)
        controls_layout.addWidget(QLabel("View mode"))
        controls_layout.addWidget(self.view_mode_combo)
        controls_layout.addWidget(QLabel("Comparison model"))
        controls_layout.addWidget(self.comparison_model_combo)
        controls_layout.addWidget(controls_scroll, stretch=1)

        metrics_panel = QWidget()
        metrics_panel.setObjectName("MetricsSection")
        metrics_layout = QVBoxLayout(metrics_panel)
        metrics_layout.setContentsMargins(14, 14, 14, 14)
        metrics_layout.setSpacing(10)
        metrics_layout.addWidget(QLabel("Quick metrics"))
        metrics_layout.addWidget(self.metric_table, stretch=1)

        plot_panel = QWidget()
        plot_panel.setObjectName("PlotSection")
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(14, 14, 14, 14)
        plot_layout.setSpacing(10)
        plot_button_row = QHBoxLayout()
        plot_button_row.addStretch(1)
        plot_button_row.addWidget(self.open_large_view_button)
        plot_layout.addLayout(plot_button_row)
        plot_layout.addWidget(self.plot_canvas, stretch=1)

        visual_splitter = QSplitter(Qt.Vertical)
        visual_splitter.addWidget(plot_panel)
        visual_splitter.addWidget(self.explanation_panel)
        configure_splitter(visual_splitter, [420, 220], stretch_factors=[1, 0])

        content_splitter = QSplitter(Qt.Vertical)
        content_splitter.addWidget(metrics_panel)
        content_splitter.addWidget(visual_splitter)
        configure_splitter(content_splitter, [220, 470], stretch_factors=[0, 1])

        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(controls_panel)
        main_splitter.addWidget(content_splitter)
        configure_splitter(main_splitter, [320, 900], stretch_factors=[0, 1])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.addWidget(main_splitter, stretch=1)

        self._comparison_mode_changed()

    def set_model(
        self,
        *,
        model_name: str,
        task_type: str,
        focus_parameter: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.task_type = task_type
        self.focus_parameter = focus_parameter
        self.heading.setText(f"Interactive Example: {model_name}")
        self._populate_scenarios()
        self._populate_comparison_models()
        self._build_controls()
        self.schedule_refresh()

    def set_focus_parameter(self, focus_parameter: str | None) -> None:
        self.focus_parameter = focus_parameter
        self.schedule_refresh()

    def schedule_refresh(self) -> None:
        if not self.model_name:
            return
        self.status_label.setText("Refreshing example...")
        self.refresh_timer.start()

    def _comparison_mode_changed(self) -> None:
        is_compare_mode = self.view_mode_combo.currentText() == "Compare Models"
        self.comparison_model_combo.setEnabled(is_compare_mode)
        self.schedule_refresh()

    def _populate_scenarios(self) -> None:
        current = self.scenario_combo.currentText()
        scenarios = interactive_scenarios(self.task_type)
        names = [scenario.name for scenario in scenarios]
        self.scenario_combo.blockSignals(True)
        self.scenario_combo.clear()
        self.scenario_combo.addItems(names)
        if current in names:
            self.scenario_combo.setCurrentText(current)
        self.scenario_combo.blockSignals(False)

    def _populate_comparison_models(self) -> None:
        current = self.comparison_model_combo.currentText()
        options = comparison_candidates(self.task_type, self.model_name)
        self.comparison_model_combo.blockSignals(True)
        self.comparison_model_combo.clear()
        self.comparison_model_combo.addItems(options)
        if current in options:
            self.comparison_model_combo.setCurrentText(current)
        elif options:
            self.comparison_model_combo.setCurrentText(suggested_comparison_model(self.task_type, self.model_name))
        self.comparison_model_combo.blockSignals(False)

    def _clear_controls(self) -> None:
        while self.controls_form.rowCount():
            self.controls_form.removeRow(0)
        self.input_widgets.clear()

    def _connect_widget(self, widget: QWidget) -> None:
        if isinstance(widget, QSpinBox):
            widget.valueChanged.connect(self.schedule_refresh)
        elif isinstance(widget, QDoubleSpinBox):
            widget.valueChanged.connect(self.schedule_refresh)
        elif isinstance(widget, QComboBox):
            widget.currentTextChanged.connect(self.schedule_refresh)
        elif isinstance(widget, QLineEdit):
            widget.textChanged.connect(self.schedule_refresh)

    def _build_controls(self) -> None:
        self._clear_controls()
        if not self.model_name:
            return
        spec = get_model_spec(self.model_name)
        parameter_names = default_demo_hyperparameters(self.model_name, limit=3)
        if self.focus_parameter and self.focus_parameter in spec.hyperparameters:
            parameter_names = [self.focus_parameter] + [name for name in parameter_names if name != self.focus_parameter]
        parameter_names = parameter_names[:3]

        if not parameter_names:
            message = QLabel(
                "This model has no major editable hyperparameters in the demo. The example still updates when you change the scenario or comparison model."
            )
            message.setWordWrap(True)
            self.controls_form.addRow(message)
            return

        for name in parameter_names:
            field_spec = spec.hyperparameters[name]
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
            widget.setToolTip(f"{field_spec.description}\n\n{field_spec.recommended}")
            self.controls_form.addRow(name, widget)
            self._connect_widget(widget)
            self.input_widgets[name] = (field_spec, widget)

    def _collect_hyperparameters(self) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, (_, widget) in self.input_widgets.items():
            if isinstance(widget, QSpinBox):
                params[name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                params[name] = widget.value()
            elif isinstance(widget, QComboBox):
                params[name] = widget.currentText()
            else:
                value = widget.text().strip()
                if value:
                    params[name] = value
        return params

    def refresh_example(self) -> None:
        if not self.model_name or not self.scenario_combo.currentText():
            return
        try:
            hyperparameters = self._collect_hyperparameters()
            if self.view_mode_combo.currentText() == "Compare Models" and self.comparison_model_combo.currentText():
                result = run_interactive_comparison(
                    task_type=self.task_type,
                    primary_model_name=self.model_name,
                    comparison_model_name=self.comparison_model_combo.currentText(),
                    scenario_name=self.scenario_combo.currentText(),
                    primary_hyperparameters=hyperparameters,
                )
                metrics_frame = interactive_metric_frame(result)
                explanation_html = build_interactive_comparison_explanation(
                    result,
                    focus_parameter=self.focus_parameter,
                )
                self._apply_view(
                    figure=plot_interactive_comparison(result),
                    metrics_frame=metrics_frame,
                    explanation_html=explanation_html,
                    render_mode="comparison",
                    payload=result,
                )
                self.status_label.setText(
                    "You are comparing two models on the same scenario and split. Change the active settings to see the gap move."
                )
                return

            result = run_interactive_example(
                task_type=self.task_type,
                model_name=self.model_name,
                scenario_name=self.scenario_combo.currentText(),
                hyperparameters=hyperparameters,
            )
            figure = (
                plot_interactive_classification(result)
                if self.task_type == "classification"
                else plot_interactive_regression(result)
            )
            metrics_frame = interactive_metric_frame(result)
            explanation_html = build_interactive_explanation(result, focus_parameter=self.focus_parameter)
            self._apply_view(
                figure=figure,
                metrics_frame=metrics_frame,
                explanation_html=explanation_html,
                render_mode="single",
                payload=result,
            )
            self.status_label.setText(
                "Adjust the controls above. The metrics, plot, and interpretation update automatically."
            )
        except Exception as exc:
            self.metric_table.setRowCount(0)
            self.metric_table.setColumnCount(0)
            self.plot_canvas.show_placeholder("The interactive example could not be rendered.")
            self.explanation_panel.set_html(f"<p><b>Example error:</b> {exc}</p>")
            self.open_large_view_button.setEnabled(False)
            self.current_metrics_frame = pd.DataFrame()
            self.current_explanation_html = f"<p><b>Example error:</b> {exc}</p>"
            self.current_render_payload = None
            self.status_label.setText("The current setting caused an example error.")

    def _apply_view(
        self,
        *,
        figure,
        metrics_frame: pd.DataFrame,
        explanation_html: str,
        render_mode: str,
        payload: Any,
    ) -> None:
        self.plot_canvas.set_figure(figure)
        self.metric_table.set_dataframe(metrics_frame)
        self.explanation_panel.set_html(explanation_html)
        self.current_metrics_frame = metrics_frame.copy()
        self.current_explanation_html = explanation_html
        self.current_render_mode = render_mode
        self.current_render_payload = payload
        self.open_large_view_button.setEnabled(True)

    def _open_large_view(self) -> None:
        if self.current_render_payload is None or self.current_metrics_frame.empty:
            return
        if self.plot_dialog is None:
            self.plot_dialog = InteractivePlotDialog(self.heading.text(), self)
        self.plot_dialog.setWindowTitle(f"{self.heading.text()} - Full Window")
        self.plot_dialog.set_content(
            figure=self._build_large_figure(),
            metrics_frame=self.current_metrics_frame,
            explanation_html=self.current_explanation_html,
        )
        self.plot_dialog.show()
        self.plot_dialog.raise_()
        self.plot_dialog.activateWindow()

    def _build_large_figure(self):
        if self.current_render_mode == "comparison":
            return plot_interactive_comparison(self.current_render_payload, size_scale=1.28)
        if self.task_type == "classification":
            return plot_interactive_classification(self.current_render_payload, size=(9, 7))
        return plot_interactive_regression(self.current_render_payload, size=(9, 6.5))
