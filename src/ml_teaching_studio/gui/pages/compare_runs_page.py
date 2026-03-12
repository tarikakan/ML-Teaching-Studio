"""Saved run comparison page."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QFileDialog,
    QHBoxLayout,
    QPushButton,
    QComboBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from ml_teaching_studio.core.explainers import explain_run_comparison
from ml_teaching_studio.gui.dialogs.run_details_dialog import RunDetailsDialog
from ml_teaching_studio.gui.widgets.explanation_panel import ExplanationPanel
from ml_teaching_studio.gui.widgets.layout_utils import configure_splitter
from ml_teaching_studio.gui.widgets.metric_table import MetricTable
from ml_teaching_studio.gui.widgets.plot_canvas import PlotCanvas
from ml_teaching_studio.plotting.comparison_plots import (
    comparison_table,
    plot_inference_time,
    plot_metric_bars,
    plot_training_time,
)
from ml_teaching_studio.utils.io import export_dataframe_csv


class CompareRunsPage(QWidget):
    def __init__(self, studio, parent=None) -> None:
        super().__init__(parent)
        self.studio = studio
        self.records = []
        self.frame = None

        self.table = MetricTable()
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.plot_combo = QComboBox()
        self.plot_combo.addItems(["Metric Bars", "Training Time", "Inference Time"])
        self.metric_combo = QComboBox()
        self.plot_canvas = PlotCanvas()
        self.explanation_panel = ExplanationPanel("Comparison Interpretation")
        self.refresh_button = QPushButton("Refresh")
        self.details_button = QPushButton("View Details")
        self.delete_button = QPushButton("Delete Selected")
        self.export_button = QPushButton("Export Summary CSV")

        button_row = QHBoxLayout()
        button_row.addWidget(self.refresh_button)
        button_row.addWidget(self.details_button)
        button_row.addWidget(self.delete_button)
        button_row.addWidget(self.export_button)

        controls_panel = QWidget()
        controls_panel.setObjectName("ControlPanel")
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(14, 14, 14, 14)
        controls_layout.setSpacing(10)
        controls_layout.addLayout(button_row)

        filter_form = QFormLayout()
        filter_form.addRow("Comparison plot", self.plot_combo)
        filter_form.addRow("Primary metric", self.metric_combo)
        controls_layout.addLayout(filter_form)

        table_panel = QWidget()
        table_panel.setObjectName("MetricsSection")
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(14, 14, 14, 14)
        table_layout.setSpacing(10)
        table_layout.addWidget(self.table, stretch=1)

        plot_panel = QWidget()
        plot_panel.setObjectName("PlotSection")
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(14, 14, 14, 14)
        plot_layout.setSpacing(10)
        plot_layout.addWidget(self.plot_canvas, stretch=1)

        analysis_splitter = QSplitter(Qt.Vertical)
        analysis_splitter.addWidget(plot_panel)
        analysis_splitter.addWidget(self.explanation_panel)
        configure_splitter(analysis_splitter, [520, 220], stretch_factors=[1, 0])

        content_splitter = QSplitter(Qt.Vertical)
        content_splitter.addWidget(table_panel)
        content_splitter.addWidget(analysis_splitter)
        configure_splitter(content_splitter, [420, 460], stretch_factors=[1, 1])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(controls_panel)
        layout.addWidget(content_splitter, stretch=1)

        self.refresh_button.clicked.connect(self.refresh)
        self.details_button.clicked.connect(self._show_details)
        self.delete_button.clicked.connect(self._delete_selected)
        self.export_button.clicked.connect(self._export_csv)
        self.plot_combo.currentTextChanged.connect(self._render_plot)
        self.metric_combo.currentTextChanged.connect(self._render_plot)
        self.table.itemSelectionChanged.connect(self._selection_changed)

        self.refresh()

    def _selected_records(self):
        selected_rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        if not selected_rows:
            return self.records
        return [self.records[row] for row in selected_rows if row < len(self.records)]

    def _available_metrics(self) -> list[str]:
        metrics = set()
        for record in self._selected_records() or self.records:
            metrics.update(record.get("test_metrics", {}).keys())
        return sorted(metrics)

    def refresh(self) -> None:
        self.records = self.studio.run_store.list_runs()
        self.frame = comparison_table(self.records)
        self.table.set_dataframe(self.frame if self.frame is not None else comparison_table([]))
        current_metric = self.metric_combo.currentText()
        self.metric_combo.clear()
        self.metric_combo.addItems(self._available_metrics())
        if current_metric in self._available_metrics():
            self.metric_combo.setCurrentText(current_metric)
        self._render_plot()

    def _selection_changed(self) -> None:
        metrics = self._available_metrics()
        current = self.metric_combo.currentText()
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        self.metric_combo.addItems(metrics)
        if current in metrics:
            self.metric_combo.setCurrentText(current)
        self.metric_combo.blockSignals(False)
        self._render_plot()

    def _render_plot(self) -> None:
        records = self._selected_records()
        if self.plot_combo.currentText() == "Metric Bars":
            metric_name = self.metric_combo.currentText() or "accuracy"
            figure = plot_metric_bars(records, metric_name)
        elif self.plot_combo.currentText() == "Training Time":
            figure = plot_training_time(records)
        else:
            figure = plot_inference_time(records)
        self.plot_canvas.set_figure(figure)
        self.explanation_panel.set_html(explain_run_comparison(records))

    def _show_details(self) -> None:
        records = self._selected_records()
        if not records:
            return
        dialog = RunDetailsDialog(records[0], self)
        dialog.exec()

    def _delete_selected(self) -> None:
        for record in self._selected_records():
            self.studio.run_store.delete_run(record["run_id"])
        self.studio.refresh_pages()

    def _export_csv(self) -> None:
        if self.frame is None or self.frame.empty:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Summary", "ml_teaching_studio_runs.csv", "CSV Files (*.csv)")
        if path:
            export_dataframe_csv(self.frame, path)
            self.studio.statusBar().showMessage(f"Exported run summary to {path}", 4000)

    def on_run_store_changed(self) -> None:
        self.refresh()
