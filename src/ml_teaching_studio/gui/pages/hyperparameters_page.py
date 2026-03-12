"""Hyperparameter teaching page."""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QFormLayout, QSplitter, QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget

from ml_teaching_studio.educational.hyperparameter_help import get_hyperparameter_help
from ml_teaching_studio.gui.widgets.interactive_example_widget import InteractiveExampleWidget
from ml_teaching_studio.gui.widgets.layout_utils import configure_splitter
from ml_teaching_studio.models.model_registry import get_model_spec, list_model_names


class HyperparametersPage(QWidget):
    def __init__(self, studio, parent=None) -> None:
        super().__init__(parent)
        self.studio = studio
        self.model_combo = QComboBox()
        self.task_combo = QComboBox()
        self.task_combo.addItems(["classification", "regression"])
        self.table = QTableWidget()
        self.detail_browser = QTextBrowser()
        self.interactive_demo = InteractiveExampleWidget()

        controls_panel = QWidget()
        controls_panel.setObjectName("ContentPanel")
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(14, 14, 14, 14)
        controls_layout.setSpacing(10)
        controls_form = QFormLayout()
        controls_form.addRow("Task type", self.task_combo)
        controls_form.addRow("Model", self.model_combo)
        controls_layout.addLayout(controls_form)
        controls_layout.addStretch(1)

        table_panel = QWidget()
        table_panel.setObjectName("MetricsSection")
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(14, 14, 14, 14)
        table_layout.setSpacing(10)
        table_layout.addWidget(self.table, stretch=1)

        left_panel = QWidget()
        left_panel.setObjectName("ControlPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(14, 14, 14, 14)

        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.addWidget(controls_panel)
        left_splitter.addWidget(table_panel)
        configure_splitter(left_splitter, [120, 620], stretch_factors=[0, 1])
        left_layout.addWidget(left_splitter, stretch=1)

        right_panel = QWidget()
        right_panel.setObjectName("ContentPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)
        right_layout.addWidget(self.detail_browser, stretch=1)

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(right_panel)
        right_splitter.addWidget(self.interactive_demo)
        configure_splitter(right_splitter, [380, 560], stretch_factors=[1, 1])

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_splitter)
        configure_splitter(splitter, [420, 1020], stretch_factors=[0, 1])

        layout = QVBoxLayout(self)
        layout.addWidget(splitter, stretch=1)
        layout.setContentsMargins(0, 0, 0, 0)

        self.task_combo.currentTextChanged.connect(self._populate_models)
        self.model_combo.currentTextChanged.connect(self._populate_table)
        self.table.itemSelectionChanged.connect(self._show_detail)
        self.task_combo.setCurrentText(self.studio.selected_task_type)
        self._populate_models()

    def _populate_models(self) -> None:
        self.model_combo.clear()
        self.model_combo.addItems(list_model_names(self.task_combo.currentText()))
        preferred = self.studio.preferred_model_name
        if preferred in list_model_names(self.task_combo.currentText()):
            self.model_combo.setCurrentText(preferred)
        self._populate_table()
        if self.model_combo.currentText():
            self.interactive_demo.set_model(
                model_name=self.model_combo.currentText(),
                task_type=self.task_combo.currentText(),
            )

    def _populate_table(self) -> None:
        model_name = self.model_combo.currentText()
        if not model_name:
            return
        spec = get_model_spec(model_name)
        help_map = get_hyperparameter_help(model_name)
        rows = []
        for name, field_spec in spec.hyperparameters.items():
            help_entry = help_map.get(name, {})
            rows.append(
                {
                    "parameter": name,
                    "default": field_spec.default,
                    "typical range": help_entry.get("typical_range", ""),
                    "role": help_entry.get("plain_language", field_spec.description),
                }
            )
        frame = pd.DataFrame(rows)
        self.table.setRowCount(len(frame.index))
        self.table.setColumnCount(len(frame.columns))
        self.table.setHorizontalHeaderLabels(list(frame.columns))
        for row_index, (_, row) in enumerate(frame.iterrows()):
            for column_index, value in enumerate(row):
                self.table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()
        if frame.shape[0]:
            self.table.selectRow(0)
        self.interactive_demo.set_model(
            model_name=model_name,
            task_type=spec.task_type,
            focus_parameter=frame.iloc[0]["parameter"] if not frame.empty else None,
        )

    def _show_detail(self) -> None:
        model_name = self.model_combo.currentText()
        selected_items = self.table.selectedItems()
        if not model_name or not selected_items:
            return
        parameter_name = selected_items[0].text()
        help_entry = get_hyperparameter_help(model_name).get(parameter_name)
        if not help_entry:
            self.detail_browser.setHtml("<p>No teaching note is available for this parameter yet.</p>")
            self.interactive_demo.set_focus_parameter(parameter_name)
            return
        self.detail_browser.setHtml(
            f"""
            <h2>{model_name}: {parameter_name}</h2>
            <p><b>Plain language:</b> {help_entry['plain_language']}</p>
            <p><b>Algorithmic role:</b> {help_entry['algorithmic_role']}</p>
            <p><b>Typical range:</b> {help_entry['typical_range']}</p>
            <p><b>If too small:</b> {help_entry['too_small']}</p>
            <p><b>If too large:</b> {help_entry['too_large']}</p>
            <p><b>Effect on bias / variance:</b> {help_entry['impact']}</p>
            <p><b>Practical tip:</b> {help_entry['tip']}</p>
            """
        )
        self.interactive_demo.set_focus_parameter(parameter_name)

    def on_dataset_changed(self) -> None:
        self.task_combo.setCurrentText(self.studio.selected_task_type)

    def on_model_changed(self) -> None:
        self._populate_models()
