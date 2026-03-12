"""Reusable table widget for metrics and comparisons."""

from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import QAbstractItemView, QHeaderView, QSizePolicy, QTableWidget, QTableWidgetItem

from ml_teaching_studio.utils.helpers import format_metric


class MetricTable(QTableWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(180)
        self.setAlternatingRowColors(True)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setWordWrap(False)
        self.setSortingEnabled(True)
        self.verticalHeader().setVisible(False)
        self.horizontalHeader().setStretchLastSection(True)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

    def set_metrics(self, metrics: dict[str, object]) -> None:
        self.setSortingEnabled(False)
        self.setRowCount(len(metrics))
        self.setColumnCount(2)
        self.setHorizontalHeaderLabels(["Metric", "Value"])
        for row, (name, value) in enumerate(metrics.items()):
            self.setItem(row, 0, QTableWidgetItem(name))
            self.setItem(row, 1, QTableWidgetItem(format_metric(value)))
        self.resizeColumnsToContents()
        self.setSortingEnabled(True)

    def set_dataframe(self, frame: pd.DataFrame) -> None:
        self.setSortingEnabled(False)
        self.setRowCount(len(frame.index))
        self.setColumnCount(len(frame.columns))
        self.setHorizontalHeaderLabels([str(column) for column in frame.columns])
        for row_index, (_, row) in enumerate(frame.iterrows()):
            for column_index, value in enumerate(row):
                self.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        self.resizeColumnsToContents()
        self.setSortingEnabled(True)
