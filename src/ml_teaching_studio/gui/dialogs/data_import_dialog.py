"""Dialog for CSV import settings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ml_teaching_studio.utils.helpers import infer_task_type


class DataImportDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import CSV Dataset")
        self.resize(760, 520)
        self._dataframe: pd.DataFrame | None = None

        self.path_edit = QLineEdit()
        self.browse_button = QPushButton("Browse")
        self.target_combo = QComboBox()
        self.task_combo = QComboBox()
        self.task_combo.addItems(["classification", "regression"])
        self.preview_table = QTableWidget()
        self.status_label = QLabel("Select a CSV file to preview its columns.")
        self.ok_button = QPushButton("Import")
        self.cancel_button = QPushButton("Cancel")

        top_row = QHBoxLayout()
        top_row.addWidget(self.path_edit, stretch=1)
        top_row.addWidget(self.browse_button)

        form = QFormLayout()
        form.addRow("CSV file", top_row)
        form.addRow("Target column", self.target_combo)
        form.addRow("Task type", self.task_combo)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.ok_button)
        button_row.addWidget(self.cancel_button)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.status_label)
        layout.addWidget(self.preview_table, stretch=1)
        layout.addLayout(button_row)

        self.browse_button.clicked.connect(self.choose_file)
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        self.target_combo.currentTextChanged.connect(self._update_task_guess)

    def choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose CSV file",
            "",
            "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            self.path_edit.setText(path)
            self.load_preview(Path(path))

    def load_preview(self, path: Path) -> None:
        self._dataframe = pd.read_csv(path)
        if self._dataframe.empty:
            self.status_label.setText("The selected file is empty.")
            return
        self.status_label.setText(f"Loaded {len(self._dataframe)} rows and {len(self._dataframe.columns)} columns.")
        self.target_combo.clear()
        self.target_combo.addItems([str(column) for column in self._dataframe.columns])
        self.target_combo.setCurrentText(str(self._dataframe.columns[-1]))
        self._update_task_guess()
        preview = self._dataframe.head(15)
        self.preview_table.setRowCount(len(preview.index))
        self.preview_table.setColumnCount(len(preview.columns))
        self.preview_table.setHorizontalHeaderLabels([str(column) for column in preview.columns])
        for row_index, (_, row) in enumerate(preview.iterrows()):
            for column_index, value in enumerate(row):
                self.preview_table.setItem(row_index, column_index, QTableWidgetItem(str(value)))
        self.preview_table.resizeColumnsToContents()

    def _update_task_guess(self) -> None:
        if self._dataframe is None or not self.target_combo.currentText():
            return
        inferred = infer_task_type(self._dataframe[self.target_combo.currentText()])
        self.task_combo.setCurrentText(inferred)

    def selected_settings(self) -> dict[str, str]:
        return {
            "path": self.path_edit.text().strip(),
            "target_column": self.target_combo.currentText(),
            "task_type": self.task_combo.currentText(),
        }
