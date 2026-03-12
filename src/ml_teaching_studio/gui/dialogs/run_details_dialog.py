"""Dialog for viewing saved run details."""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QTextBrowser, QVBoxLayout

from ml_teaching_studio.utils.helpers import safe_json_dumps


class RunDetailsDialog(QDialog):
    def __init__(self, run_record: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Run Details")
        self.resize(760, 560)
        browser = QTextBrowser()
        browser.setPlainText(safe_json_dumps(run_record))

        layout = QVBoxLayout(self)
        layout.addWidget(browser)
