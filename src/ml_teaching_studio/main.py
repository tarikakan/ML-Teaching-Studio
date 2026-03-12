"""Application entry point."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ml_teaching_studio.gui.main_window import MainWindow
from ml_teaching_studio.utils.config import APP_NAME, APP_STYLESHEET
from ml_teaching_studio.utils.logging_utils import configure_logging


def main() -> int:
    configure_logging()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
