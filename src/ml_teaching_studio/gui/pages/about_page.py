"""About page."""

from __future__ import annotations

from PySide6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget

from ml_teaching_studio.utils.config import APP_NAME, APP_VERSION, AUTHOR_NAME, COPYRIGHT_NOTICE, LICENSE_NAME


class AboutPage(QWidget):
    def __init__(self, studio, parent=None) -> None:
        super().__init__(parent)
        del studio
        browser = QTextBrowser()
        browser.setHtml(
            f"""
            <h1>{APP_NAME}</h1>
            <p><b>Version:</b> {APP_VERSION}</p>
            <p><b>Purpose:</b> Educational machine learning studio for students, self-learners, and instructors.</p>
            <p><b>Author:</b> {AUTHOR_NAME}</p>
            <p><b>Copyright:</b> {COPYRIGHT_NOTICE}</p>
            <p><b>License:</b> {LICENSE_NAME}</p>
            <p>This application is designed to explain not only what a model scored, but why it behaved that way.</p>
            """
        )
        layout = QVBoxLayout(self)
        layout.addWidget(browser)
