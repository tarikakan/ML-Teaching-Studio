"""Reusable explanation panel."""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QSizePolicy, QTextBrowser, QVBoxLayout, QWidget


class ExplanationPanel(QWidget):
    def __init__(self, title: str = "Interpretation", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Card")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(180)
        self.title_label = QLabel(title)
        self.title_label.setProperty("role", "sectionTitle")
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        self.browser.document().setDocumentMargin(10)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)
        layout.addWidget(self.title_label)
        layout.addWidget(self.browser, stretch=1)

    def set_html(self, html: str) -> None:
        self.browser.setHtml(html)

    def set_text(self, text: str) -> None:
        self.browser.setPlainText(text)
