"""Theory lessons viewer."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QListWidget, QListWidgetItem, QPushButton, QSplitter, QTextBrowser, QVBoxLayout, QWidget

from ml_teaching_studio.educational.lessons import LESSONS
from ml_teaching_studio.gui.widgets.layout_utils import configure_splitter


class TheoryPage(QWidget):
    def __init__(self, studio, parent=None) -> None:
        super().__init__(parent)
        self.studio = studio
        self.lesson_list = QListWidget()
        self.lesson_browser = QTextBrowser()
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.try_button = QPushButton("Open Related Tool")
        self.try_button.clicked.connect(self._open_action)

        left_panel = QWidget()
        left_panel.setObjectName("ControlPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)
        left_layout.addWidget(QLabel("Lessons"))
        left_layout.addWidget(self.summary_label)
        left_layout.addWidget(self.lesson_list, stretch=1)
        left_layout.addWidget(self.try_button)

        right_panel = QWidget()
        right_panel.setObjectName("ContentPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.addWidget(self.lesson_browser, stretch=1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        configure_splitter(splitter, [320, 1120], stretch_factors=[0, 1])

        layout = QVBoxLayout(self)
        layout.addWidget(splitter, stretch=1)
        layout.setContentsMargins(0, 0, 0, 0)

        for lesson in LESSONS:
            item = QListWidgetItem(lesson.title)
            item.setToolTip(lesson.summary)
            self.lesson_list.addItem(item)
        self.lesson_list.currentRowChanged.connect(self._show_lesson)
        self.lesson_list.setCurrentRow(0)

    def _show_lesson(self, row: int) -> None:
        if row < 0:
            return
        lesson = LESSONS[row]
        self.summary_label.setText(lesson.summary)
        self.lesson_browser.setHtml(lesson.html)
        self.try_button.setText(lesson.action_label or "Open Related Tool")
        self.try_button.setEnabled(bool(lesson.action_page))

    def _open_action(self) -> None:
        lesson = LESSONS[self.lesson_list.currentRow()]
        if lesson.action_page:
            self.studio.navigate(lesson.action_page)
