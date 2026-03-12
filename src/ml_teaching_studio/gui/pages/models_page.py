"""Model browser page."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QLabel, QListWidget, QPushButton, QSplitter, QTextBrowser, QVBoxLayout, QWidget

from ml_teaching_studio.educational.model_help import get_model_help
from ml_teaching_studio.gui.widgets.interactive_example_widget import InteractiveExampleWidget
from ml_teaching_studio.gui.widgets.layout_utils import configure_splitter
from ml_teaching_studio.models.model_registry import get_model_spec, list_models
from ml_teaching_studio.utils.helpers import list_to_html, make_html_card


class ModelsPage(QWidget):
    def __init__(self, studio, parent=None) -> None:
        super().__init__(parent)
        self.studio = studio
        self.task_combo = QComboBox()
        self.task_combo.addItems(["classification", "regression"])
        self.model_list = QListWidget()
        self.detail_browser = QTextBrowser()
        self.interactive_demo = InteractiveExampleWidget()
        self.use_button = QPushButton("Use This Model in Training")
        self.use_button.clicked.connect(self._use_model)

        left_panel = QWidget()
        left_panel.setObjectName("ControlPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)
        left_layout.addWidget(QLabel("Task"))
        left_layout.addWidget(self.task_combo)
        left_layout.addWidget(QLabel("Models"))
        left_layout.addWidget(self.model_list, stretch=1)
        left_layout.addWidget(self.use_button)

        detail_panel = QWidget()
        detail_panel.setObjectName("ContentPanel")
        right_layout = QVBoxLayout(detail_panel)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)
        right_layout.addWidget(self.detail_browser, stretch=1)

        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.addWidget(detail_panel)
        right_splitter.addWidget(self.interactive_demo)
        configure_splitter(right_splitter, [420, 520], stretch_factors=[1, 1])

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_splitter)
        configure_splitter(splitter, [300, 1140], stretch_factors=[0, 1])

        layout = QVBoxLayout(self)
        layout.addWidget(splitter, stretch=1)
        layout.setContentsMargins(0, 0, 0, 0)

        self.task_combo.currentTextChanged.connect(self._populate_models)
        self.model_list.currentRowChanged.connect(self._show_model)
        self.task_combo.setCurrentText(self.studio.selected_task_type)
        self._populate_models()

    def _populate_models(self) -> None:
        self.model_list.clear()
        for spec in list_models(self.task_combo.currentText()):
            self.model_list.addItem(spec.name)
        self.model_list.setCurrentRow(0)

    def _show_model(self, row: int) -> None:
        if row < 0:
            return
        model_name = self.model_list.item(row).text()
        help_data = get_model_help(model_name)
        spec = get_model_spec(model_name)
        html = make_html_card(
            model_name,
            f"""
            <p><b>Plain-language explanation:</b> {help_data['explanation']}</p>
            <p><b>Mathematical intuition:</b> {help_data['intuition']}</p>
            <p><b>Assumptions</b></p><ul>{list_to_html(help_data['assumptions'])}</ul>
            <p><b>Strengths</b></p><ul>{list_to_html(help_data['strengths'])}</ul>
            <p><b>Weaknesses</b></p><ul>{list_to_html(help_data['weaknesses'])}</ul>
            <p><b>Common use cases</b></p><ul>{list_to_html(help_data['use_cases'])}</ul>
            <p><b>Limitations</b></p><ul>{list_to_html(help_data['limitations'])}</ul>
            <p><b>Beginner mistakes</b></p><ul>{list_to_html(help_data['beginner_mistakes'])}</ul>
            <p><b>Example visuals</b></p><ul>{list_to_html(help_data['visuals'])}</ul>
            <p><b>Important hyperparameters:</b> {", ".join(spec.important_hyperparameters) or "None"}</p>
            """,
            subtitle=spec.description,
        )
        self.detail_browser.setHtml(html)
        self.interactive_demo.set_model(model_name=model_name, task_type=spec.task_type)

    def _use_model(self) -> None:
        current_item = self.model_list.currentItem()
        if not current_item:
            return
        self.studio.set_preferred_model(current_item.text())
        self.studio.navigate("Training")

    def on_dataset_changed(self) -> None:
        self.task_combo.setCurrentText(self.studio.selected_task_type)
