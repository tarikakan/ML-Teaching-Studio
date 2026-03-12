"""Dialog for enlarged interactive example views."""

from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt
from matplotlib.figure import Figure

from ml_teaching_studio.gui.widgets.explanation_panel import ExplanationPanel
from ml_teaching_studio.gui.widgets.layout_utils import configure_splitter
from ml_teaching_studio.gui.widgets.metric_table import MetricTable
from ml_teaching_studio.gui.widgets.plot_canvas import PlotCanvas


class InteractivePlotDialog(QDialog):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1380, 920)

        self.metric_table = MetricTable()
        self.plot_canvas = PlotCanvas()
        self.explanation_panel = ExplanationPanel("Interactive Interpretation")
        self.fullscreen_button = QPushButton("Toggle Fullscreen")
        self.close_button = QPushButton("Close")

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self.fullscreen_button)
        button_row.addWidget(self.close_button)

        metrics_panel = QWidget()
        metrics_panel.setObjectName("MetricsSection")
        metrics_layout = QVBoxLayout(metrics_panel)
        metrics_layout.setContentsMargins(14, 14, 14, 14)
        metrics_layout.setSpacing(10)
        metrics_layout.addWidget(self.metric_table, stretch=1)

        plot_panel = QWidget()
        plot_panel.setObjectName("PlotSection")
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(14, 14, 14, 14)
        plot_layout.setSpacing(10)
        plot_layout.addWidget(self.plot_canvas, stretch=1)

        visual_splitter = QSplitter(Qt.Vertical)
        visual_splitter.addWidget(plot_panel)
        visual_splitter.addWidget(self.explanation_panel)
        configure_splitter(visual_splitter, [620, 220], stretch_factors=[1, 0])

        content_splitter = QSplitter(Qt.Vertical)
        content_splitter.addWidget(metrics_panel)
        content_splitter.addWidget(visual_splitter)
        configure_splitter(content_splitter, [230, 610], stretch_factors=[0, 1])

        layout = QVBoxLayout(self)
        layout.addLayout(button_row)
        layout.addWidget(content_splitter, stretch=1)

        self.fullscreen_button.clicked.connect(self._toggle_fullscreen)
        self.close_button.clicked.connect(self.close)

    def set_content(self, *, figure: Figure, metrics_frame: pd.DataFrame, explanation_html: str) -> None:
        self.metric_table.set_dataframe(metrics_frame)
        self.plot_canvas.set_figure(figure)
        self.explanation_panel.set_html(explanation_html)

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()
