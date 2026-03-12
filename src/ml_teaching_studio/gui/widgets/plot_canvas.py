"""Matplotlib canvas widget for PySide6."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget


class PlotCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PlotCanvas")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(320)

        self.figure = Figure(figsize=(6, 4), constrained_layout=True)
        self.canvas: FigureCanvasQTAgg | None = None
        self.toolbar: NavigationToolbar2QT | None = None
        self.placeholder = QLabel("No plot selected yet.")
        self.placeholder.setWordWrap(True)
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setObjectName("PlotPlaceholder")

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setSpacing(10)
        self._install_canvas(self.figure)
        self._layout.addWidget(self.placeholder, stretch=1)
        self.show_placeholder("No plot selected yet.")

    def set_figure(self, figure: Figure) -> None:
        try:
            figure.set_constrained_layout(True)
        except Exception:
            pass
        self._replace_canvas(figure)
        assert self.toolbar is not None
        assert self.canvas is not None
        self.toolbar.show()
        self.canvas.show()
        self.canvas.draw()
        self.placeholder.hide()

    def show_placeholder(self, message: str) -> None:
        self.placeholder.setText(message)
        self.placeholder.show()
        if self.toolbar is not None:
            self.toolbar.hide()
        if self.canvas is not None:
            self.canvas.hide()

    def export_current_figure(self, path: str | Path) -> None:
        if self.figure is not None:
            self.figure.savefig(path, bbox_inches="tight", dpi=160)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.canvas is not None and self.canvas.isVisible():
            self.canvas.draw_idle()

    def _install_canvas(self, figure: Figure) -> None:
        self.figure = figure
        self.canvas = FigureCanvasQTAgg(figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.updateGeometry()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self._layout.insertWidget(0, self.toolbar)
        self._layout.insertWidget(1, self.canvas, stretch=1)

    def _replace_canvas(self, figure: Figure) -> None:
        old_figure = self.figure
        if self.toolbar is not None:
            self._layout.removeWidget(self.toolbar)
            self.toolbar.deleteLater()
            self.toolbar = None
        if self.canvas is not None:
            self._layout.removeWidget(self.canvas)
            self.canvas.setParent(None)
            self.canvas.deleteLater()
            self.canvas = None
        if old_figure is not None and old_figure is not figure:
            plt.close(old_figure)
        self._install_canvas(figure)
