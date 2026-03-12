"""Shared layout helpers for resizable page composition."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QScrollArea, QSizePolicy, QSplitter, QWidget


def wrap_in_scroll_area(widget: QWidget, *, object_name: str = "PanelScroll") -> QScrollArea:
    """Wrap a widget in a borderless, resizable scroll area."""

    area = QScrollArea()
    area.setObjectName(object_name)
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    area.setWidget(widget)
    return area


def configure_splitter(
    splitter: QSplitter,
    sizes: Sequence[int],
    *,
    stretch_factors: Sequence[int] | None = None,
) -> QSplitter:
    """Apply consistent behavior to adjustable splitters across the app."""

    splitter.setChildrenCollapsible(False)
    splitter.setOpaqueResize(False)
    splitter.setHandleWidth(10)
    for index in range(splitter.count()):
        splitter.setCollapsible(index, False)
        if stretch_factors and index < len(stretch_factors):
            splitter.setStretchFactor(index, stretch_factors[index])
    splitter.setSizes(list(sizes))
    return splitter
