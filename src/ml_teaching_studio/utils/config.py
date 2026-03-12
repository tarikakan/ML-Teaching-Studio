"""Application configuration values."""

from __future__ import annotations

from pathlib import Path

APP_NAME = "ML-Teaching Studio"
APP_SLUG = "ml_teaching_studio"
APP_VERSION = "1.0.0"
AUTHOR_NAME = "Tarik Akan"
LICENSE_NAME = "MIT"
COPYRIGHT_NOTICE = "Copyright (c) 2026 Tarik Akan"

DEFAULT_WINDOW_WIDTH = 1580
DEFAULT_WINDOW_HEIGHT = 980
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_SEED = 42
DEFAULT_CV_FOLDS = 5
MAX_DATA_PREVIEW_ROWS = 150

APP_DATA_DIR = Path.home() / ".ml_teaching_studio"
RUN_STORE_PATH = APP_DATA_DIR / "runs.json"
EXPORT_DIR = APP_DATA_DIR / "exports"

COLOR_PALETTE = {
    "background": "#f4f5f7",
    "surface": "#ffffff",
    "surface_alt": "#eef1f5",
    "surface_alt_2": "#e2e8f0",
    "primary": "#166088",
    "primary_soft": "#d8ecf5",
    "secondary": "#355070",
    "success": "#2a7f62",
    "warning": "#b7791f",
    "danger": "#b83232",
    "text": "#1f2933",
    "muted_text": "#52606d",
    "border": "#d9e2ec",
    "accent": "#de9151",
}

APP_STYLESHEET = f"""
QMainWindow, QWidget {{
    background: {COLOR_PALETTE["background"]};
    color: {COLOR_PALETTE["text"]};
    font-family: "Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif;
    font-size: 13px;
}}
QWidget#PageShell, QWidget#SidebarPanel, QWidget#ControlPanel, QWidget#ResultsPanel, QWidget#PlotSection, QWidget#MetricsSection, QWidget#ContentPanel, QFrame#Card, QGroupBox, QWidget#PlotCanvas {{
    background: {COLOR_PALETTE["surface"]};
    border: 1px solid {COLOR_PALETTE["border"]};
    border-radius: 12px;
}}
QWidget#PageShell, QWidget#SidebarPanel, QWidget#ControlPanel, QWidget#ResultsPanel, QWidget#PlotSection, QWidget#MetricsSection, QWidget#ContentPanel, QWidget#PlotCanvas {{
    padding: 0px;
}}
QGroupBox {{
    margin-top: 12px;
    padding: 14px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
}}
QPushButton {{
    background: {COLOR_PALETTE["primary"]};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 9px 14px;
    font-weight: 600;
}}
QPushButton:hover {{
    background: {COLOR_PALETTE["secondary"]};
}}
QPushButton:disabled {{
    background: {COLOR_PALETTE["surface_alt_2"]};
    color: {COLOR_PALETTE["muted_text"]};
}}
QPushButton[variant="secondary"] {{
    background: {COLOR_PALETTE["surface_alt"]};
    color: {COLOR_PALETTE["text"]};
    border: 1px solid {COLOR_PALETTE["border"]};
}}
QLineEdit, QTextEdit, QTextBrowser, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QListWidget, QTableWidget {{
    background: {COLOR_PALETTE["surface"]};
    border: 1px solid {COLOR_PALETTE["border"]};
    border-radius: 8px;
    padding: 6px 8px;
    selection-background-color: {COLOR_PALETTE["primary"]};
}}
QTextBrowser, QPlainTextEdit, QTextEdit, QTableWidget {{
    padding: 8px;
}}
QComboBox {{
    min-height: 18px;
}}
QComboBox::drop-down {{
    border: none;
    width: 28px;
}}
QLineEdit:focus, QTextEdit:focus, QTextBrowser:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QListWidget:focus, QTableWidget:focus {{
    border: 1px solid {COLOR_PALETTE["primary"]};
}}
QTabWidget::pane, QStackedWidget {{
    border: none;
}}
QTabBar::tab {{
    background: {COLOR_PALETTE["surface_alt"]};
    border: 1px solid {COLOR_PALETTE["border"]};
    border-bottom: none;
    padding: 8px 14px;
    margin-right: 4px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
}}
QTabBar::tab:selected {{
    background: {COLOR_PALETTE["surface"]};
    color: {COLOR_PALETTE["primary"]};
}}
QHeaderView::section {{
    background: {COLOR_PALETTE["surface_alt"]};
    color: {COLOR_PALETTE["text"]};
    padding: 6px;
    border: none;
    border-bottom: 1px solid {COLOR_PALETTE["border"]};
}}
QLabel[role="sidebarTitle"] {{
    font-size: 17px;
    font-weight: 700;
    color: {COLOR_PALETTE["secondary"]};
    padding: 4px 0 10px 0;
}}
QLabel[role="sectionTitle"] {{
    font-size: 15px;
    font-weight: 700;
}}
QListWidget#Sidebar {{
    background: {COLOR_PALETTE["surface"]};
    border: none;
    padding: 6px 0;
}}
QListWidget#Sidebar::item {{
    border-radius: 8px;
    padding: 11px 12px;
    margin: 2px 4px;
}}
QListWidget#Sidebar::item:selected {{
    background: {COLOR_PALETTE["primary_soft"]};
    color: {COLOR_PALETTE["primary"]};
    font-weight: 700;
}}
QScrollArea, QScrollArea > QWidget > QWidget {{
    background: transparent;
    border: none;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 12px;
    margin: 4px 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 12px;
    margin: 0 4px;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {COLOR_PALETTE["surface_alt_2"]};
    border-radius: 6px;
    min-height: 24px;
    min-width: 24px;
}}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background: {COLOR_PALETTE["primary_soft"]};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    background: none;
    border: none;
}}
QSplitter::handle {{
    background: transparent;
}}
QSplitter::handle:horizontal {{
    width: 10px;
}}
QSplitter::handle:vertical {{
    height: 10px;
}}
QSplitter::handle:hover {{
    background: {COLOR_PALETTE["primary_soft"]};
    border-radius: 5px;
}}
QProgressBar {{
    border: 1px solid {COLOR_PALETTE["border"]};
    border-radius: 6px;
    text-align: center;
    background: {COLOR_PALETTE["surface"]};
}}
QProgressBar::chunk {{
    background: {COLOR_PALETTE["success"]};
    border-radius: 5px;
}}
QToolBar {{
    spacing: 8px;
    background: {COLOR_PALETTE["surface"]};
    border-bottom: 1px solid {COLOR_PALETTE["border"]};
    padding: 6px 10px;
}}
QToolBar QToolButton {{
    background: {COLOR_PALETTE["surface_alt"]};
    color: {COLOR_PALETTE["text"]};
    border: 1px solid {COLOR_PALETTE["border"]};
    border-radius: 8px;
    padding: 8px 12px;
}}
QToolBar QToolButton:hover {{
    background: {COLOR_PALETTE["primary_soft"]};
}}
QStatusBar {{
    background: {COLOR_PALETTE["surface"]};
    border-top: 1px solid {COLOR_PALETTE["border"]};
}}
QStatusBar::item {{
    border: none;
}}
QLabel#PlotPlaceholder {{
    color: {COLOR_PALETTE["muted_text"]};
    font-size: 14px;
    padding: 20px;
}}
"""

NAVIGATION_PAGES = [
    "Home",
    "Theory",
    "Datasets",
    "Models",
    "Hyperparameters",
    "Training",
    "Visualizations",
    "Hyperparameter Lab",
    "Compare Runs",
    "Quiz / Practice",
    "About",
]
