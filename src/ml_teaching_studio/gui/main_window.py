"""Main window for ML-Teaching Studio."""

from __future__ import annotations

from typing import Any

from PySide6.QtGui import QAction
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QStatusBar,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ml_teaching_studio.core.datasets import available_builtin_datasets, load_builtin_dataset
from ml_teaching_studio.core.run_store import RunStore
from ml_teaching_studio.models.model_registry import default_model_for_task, get_model_spec
from ml_teaching_studio.utils.config import APP_NAME, DEFAULT_WINDOW_HEIGHT, DEFAULT_WINDOW_WIDTH, NAVIGATION_PAGES

from .widgets.layout_utils import configure_splitter
from .pages.about_page import AboutPage
from .pages.compare_runs_page import CompareRunsPage
from .pages.datasets_page import DatasetsPage
from .pages.home_page import HomePage
from .pages.hyperparameter_lab_page import HyperparameterLabPage
from .pages.hyperparameters_page import HyperparametersPage
from .pages.models_page import ModelsPage
from .pages.quiz_page import QuizPage
from .pages.theory_page import TheoryPage
from .pages.training_page import TrainingPage
from .pages.visualizations_page import VisualizationsPage


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)

        self.run_store = RunStore()
        self.current_dataset = load_builtin_dataset("Iris")
        self.selected_target_column = self.current_dataset.target_column
        self.selected_feature_columns = self.current_dataset.feature_columns.copy()
        self.selected_task_type = self.current_dataset.task_type
        self.preferred_model_name = default_model_for_task(self.selected_task_type)
        self.current_training_result = None
        self.current_sweep_result = None

        self.sidebar = QListWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setMinimumWidth(220)
        self.sidebar.addItems(NAVIGATION_PAGES)
        self.stack = QStackedWidget()
        self.page_map: dict[str, QWidget] = {}

        sidebar_panel = QWidget()
        sidebar_panel.setObjectName("SidebarPanel")
        sidebar_layout = QVBoxLayout(sidebar_panel)
        sidebar_layout.setContentsMargins(14, 14, 14, 14)
        sidebar_layout.setSpacing(10)
        sidebar_title = QLabel(APP_NAME)
        sidebar_title.setProperty("role", "sidebarTitle")
        sidebar_layout.addWidget(sidebar_title)
        sidebar_layout.addWidget(self.sidebar, stretch=1)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.addWidget(sidebar_panel)
        self.main_splitter.addWidget(self.stack)
        configure_splitter(self.main_splitter, [270, 1300], stretch_factors=[0, 1])

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.addWidget(self.main_splitter, stretch=1)
        self.setCentralWidget(central)
        self.setStatusBar(QStatusBar())

        self._build_toolbar()
        self._build_pages()

        self.sidebar.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.sidebar.setCurrentRow(0)
        self.statusBar().showMessage("Loaded default dataset: Iris", 4000)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Main")
        self.addToolBar(toolbar)

        self.toolbar_dataset_combo = QComboBox()
        self.toolbar_dataset_combo.addItems(available_builtin_datasets())
        self.toolbar_dataset_combo.setCurrentText(self.current_dataset.name)
        load_dataset_action = QAction("Load Dataset", self)
        open_datasets_action = QAction("Datasets", self)
        train_action = QAction("Training", self)
        save_run_action = QAction("Save Run", self)
        compare_action = QAction("Compare Runs", self)

        load_dataset_action.triggered.connect(
            lambda: self.set_current_dataset(load_builtin_dataset(self.toolbar_dataset_combo.currentText()))
        )
        open_datasets_action.triggered.connect(lambda: self.navigate("Datasets"))
        train_action.triggered.connect(lambda: self.navigate("Training"))
        save_run_action.triggered.connect(lambda: self.save_training_result())
        compare_action.triggered.connect(lambda: self.navigate("Compare Runs"))

        toolbar.addWidget(QLabel("Built-in dataset"))
        toolbar.addWidget(self.toolbar_dataset_combo)
        toolbar.addAction(load_dataset_action)
        toolbar.addAction(open_datasets_action)
        toolbar.addAction(train_action)
        toolbar.addAction(save_run_action)
        toolbar.addAction(compare_action)

    def _build_pages(self) -> None:
        self.page_map = {
            "Home": HomePage(self),
            "Theory": TheoryPage(self),
            "Datasets": DatasetsPage(self),
            "Models": ModelsPage(self),
            "Hyperparameters": HyperparametersPage(self),
            "Training": TrainingPage(self),
            "Visualizations": VisualizationsPage(self),
            "Hyperparameter Lab": HyperparameterLabPage(self),
            "Compare Runs": CompareRunsPage(self),
            "Quiz / Practice": QuizPage(self),
            "About": AboutPage(self),
        }
        for name in NAVIGATION_PAGES:
            self.stack.addWidget(self.page_map[name])

    def navigate(self, page_name: str) -> None:
        if page_name not in self.page_map:
            return
        index = NAVIGATION_PAGES.index(page_name)
        self.sidebar.setCurrentRow(index)

    def refresh_pages(self) -> None:
        for page in self.page_map.values():
            if hasattr(page, "on_dataset_changed"):
                page.on_dataset_changed()
            if hasattr(page, "on_training_result_changed"):
                page.on_training_result_changed()
            if hasattr(page, "on_sweep_result_changed"):
                page.on_sweep_result_changed()
            if hasattr(page, "on_run_store_changed"):
                page.on_run_store_changed()
            if hasattr(page, "on_model_changed"):
                page.on_model_changed()

    def set_current_dataset(self, bundle: Any) -> None:
        self.current_dataset = bundle
        if hasattr(self, "toolbar_dataset_combo") and bundle.name in available_builtin_datasets():
            self.toolbar_dataset_combo.setCurrentText(bundle.name)
        self.selected_target_column = bundle.target_column
        self.selected_feature_columns = bundle.feature_columns.copy()
        self.selected_task_type = bundle.task_type
        self.preferred_model_name = default_model_for_task(bundle.task_type)
        self.current_training_result = None
        self.current_sweep_result = None
        self.refresh_pages()
        self.statusBar().showMessage(f"Active dataset set to {bundle.name}", 4000)

    def apply_dataset_configuration(
        self,
        *,
        target_column: str,
        feature_columns: list[str],
        task_type: str,
        refresh: bool = True,
    ) -> None:
        self.selected_target_column = target_column
        self.selected_feature_columns = feature_columns
        self.selected_task_type = task_type
        try:
            if get_model_spec(self.preferred_model_name).task_type != task_type:
                self.preferred_model_name = default_model_for_task(task_type)
        except Exception:
            self.preferred_model_name = default_model_for_task(task_type)
        if refresh:
            self.refresh_pages()

    def set_preferred_model(self, model_name: str, *, refresh: bool = True) -> None:
        self.preferred_model_name = model_name
        if refresh:
            for page in self.page_map.values():
                if hasattr(page, "on_model_changed"):
                    page.on_model_changed()

    def set_training_result(self, result: Any) -> None:
        self.current_training_result = result
        for page in self.page_map.values():
            if hasattr(page, "on_training_result_changed"):
                page.on_training_result_changed()
        self.statusBar().showMessage(f"Finished training {result.model_name}", 4000)

    def set_sweep_result(self, result: Any) -> None:
        self.current_sweep_result = result
        for page in self.page_map.values():
            if hasattr(page, "on_sweep_result_changed"):
                page.on_sweep_result_changed()
        self.statusBar().showMessage(f"Finished hyperparameter sweep for {result.model_name}", 4000)

    def save_training_result(self, notes: str = "") -> None:
        if self.current_training_result is None:
            self.show_error("Save Run", "No completed training run is available to save.")
            return
        record = self.current_training_result.to_record()
        if notes:
            record["notes"] = notes
        self.run_store.save_run(record)
        self.statusBar().showMessage("Saved current training run.", 4000)
        for page in self.page_map.values():
            if hasattr(page, "on_run_store_changed"):
                page.on_run_store_changed()

    def save_sweep_result(self, notes: str = "") -> None:
        if self.current_sweep_result is None:
            self.show_error("Save Sweep", "No hyperparameter sweep is available to save.")
            return
        record = self.current_sweep_result.to_record()
        if notes:
            record["notes"] = notes
        self.run_store.save_sweep(record)
        self.statusBar().showMessage("Saved current hyperparameter sweep.", 4000)

    def show_error(self, title: str, message: str) -> None:
        QMessageBox.critical(self, title, message)
