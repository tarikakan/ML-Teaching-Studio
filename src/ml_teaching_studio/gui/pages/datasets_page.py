"""Dataset loading and exploration page."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ml_teaching_studio.core.datasets import (
    available_builtin_datasets,
    dataset_preview,
    load_builtin_dataset,
    load_csv_dataset,
    summarize_dataset,
    target_overview,
)
from ml_teaching_studio.gui.dialogs.data_import_dialog import DataImportDialog
from ml_teaching_studio.gui.widgets.explanation_panel import ExplanationPanel
from ml_teaching_studio.gui.widgets.layout_utils import configure_splitter
from ml_teaching_studio.gui.widgets.metric_table import MetricTable
from ml_teaching_studio.gui.widgets.plot_canvas import PlotCanvas
from ml_teaching_studio.plotting import data_plots


class DatasetsPage(QWidget):
    def __init__(self, studio, parent=None) -> None:
        super().__init__(parent)
        self.studio = studio
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(available_builtin_datasets())
        self.load_button = QPushButton("Load Built-in Dataset")
        self.import_button = QPushButton("Import CSV")
        self.apply_button = QPushButton("Apply Target / Features")
        self.target_combo = QComboBox()
        self.task_combo = QComboBox()
        self.task_combo.addItems(["classification", "regression"])
        self.feature_list = QListWidget()
        self.feature_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.summary_browser = QTextBrowser()
        self.preview_table = MetricTable()
        self.plot_combo = QComboBox()
        self.plot_canvas = PlotCanvas()
        self.explanation_panel = ExplanationPanel("Dataset Interpretation")

        controls_box = QGroupBox("Dataset Controls")
        controls_layout = QFormLayout(controls_box)
        builtin_row = QHBoxLayout()
        builtin_row.addWidget(self.dataset_combo, stretch=1)
        builtin_row.addWidget(self.load_button)
        builtin_row.addWidget(self.import_button)
        controls_layout.addRow("Source", builtin_row)
        controls_layout.addRow("Target column", self.target_combo)
        controls_layout.addRow("Task type", self.task_combo)
        controls_layout.addRow("Feature selection", self.feature_list)
        controls_layout.addRow("", self.apply_button)

        summary_panel = QWidget()
        summary_panel.setObjectName("ContentPanel")
        summary_layout = QVBoxLayout(summary_panel)
        summary_layout.setContentsMargins(14, 14, 14, 14)
        summary_layout.setSpacing(10)
        summary_layout.addWidget(QLabel("Dataset summary"))
        summary_layout.addWidget(self.summary_browser, stretch=1)

        preview_panel = QWidget()
        preview_panel.setObjectName("MetricsSection")
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(14, 14, 14, 14)
        preview_layout.setSpacing(10)
        preview_layout.addWidget(QLabel("Preview table"))
        preview_layout.addWidget(self.preview_table, stretch=1)

        left_lower_splitter = QSplitter(Qt.Vertical)
        left_lower_splitter.addWidget(summary_panel)
        left_lower_splitter.addWidget(preview_panel)
        configure_splitter(left_lower_splitter, [260, 320], stretch_factors=[1, 1])

        left_panel = QWidget()
        left_panel.setObjectName("ControlPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(12)
        left_layout.addWidget(controls_box)
        left_layout.addWidget(left_lower_splitter, stretch=1)

        plot_panel = QWidget()
        plot_panel.setObjectName("PlotSection")
        right_layout = QVBoxLayout(plot_panel)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)
        right_layout.addWidget(QLabel("Dataset visual"))
        right_layout.addWidget(self.plot_combo)
        right_layout.addWidget(self.plot_canvas, stretch=1)

        right_panel = QSplitter(Qt.Vertical)
        right_panel.addWidget(plot_panel)
        right_panel.addWidget(self.explanation_panel)
        configure_splitter(right_panel, [620, 260], stretch_factors=[1, 0])

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        configure_splitter(splitter, [520, 920], stretch_factors=[0, 1])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self.load_button.clicked.connect(self._load_builtin)
        self.import_button.clicked.connect(self._import_csv)
        self.apply_button.clicked.connect(self._apply_configuration)
        self.target_combo.currentTextChanged.connect(self._target_changed)
        self.plot_combo.currentTextChanged.connect(self._render_plot)

        self._populate_from_dataset()

    def _populate_from_dataset(self) -> None:
        bundle = self.studio.current_dataset
        if bundle.name in available_builtin_datasets():
            self.dataset_combo.setCurrentText(bundle.name)
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItems([str(column) for column in bundle.dataframe.columns])
        self.target_combo.setCurrentText(self.studio.selected_target_column)
        self.target_combo.blockSignals(False)
        self.task_combo.setCurrentText(self.studio.selected_task_type)
        self._populate_feature_list()
        self.preview_table.set_dataframe(dataset_preview(bundle, rows=20))
        summary = summarize_dataset(bundle)
        target_info = target_overview(bundle)
        class_balance_html = ""
        if target_info.get("balance"):
            balance_items = "".join(
                f"<li>{label}: {count}</li>" for label, count in target_info["balance"].items()
            )
            class_balance_html = f"<p><b>Class balance</b></p><ul>{balance_items}</ul>"
        self.summary_browser.setHtml(
            f"""
            <h2>{bundle.name}</h2>
            <p>{bundle.description}</p>
            <p><b>Rows:</b> {summary.rows} | <b>Columns:</b> {summary.columns}</p>
            <p><b>Numeric features:</b> {len(summary.numeric_features)} | <b>Categorical features:</b> {len(summary.categorical_features)}</p>
            {class_balance_html}
            <p><b>Notes:</b> {'; '.join(bundle.notes) if bundle.notes else 'No extra notes.'}</p>
            """
        )
        self._populate_plot_choices()
        self._render_plot()

    def _populate_feature_list(self) -> None:
        bundle = self.studio.current_dataset
        self.feature_list.clear()
        selected = set(self.studio.selected_feature_columns)
        target_column = self.target_combo.currentText() or self.studio.selected_target_column
        for column in bundle.dataframe.columns:
            if column == target_column:
                continue
            item = QListWidgetItem(column)
            self.feature_list.addItem(item)
            item.setSelected(column in selected or not selected)

    def _populate_plot_choices(self) -> None:
        self.plot_combo.clear()
        options = ["Feature Histograms", "Correlation Heatmap", "Missing Values", "PCA Projection", "2D Preview"]
        if self.studio.current_dataset.task_type == "classification":
            options.insert(0, "Class Distribution")
        self.plot_combo.addItems(options)

    def _load_builtin(self) -> None:
        bundle = load_builtin_dataset(self.dataset_combo.currentText())
        self.studio.set_current_dataset(bundle)

    def _import_csv(self) -> None:
        dialog = DataImportDialog(self)
        if dialog.exec() != dialog.Accepted:
            return
        settings = dialog.selected_settings()
        bundle = load_csv_dataset(
            settings["path"],
            target_column=settings["target_column"],
            task_type=settings["task_type"],
        )
        self.studio.set_current_dataset(bundle)

    def _apply_configuration(self) -> None:
        features = [item.text() for item in self.feature_list.selectedItems()]
        self.studio.apply_dataset_configuration(
            target_column=self.target_combo.currentText(),
            feature_columns=features,
            task_type=self.task_combo.currentText(),
        )
        self.explanation_panel.set_html(
            "<p>The active target, feature selection, and task type were updated. The Training and Hyperparameter Lab pages now use this configuration.</p>"
        )

    def _target_changed(self) -> None:
        self._populate_feature_list()

    def _render_plot(self) -> None:
        bundle = self.studio.current_dataset
        plot_name = self.plot_combo.currentText()
        if plot_name == "Class Distribution":
            figure = data_plots.plot_class_distribution(bundle)
            explanation = "<p>This chart shows whether the classes are balanced. Strong imbalance can make accuracy misleading.</p>"
        elif plot_name == "Feature Histograms":
            figure = data_plots.plot_feature_histograms(bundle)
            explanation = "<p>Feature distributions help you notice skew, outliers, and scale differences before choosing a model.</p>"
        elif plot_name == "Correlation Heatmap":
            figure = data_plots.plot_correlation_heatmap(bundle)
            explanation = "<p>Correlation heatmaps show linear relationships between numeric features. Very strong correlations can affect linear models.</p>"
        elif plot_name == "Missing Values":
            figure = data_plots.plot_missing_values(bundle)
            explanation = "<p>This view shows whether imputation is likely to matter before training.</p>"
        elif plot_name == "PCA Projection":
            figure = data_plots.plot_pca_projection(bundle)
            explanation = "<p>PCA gives a compressed 2D view of major variation. Separation in this view often hints at learnability, but it is only an approximation.</p>"
        else:
            figure = data_plots.plot_scatter_preview(bundle)
            explanation = "<p>A 2D scatter preview is especially useful on synthetic datasets, where the geometry of the problem is easier to interpret visually.</p>"
        self.plot_canvas.set_figure(figure)
        self.explanation_panel.set_html(explanation)

    def on_dataset_changed(self) -> None:
        self._populate_from_dataset()
