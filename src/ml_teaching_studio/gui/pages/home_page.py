"""Home page."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QPushButton, QTextBrowser, QVBoxLayout, QWidget

from ml_teaching_studio.utils.config import APP_NAME


class HomePage(QWidget):
    def __init__(self, studio, parent=None) -> None:
        super().__init__(parent)
        self.studio = studio

        self.hero = QTextBrowser()
        self.hero.setHtml(
            f"""
            <h1>{APP_NAME}</h1>
            <p>A desktop learning lab for understanding machine learning models, preprocessing choices, and hyperparameter effects.</p>
            <p>Use the navigation on the left to move from theory into datasets, training, hyperparameter sweeps, and run comparisons.</p>
            """
        )
        self.dataset_card = self._build_card("Active Dataset")
        self.model_card = self._build_card("Preferred Model")
        self.run_card = self._build_card("Saved Runs")

        quick_training = QPushButton("Go to Training")
        quick_theory = QPushButton("Open Theory")
        quick_lab = QPushButton("Open Hyperparameter Lab")
        quick_training.clicked.connect(lambda: self.studio.navigate("Training"))
        quick_theory.clicked.connect(lambda: self.studio.navigate("Theory"))
        quick_lab.clicked.connect(lambda: self.studio.navigate("Hyperparameter Lab"))

        cards_layout = QGridLayout()
        cards_layout.addWidget(self.dataset_card, 0, 0)
        cards_layout.addWidget(self.model_card, 0, 1)
        cards_layout.addWidget(self.run_card, 0, 2)

        button_layout = QGridLayout()
        button_layout.addWidget(quick_training, 0, 0)
        button_layout.addWidget(quick_theory, 0, 1)
        button_layout.addWidget(quick_lab, 0, 2)

        layout = QVBoxLayout(self)
        layout.addWidget(self.hero)
        layout.addLayout(cards_layout)
        layout.addLayout(button_layout)
        layout.addStretch(1)

        self.refresh()

    def _build_card(self, title: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName("Card")
        layout = QVBoxLayout(frame)
        heading = QLabel(title)
        heading.setStyleSheet("font-size: 15px; font-weight: 700;")
        body = QLabel("")
        body.setWordWrap(True)
        layout.addWidget(heading)
        layout.addWidget(body)
        frame.body_label = body  # type: ignore[attr-defined]
        return frame

    def refresh(self) -> None:
        dataset_text = (
            f"{self.studio.current_dataset.name}\n"
            f"Target: {self.studio.selected_target_column}\n"
            f"Features: {len(self.studio.selected_feature_columns)} selected"
        )
        self.dataset_card.body_label.setText(dataset_text)  # type: ignore[attr-defined]
        self.model_card.body_label.setText(self.studio.preferred_model_name)  # type: ignore[attr-defined]
        self.run_card.body_label.setText(
            f"{len(self.studio.run_store.list_runs())} saved experiment runs"
        )  # type: ignore[attr-defined]

    def on_dataset_changed(self) -> None:
        self.refresh()

    def on_run_store_changed(self) -> None:
        self.refresh()
