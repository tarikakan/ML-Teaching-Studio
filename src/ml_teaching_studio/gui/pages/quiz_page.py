"""Quiz and practice page."""

from __future__ import annotations

import random

from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ml_teaching_studio.educational.quizzes import QUIZ_QUESTIONS
from ml_teaching_studio.gui.widgets.explanation_panel import ExplanationPanel


class QuizPage(QWidget):
    def __init__(self, studio, parent=None) -> None:
        super().__init__(parent)
        self.studio = studio
        self.current_question = None
        self.correct_count = 0
        self.total_answered = 0

        self.level_combo = QComboBox()
        self.level_combo.addItems(["All", "Beginner", "Intermediate", "Advanced", "Random"])
        self.question_label = QLabel()
        self.question_label.setWordWrap(True)
        self.button_group = QButtonGroup(self)
        self.option_buttons = [QRadioButton() for _ in range(4)]
        self.check_button = QPushButton("Check Answer")
        self.next_button = QPushButton("Next Question")
        self.score_label = QLabel("Score: 0/0")
        self.explanation_panel = ExplanationPanel("Answer Explanation")

        layout = QVBoxLayout(self)
        layout.addWidget(self.level_combo)
        layout.addWidget(self.question_label)
        for index, button in enumerate(self.option_buttons):
            self.button_group.addButton(button, index)
            layout.addWidget(button)
        layout.addWidget(self.check_button)
        layout.addWidget(self.next_button)
        layout.addWidget(self.score_label)
        layout.addWidget(self.explanation_panel, stretch=1)

        self.check_button.clicked.connect(self.check_answer)
        self.next_button.clicked.connect(self.load_question)
        self.level_combo.currentTextChanged.connect(self.load_question)
        self.load_question()

    def _filtered_questions(self):
        level = self.level_combo.currentText()
        if level == "All":
            return QUIZ_QUESTIONS
        if level == "Random":
            return QUIZ_QUESTIONS
        return [question for question in QUIZ_QUESTIONS if question.level == level]

    def load_question(self) -> None:
        questions = self._filtered_questions()
        if not questions:
            return
        self.current_question = random.choice(questions) if self.level_combo.currentText() == "Random" else questions[0]
        self.question_label.setText(self.current_question.prompt)
        for index, option_text in enumerate(self.current_question.options):
            self.option_buttons[index].setText(option_text)
            self.option_buttons[index].setChecked(False)
        self.explanation_panel.set_text("Select an answer, then check it to see the explanation.")

    def check_answer(self) -> None:
        if self.current_question is None:
            return
        selected_id = self.button_group.checkedId()
        if selected_id < 0:
            self.explanation_panel.set_text("Choose an option before checking your answer.")
            return
        self.total_answered += 1
        if selected_id == self.current_question.answer_index:
            self.correct_count += 1
            prefix = "<p><b>Correct.</b></p>"
        else:
            prefix = "<p><b>Not quite.</b></p>"
        self.score_label.setText(f"Score: {self.correct_count}/{self.total_answered}")
        self.explanation_panel.set_html(prefix + f"<p>{self.current_question.explanation}</p>")
