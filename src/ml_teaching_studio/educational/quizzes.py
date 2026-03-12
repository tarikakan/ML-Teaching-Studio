"""Quiz content for the practice module."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuizQuestion:
    question_id: str
    level: str
    topic: str
    prompt: str
    options: tuple[str, ...]
    answer_index: int
    explanation: str


QUIZ_QUESTIONS = [
    QuizQuestion(
        question_id="q1",
        level="Beginner",
        topic="Overfitting",
        prompt="A decision tree has very high training accuracy and much lower test accuracy. What is the most likely diagnosis?",
        options=("Overfitting", "Underfitting", "Perfect generalization", "Data leakage is impossible"),
        answer_index=0,
        explanation="Large train-test gaps usually indicate overfitting. The tree likely became too flexible.",
    ),
    QuizQuestion(
        question_id="q2",
        level="Beginner",
        topic="KNN",
        prompt="What usually happens when K in KNN is set to 1 on a noisy dataset?",
        options=("Boundary becomes smoother", "Model becomes more flexible and noise-sensitive", "Distances stop mattering", "Scaling becomes irrelevant"),
        answer_index=1,
        explanation="Very small K makes KNN highly flexible, so it can fit local noise and overfit.",
    ),
    QuizQuestion(
        question_id="q3",
        level="Beginner",
        topic="Metrics",
        prompt="Which metric is generally more informative than accuracy on a highly imbalanced binary classification problem?",
        options=("Precision and recall", "R²", "Mean absolute error", "Only training accuracy"),
        answer_index=0,
        explanation="Precision and recall focus on the positive class and expose failure modes hidden by accuracy.",
    ),
    QuizQuestion(
        question_id="q4",
        level="Intermediate",
        topic="SVM",
        prompt="Increasing C in an SVM usually means:",
        options=("Stronger regularization", "Weaker regularization and a tighter fit to training data", "Fewer support vectors by definition", "Kernel changes automatically"),
        answer_index=1,
        explanation="C is the penalty for training errors. Larger C relaxes regularization and allows a more complex separator.",
    ),
    QuizQuestion(
        question_id="q5",
        level="Intermediate",
        topic="Preprocessing",
        prompt="Why is scaling especially important for KNN and SVM?",
        options=("They only accept integer features", "They depend on distances or margins that are distorted by unequal feature scales", "Scaling increases dataset size", "Tree models require it"),
        answer_index=1,
        explanation="Distance-based and margin-based models are sensitive to the relative size of features.",
    ),
    QuizQuestion(
        question_id="q6",
        level="Intermediate",
        topic="Model Selection",
        prompt="Which comparison is fairest when evaluating two models?",
        options=("Different test sets and different metrics", "Same train/test split, same preprocessing, same metric", "One model tuned, the other left at defaults", "Comparing only training scores"),
        answer_index=1,
        explanation="Fair comparison controls the evaluation setup so differences mostly reflect the models themselves.",
    ),
    QuizQuestion(
        question_id="q7",
        level="Intermediate",
        topic="Regularization",
        prompt="What is a common effect of increasing alpha in Ridge or Lasso regression?",
        options=("The model becomes less regularized", "The model ignores scaling", "Coefficients are shrunk more strongly", "The target becomes categorical"),
        answer_index=2,
        explanation="Alpha controls regularization strength. Larger values shrink coefficients and can reduce variance.",
    ),
    QuizQuestion(
        question_id="q8",
        level="Intermediate",
        topic="Boosting",
        prompt="In gradient boosting, lowering the learning rate while increasing the number of estimators often:",
        options=("Makes the model learn more gradually", "Forces linear decision boundaries", "Removes the need for validation", "Eliminates overfitting risk"),
        answer_index=0,
        explanation="Lower learning rates make each stage smaller, so more stages are needed to build the final model.",
    ),
    QuizQuestion(
        question_id="q9",
        level="Advanced",
        topic="Validation Curves",
        prompt="A validation curve peaks in the middle of the hyperparameter range and drops at both extremes. What does this suggest?",
        options=("There is a useful bias-variance balance in the middle", "The metric is broken", "Data leakage occurred automatically", "Preprocessing does not matter"),
        answer_index=0,
        explanation="Very small or very large values can respectively underfit or overfit, so moderate values may generalize best.",
    ),
    QuizQuestion(
        question_id="q10",
        level="Advanced",
        topic="Interpretation",
        prompt="A random forest shows strong feature importance for one variable. What should a learner conclude first?",
        options=("That feature definitely causes the outcome", "The model relied heavily on that feature for prediction, but causality is not established", "The feature must be scaled", "The variable is irrelevant"),
        answer_index=1,
        explanation="Feature importance indicates predictive dependence, not causal proof.",
    ),
]
