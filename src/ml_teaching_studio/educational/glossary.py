"""Glossary definitions."""

from __future__ import annotations

GLOSSARY = {
    "accuracy": "Fraction of predictions that are correct.",
    "bias": "Error from overly simple assumptions that miss important patterns.",
    "calibration": "How well predicted probabilities match observed frequencies.",
    "classification": "Predicting a discrete class label.",
    "cross-validation": "Repeated train/validation splits used to estimate performance more reliably.",
    "decision boundary": "Surface separating classes in feature space.",
    "feature": "An input variable used for prediction.",
    "feature importance": "Estimate of how much a feature influences a model's output.",
    "generalization": "How well a model performs on unseen data.",
    "hyperparameter": "A model setting chosen before training, such as max_depth or C.",
    "inference": "Using a trained model to make predictions.",
    "learning curve": "Plot of performance as training size changes.",
    "normalization": "Rescaling each sample, often to unit length.",
    "overfitting": "Model learns noise or split-specific quirks instead of stable signal.",
    "pca": "Principal Component Analysis, a projection to fewer dimensions capturing major variation.",
    "precision": "Among predicted positives, the fraction that are truly positive.",
    "preprocessing": "Data preparation steps applied before modeling.",
    "recall": "Among true positives, the fraction that the model successfully finds.",
    "regression": "Predicting a continuous numeric target.",
    "regularization": "Techniques that penalize model complexity to improve generalization.",
    "residual": "Prediction error for one regression example.",
    "roc auc": "Probability-based metric measuring class ranking quality across thresholds.",
    "scaling": "Adjusting features to comparable numeric ranges.",
    "target": "The variable the model tries to predict.",
    "underfitting": "Model is too simple to capture the main pattern in the data.",
    "variance": "Sensitivity of model behavior to the specific training sample.",
}
