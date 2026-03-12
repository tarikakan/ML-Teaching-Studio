"""Neural-network model helpers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class NeuralDemo:
    name: str
    backend: str
    description: str


def torch_available() -> bool:
    return find_spec("torch") is not None


def available_neural_demos() -> list[NeuralDemo]:
    demos = [
        NeuralDemo(
            name="MLP Classifier",
            backend="scikit-learn",
            description="Feed-forward neural network for classification with editable hidden layers.",
        ),
        NeuralDemo(
            name="MLP Regressor",
            backend="scikit-learn",
            description="Feed-forward neural network for regression with editable hidden layers.",
        ),
    ]
    if torch_available():
        demos.append(
            NeuralDemo(
                name="PyTorch Demo Available",
                backend="PyTorch",
                description="PyTorch is installed, so custom neural-network demos can be added later.",
            )
        )
    return demos
