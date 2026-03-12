"""Background worker for model training."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from ml_teaching_studio.core.trainers import TrainingRequest, train_and_evaluate


class TrainingWorker(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str, int)

    def __init__(self, request: TrainingRequest) -> None:
        super().__init__()
        self.request = request

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit("Preparing training pipeline...", 10)
            result = train_and_evaluate(self.request)
            self.progress.emit("Training complete.", 100)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))
