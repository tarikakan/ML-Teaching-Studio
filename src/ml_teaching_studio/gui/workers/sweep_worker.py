"""Background worker for hyperparameter sweeps."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from ml_teaching_studio.core.hyperparameter_sweeps import SweepRequest, run_hyperparameter_sweep


class SweepWorker(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str, int)

    def __init__(self, request: SweepRequest) -> None:
        super().__init__()
        self.request = request

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit("Starting hyperparameter sweep...", 5)
            result = run_hyperparameter_sweep(
                self.request,
                progress_callback=lambda message, value: self.progress.emit(message, value),
            )
            self.progress.emit("Sweep complete.", 100)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))
