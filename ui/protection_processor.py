"""Nonblocking dummy backend.

An AI worker can later expose this same start/signals interface. Run expensive
inference in a worker thread/process, not in the GUI timer callback.
"""
from dataclasses import dataclass
from pathlib import Path
import logging
from error_handling import friendly_error

logger = logging.getLogger(__name__)

from PySide6.QtCore import QObject, Signal, QTimer, QElapsedTimer


@dataclass(frozen=True)
class ProtectionResult:
    original_path: Path
    elapsed_seconds: float
    confidence: int = 92  # Fixed test value, not an AI confidence estimate.
    output_path: Path | None = None  # Set to the AI/backend output when available.

    @property
    def protected_name(self):
        return f"{self.original_path.stem}_protected{self.original_path.suffix}"


class DummyProtectionProcessor(QObject):
    running_changed = Signal(bool)
    file_started = Signal(int, int)  # zero-based index, batch size
    file_finished = Signal(int)
    file_failed = Signal(int, str)
    progress_changed = Signal(int)  # overall batch percentage
    finished = Signal()

    def __init__(self, parent=None, file_duration_ms=1500):
        super().__init__(parent)
        if file_duration_ms <= 0:
            raise ValueError("File duration must be positive")
        self.file_duration_ms = file_duration_ms
        self.running = False
        self._files = ()
        self._index = 0
        self._elapsed = QElapsedTimer()
        self._results = []
        self.failures = {}
        self._checked = False
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._tick)

    @property
    def results(self):
        return tuple(self._results)

    def start(self, files):
        if self.running or not files:
            return False
        self._files = tuple(files)
        self._index = 0
        self._results.clear()
        self.failures.clear()
        self.running = True
        self.running_changed.emit(True)
        self.progress_changed.emit(0)
        self._start_current_file()
        self._timer.start()
        return True

    def _start_current_file(self):
        self._checked = False
        self._elapsed.start()
        self.file_started.emit(self._index, len(self._files))

    def _tick(self):
        if not self.running:
            return
        try:
            path = Path(getattr(self._files[self._index], "path", self._files[self._index]))
            if not self._checked:
                self._validate_file(path)
                self._checked = True
            fraction = min(1.0, self._elapsed.elapsed() / self.file_duration_ms)
            self.progress_changed.emit(int(100 * (self._index + fraction) / len(self._files)))
            if fraction < 1:
                return
            self._validate_file(path)
            self._results.append(self._build_result(path))
        except Exception as error:
            logger.exception("Processing failed at index %s: %r", self._index, self._files[self._index])
            message = friendly_error(error, "이 파일을 처리하지 못했습니다. 파일을 확인한 뒤 다시 시도해주세요.")
            self.failures[self._index] = message
            self.file_failed.emit(self._index, message)
        else:
            self.file_finished.emit(self._index)
        self.progress_changed.emit(int(100 * (self._index + 1) / len(self._files)))
        self._advance()

    @staticmethod
    def _validate_file(path):
        if path.suffix.lower() not in (".wav", ".mp3", ".mp4"):
            raise ValueError("Unsupported media extension")
        with path.open("rb") as stream:
            stream.read(1)

    def _build_result(self, path):
        """Replace this dummy result factory when connecting the AI backend."""
        return ProtectionResult(path, self._elapsed.elapsed() / 1000)

    def _advance(self):
        self._index += 1
        if self._index < len(self._files):
            self._start_current_file()
        else:
            self._timer.stop()
            self.running = False
            self.running_changed.emit(False)
            self.finished.emit()
