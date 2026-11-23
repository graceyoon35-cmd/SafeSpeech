"""Run the media pipeline off the Qt event loop and dispatch signals on the UI thread."""
import logging
from PySide6.QtCore import QObject, QThread, Signal, Slot

logger = logging.getLogger(__name__)


class _Worker(QObject):
    started = Signal(int, int)
    succeeded = Signal(int, object)
    failed = Signal(int, str)
    progress = Signal(int)
    completed = Signal()

    def __init__(self, client, files):
        super().__init__()
        self.client, self.files = client, files

    @Slot()
    def run(self):
        try:
            for index, item in enumerate(self.files):
                self.started.emit(index, len(self.files))
                try:
                    def report(progress):
                        percent = int(100 * (index + progress.overall) / len(self.files))
                        self.progress.emit(percent)
                    result = self.client.protect(item, on_progress=report)
                except Exception as error:
                    logger.exception('SPEC failed at batch index %s', index)
                    self.failed.emit(index, str(error))
                else:
                    self.succeeded.emit(index, result)
                self.progress.emit(int(100*(index+1)/len(self.files)))
        finally:
            self.completed.emit()


class SpecProtectionProcessor(QObject):
    running_changed = Signal(bool)
    file_started = Signal(int, int)
    file_finished = Signal(int)
    file_failed = Signal(int, str)
    progress_changed = Signal(int)
    finished = Signal()

    def __init__(self, client, parent=None):
        super().__init__(parent)
        self.client = client
        self.running = False
        self.failures = {}
        self._results = []
        self._thread = None
        self._worker = None

    @property
    def results(self):
        return tuple(self._results)

    def start(self, files):
        if self.running or not files:
            return False
        files = tuple(files)
        if not files:
            return False
        self._results.clear()
        self.failures.clear()
        self.running = True
        self.running_changed.emit(True)
        self.progress_changed.emit(0)
        self._thread = QThread(self)
        self._worker = _Worker(self.client, files)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.started.connect(self._on_started)
        self._worker.succeeded.connect(self._on_success)
        self._worker.failed.connect(self._on_failure)
        self._worker.progress.connect(self._on_progress)
        self._worker.completed.connect(self._thread.quit)
        self._worker.completed.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._on_finished)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()
        return True

    @Slot(int, int)
    def _on_started(self, index, total):
        self.file_started.emit(index, total)

    @Slot(int, object)
    def _on_success(self, index, result):
        self._results.append(result)
        self.file_finished.emit(index)

    @Slot(int, str)
    def _on_failure(self, index, message):
        self.failures[index] = message
        self.file_failed.emit(index, message)

    @Slot(int)
    def _on_progress(self, value):
        self.progress_changed.emit(value)

    @Slot()
    def _on_finished(self):
        self._worker = None
        self._thread = None
        self.running = False
        self.running_changed.emit(False)
        self.finished.emit()
