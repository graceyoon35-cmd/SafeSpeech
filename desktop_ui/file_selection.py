"""File queue validation independent of Qt."""
from dataclasses import dataclass, replace
import os
import logging
from pathlib import Path
from error_handling import friendly_error

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SelectedFile:
    path: Path
    size_bytes: int
    status: str = "대기"
    error_message: str = ""

    @property
    def name(self):
        return self.path.name

    @property
    def extension(self):
        return self.path.suffix[1:].upper()

    @property
    def mb(self):
        return self.size_bytes / 1_000_000


class FileSelection:
    MAX_FILES = 10
    EXTENSIONS = frozenset((".wav", ".mp3", ".mp4"))

    def __init__(self):
        self._files = []

    @property
    def files(self):
        return tuple(self._files)

    def add(self, paths):
        """Add valid files in order; return (filename, reason) for rejections."""
        rejected = []
        keys = {os.path.normcase(str(item.path)) for item in self._files}
        for value in paths:
            try:
                path = Path(value)
                path = path.expanduser().resolve(strict=True)
                if not path.is_file():
                    raise OSError("Not a file")
                if path.suffix.lower() not in self.EXTENSIONS:
                    logger.warning("Unsupported file extension: %s", path)
                    rejected.append((path.name, "WAV, MP3, MP4 파일만 추가할 수 있습니다."))
                    continue
                key = os.path.normcase(str(path))
                if key in keys:
                    continue
                if len(self._files) >= self.MAX_FILES:
                    logger.warning("File limit exceeded: %s", path)
                    rejected.append((path.name, "최대 10개까지 추가할 수 있습니다."))
                    continue
                with path.open("rb") as stream:
                    stream.read(1)
                self._files.append(SelectedFile(path, path.stat().st_size))
                keys.add(key)
            except Exception as error:
                logger.exception("Could not add file: %r", value)
                rejected.append((str(value), friendly_error(error, "파일 경로가 올바르지 않거나 읽을 수 없습니다.")))
        return rejected

    def remove(self, index):
        del self._files[index]

    def set_status(self, index, status, error_message=""):
        if status not in ("대기", "처리 중", "완료", "실패"):
            raise ValueError(f"Unknown file status: {status}")
        self._files[index] = replace(self._files[index], status=status, error_message=error_message)
