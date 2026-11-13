"""Validate save destinations and export real outputs, without any UI code."""
from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import tempfile

from protection_processor import ProtectionResult


@dataclass(frozen=True)
class SaveOutcome:
    destination: Path
    written: bool


def prepare_destination(result: ProtectionResult, destination) -> Path:
    path = Path(destination).expanduser()
    if not path.suffix:
        path = path.with_suffix(result.original_path.suffix)
    if path.suffix.lower() != result.original_path.suffix.lower():
        raise ValueError("원본과 동일한 확장자로 저장해주세요.")
    path = path.resolve()
    if path == result.original_path.resolve():
        raise ValueError("원본 파일을 덮어쓸 수 없습니다. 다른 파일명을 선택해주세요.")
    if not path.parent.is_dir():
        raise FileNotFoundError("저장할 폴더가 존재하지 않습니다.")
    if path.is_dir():
        raise IsADirectoryError("파일명 대신 폴더가 선택되었습니다.")
    return path


def save_result(result: ProtectionResult, destination) -> SaveOutcome:
    destination = prepare_destination(result, destination)
    if result.output_path is None:
        # Explicit dry run: never copy the unprotected original as a result.
        return SaveOutcome(destination, written=False)
    source = Path(result.output_path).resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError("보호 처리 결과 파일을 찾을 수 없습니다.")
    if source == destination:
        return SaveOutcome(destination, written=True)
    # Copy to a sibling temporary file first so copy errors do not truncate an
    # existing destination. Replace only after a complete successful copy.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".voice-protect-", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return SaveOutcome(destination, written=True)
