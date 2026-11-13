"""Friendly user messages and full console diagnostics."""
import logging
import sys

def friendly_error(error, fallback="작업을 완료하지 못했습니다. 잠시 후 다시 시도해주세요."):
    if isinstance(error, FileNotFoundError):
        return "파일 또는 폴더를 찾을 수 없습니다. 삭제되거나 이동했는지 확인해주세요."
    if isinstance(error, PermissionError):
        return "파일이나 폴더에 접근할 수 없습니다. 접근 권한과 다른 프로그램의 사용 여부를 확인해주세요."
    if isinstance(error, IsADirectoryError):
        return "폴더 대신 파일을 선택해주세요."
    return fallback


def install_error_handler():
    from PySide6.QtWidgets import QApplication, QMessageBox
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    def handle_exception(error_type, error, traceback):
        if issubclass(error_type, (KeyboardInterrupt, SystemExit)):
            sys.__excepthook__(error_type, error, traceback)
            return
        logging.getLogger("voice_protect").error("Unhandled application error", exc_info=(error_type, error, traceback))
        if QApplication.instance() is not None:
            QMessageBox.warning(None, "작업 오류", friendly_error(error))

    sys.excepthook = handle_exception
