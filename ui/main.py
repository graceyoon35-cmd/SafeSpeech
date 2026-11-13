"""Voice Protect — run with .venv/Scripts/python.exe main.py."""
import sys
from pathlib import Path

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QStackedWidget

from ui.start_page import StartPage
from ui.main_page import MainPage
from error_handling import install_error_handler


class VoiceProtect(QStackedWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Protect")
        self.resize(1180, 820)
        self.setMinimumSize(800, 560)
        self.start_page = StartPage()
        self.main_page = MainPage()
        self.addWidget(self.start_page)
        self.addWidget(self.main_page)
        self.start_page.prepare_layout()
        self.main_page.prepare_layout()
        self.start_page.start_requested.connect(lambda: self.setCurrentWidget(self.main_page))


def create_app():
    install_error_handler()
    app = QApplication.instance() or QApplication(sys.argv)
    font_path = Path(__file__).parent / "assets/fonts/NanumSquare_acR.ttf"
    font_id = QFontDatabase.addApplicationFont(str(font_path))
    families = QFontDatabase.applicationFontFamilies(font_id)
    if not families:
        raise RuntimeError(f"Could not load application font: {font_path}")
    family = families[0]
    font = QFont(family)
    font.setWeight(QFont.Weight.Normal)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    font.setPixelSize(12)
    app.setFont(font)
    stylesheet = Path(__file__).with_name("styles.qss").read_text(encoding="utf-8")
    app.setStyleSheet(stylesheet.replace("__APP_FONT_FAMILY__", family))
    return app


if __name__ == "__main__":
    application = create_app()
    window = VoiceProtect()
    window.showMaximized()
    sys.exit(application.exec())
