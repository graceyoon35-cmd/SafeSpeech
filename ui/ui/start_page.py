"""Figma 29:2. Shared drawing helpers retain Figma logical pixel geometry."""
from pathlib import Path
import re

from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPainterPath
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtSvgWidgets import QSvgWidget
from PySide6.QtWidgets import QWidget, QFrame, QLabel, QPushButton, QGraphicsDropShadowEffect

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def label(parent, text, rect, size=12, color="#344054", weight=400, align=Qt.AlignLeft):
    widget = QLabel(text, parent)
    widget.setGeometry(*rect)
    widget.setAlignment(align | Qt.AlignVCenter)
    widget.setStyleSheet(f"font-size: {size}px; font-weight: {weight}; color: {color}; background: transparent; border: none;")
    return widget


def button(parent, text, rect, role="secondary"):
    widget = QPushButton(text, parent)
    widget.setProperty("role", role)
    widget.setGeometry(*rect)
    widget.setCursor(Qt.PointingHandCursor)
    return widget


class SurfaceFrame(QFrame):
    """Keep shadow rasterization on a sibling containing only the background."""
    def __init__(self, parent, role):
        self.backdrop = QFrame(parent)
        self.backdrop.setProperty("role", role)
        self.backdrop.setAttribute(Qt.WA_TransparentForMouseEvents)
        super().__init__(parent)
        self.setProperty("role", "surfaceContent")
        self.destroyed.connect(self.backdrop.deleteLater)
        effect = QGraphicsDropShadowEffect(self.backdrop)
        effect.setBlurRadius(48 if role == "glass" else 24)
        effect.setOffset(0, 18 if role == "glass" else 8)
        effect.setColor(QColor(41, 61, 102, 31) if role == "glass" else QColor(46, 71, 117, 26))
        self.backdrop.setGraphicsEffect(effect)

    def moveEvent(self, event):
        self.backdrop.setGeometry(self.geometry())
        super().moveEvent(event)

    def resizeEvent(self, event):
        self.backdrop.setGeometry(self.geometry())
        super().resizeEvent(event)

    def showEvent(self, event):
        self.backdrop.setGeometry(self.geometry())
        self.backdrop.show()
        self.backdrop.stackUnder(self)
        super().showEvent(event)

    def hideEvent(self, event):
        self.backdrop.hide()
        super().hideEvent(event)


def panel(parent, rect, role="card"):
    if role in ("card", "glass"):
        widget = SurfaceFrame(parent, role)
        widget.setGeometry(*rect)
        widget.backdrop.setGeometry(*rect)
        return widget
    widget = QFrame(parent)
    widget.setProperty("role", role)
    widget.setGeometry(*rect)
    return widget


class DesignPage(QWidget):
    def __init__(self, main=False):
        super().__init__()
        self._layout_ready = False
        self.resize(1180, 820)
        self.main = main
        prefix = "main" if main else "start"
        self.glows = [QSvgRenderer(str(ASSETS / f"{prefix}_{name}.svg")) for name in ("blue", "mint")]
        self.glass = panel(self, (70, 60, 1040, 700), "glass")
        self.glass.setProperty("main", main)
        self.glass.backdrop.setProperty("main", main)

    def prepare_layout(self):
        self._layout_ready = True
        self.adapt_layout()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._layout_ready:
            self.adapt_layout()

    def adapt_layout(self):
        if not self._layout_ready:
            return
        sx, sy = self.width() / 1180, self.height() / 820
        scale = min(sx, sy)
        widgets = self.findChildren(QWidget)
        # Snapshot all original rectangles before parent resize events run.
        for widget in widgets:
            if not hasattr(widget, "_design_rect"):
                widget._design_rect = widget.geometry()
                widget._design_style = widget.styleSheet()
                widget.ensurePolished()
                widget._design_font_size = widget.font().pixelSize()
        for widget in widgets:
            rect = widget._design_rect
            x, y = round(rect.x() * sx), round(rect.y() * sy)
            width, height = round(rect.width() * sx), round(rect.height() * sy)
            if isinstance(widget, QSvgWidget):
                # Keep the logo's original proportions on wide monitors.
                logo_width, logo_height = round(rect.width() * scale), round(rect.height() * scale)
                x += (width - logo_width) // 2
                y += (height - logo_height) // 2
                width, height = logo_width, logo_height
            widget.setGeometry(x, y, width, height)
            style = re.sub(r"(\d+(?:\.\d+)?)px", lambda match: f"{float(match[1]) * scale:.2f}px", widget._design_style)
            if isinstance(widget, (QLabel, QPushButton)) and "font-size:" not in style:
                style += f"; font-size: {max(1, widget._design_font_size * scale):.2f}px;"
            if widget.styleSheet() != style:
                widget.setStyleSheet(style)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        clip = QPainterPath()
        clip.addRoundedRect(QRectF(self.rect()), 28, 28)
        painter.setClipPath(clip)
        painter.scale(self.width() / 1180, self.height() / 820)
        gradient = QLinearGradient(0, 0, 1180, 390 if self.main else 382)
        colors = ("#dde7ff", "#eaf7f4", "#d9f4e8") if self.main else ("#e4ebff", "#eef7f7", "#ddf5e9")
        for stop, color in zip((0, .48, 1), colors):
            gradient.setColorAt(stop, QColor(color))
        painter.fillRect(QRectF(0, 0, 1180, 820), gradient)
        # SVG bounds include the original Figma blur/filter overflow.
        bounds = [(-190, -210, 560, 560), (790, 450, 520, 520)] if self.main else [(-210, -200, 600, 600), (765, 435, 590, 590)]
        for renderer, rect in zip(self.glows, bounds):
            renderer.render(painter, QRectF(*rect))


class StartPage(DesignPage):
    start_requested = Signal()

    def __init__(self):
        super().__init__()
        # Figma content origin is (415, 139) inside the 1040 × 700 glass.
        self.logo = QSvgWidget(str(ASSETS / "logo.svg"), self.glass)
        self.logo.setGeometry(413, 131, 214, 218)
        label(self.glass, "Voice Protect", (390, 361, 260, 39), 32, "#1d2939", 600, Qt.AlignHCenter)
        label(self.glass, "음성·영상 딥보이스 보호", (370, 412, 300, 17), 14, "#667085", 500, Qt.AlignHCenter)
        label(self.glass, "내 음성과 영상을 안전하게 보호하세요.", (350, 441, 340, 18), 13, "#7c8798", 400, Qt.AlignHCenter)
        self.start_button = button(self.glass, "시작하기", (430, 491, 180, 46), "start")
        self.start_button.clicked.connect(self.start_requested)
        label(self.glass, "WAV · MP3 · MP4 지원", (400, 549, 240, 14), 10, "#98a2b3", 400, Qt.AlignHCenter)
