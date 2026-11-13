"""Exclusive original/protected audio playback using QtMultimedia.

A single player guarantees the two tracks cannot play simultaneously.
MP4 uses its audio track; no video widget is attached.
"""
from pathlib import Path
import logging
from error_handling import friendly_error

logger = logging.getLogger(__name__)

from PySide6.QtCore import QObject, Signal, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer


class MediaPlayback(QObject):
    state_changed = Signal(str, str)  # original/protected, playing/paused/stopped
    error_occurred = Signal(str)
    ROLES = ("original", "protected")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.8)
        self.player.setAudioOutput(self.audio_output)
        self.sources = dict.fromkeys(self.ROLES)
        self.active_role = None
        self.player.playbackStateChanged.connect(self._on_state_changed)
        self.player.errorOccurred.connect(self._on_error)

    def set_sources(self, original=None, protected=None):
        self.stop()
        self.player.setSource(QUrl())
        self.sources = {"original": Path(original) if original is not None else None,
                        "protected": Path(protected) if protected is not None else None}

    def available(self, role):
        path = self.sources.get(role)
        try:
            return path is not None and path.is_file()
        except OSError:
            return False

    def toggle(self, role):
        try:
            self._toggle(role)
        except Exception as error:
            logger.exception("Playback failed for %s: %r", role, self.sources.get(role))
            self.stop()
            self.error_occurred.emit(friendly_error(error, "파일을 재생하지 못했습니다. 지원하는 미디어 파일인지 확인해주세요."))

    def _toggle(self, role):
        if role not in self.ROLES:
            raise ValueError(f"Unknown playback role: {role}")
        if not self.available(role):
            logger.error("Playback source missing or invalid: role=%s path=%r", role, self.sources.get(role))
            if self.active_role == role:
                self.stop()
            self.error_occurred.emit("재생할 파일을 찾을 수 없습니다.")
            return
        if self.active_role == role and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            return
        if self.active_role != role:
            self.stop()
            self.active_role = role
            self.player.setSource(QUrl.fromLocalFile(str(self.sources[role].resolve())))
        self.player.play()

    def stop(self):
        self.player.stop()
        self.active_role = None
        for role in self.ROLES:
            self.state_changed.emit(role, "stopped")

    def _on_state_changed(self, state):
        if self.active_role is not None:
            name = {QMediaPlayer.PlaybackState.PlayingState: "playing",
                    QMediaPlayer.PlaybackState.PausedState: "paused",
                    QMediaPlayer.PlaybackState.StoppedState: "stopped"}[state]
            self.state_changed.emit(self.active_role, name)

    def _on_error(self, error, message):
        if error == QMediaPlayer.Error.NoError:
            return
        logger.error("Qt media error: role=%s path=%r code=%s detail=%s", self.active_role, self.sources.get(self.active_role), error, message)
        self.stop()
        self.error_occurred.emit("파일을 재생하지 못했습니다. 파일이 손상되었거나 지원하지 않는 오디오 형식일 수 있습니다.")
