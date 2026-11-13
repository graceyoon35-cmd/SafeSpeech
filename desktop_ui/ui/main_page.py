"""Figma 13:2 / 22:2 / 23:2: final main screen, without strength control."""
import logging
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QMessageBox, QProgressBar

from .start_page import DesignPage, button, label, panel
from file_selection import FileSelection
from protection_processor import DummyProtectionProcessor
from result_saver import prepare_destination, save_result as export_result
from media_playback import MediaPlayback
from error_handling import friendly_error

logger = logging.getLogger(__name__)


class MainPage(DesignPage):
    def __init__(self):
        super().__init__(main=True)
        self.setAcceptDrops(True)
        self.selection = FileSelection()
        self.processor = DummyProtectionProcessor(self)
        self.playback = MediaPlayback(self)
        self.playback.state_changed.connect(self.update_playback_button)
        self.playback.error_occurred.connect(self.on_playback_error)
        self.file_offset = 0
        self.rows = []
        self.current_result = None
        self.last_save_path = None
        header = panel(self.glass, (1, 1, 1038, 68), "header")
        mark = label(header, "V", (24, 19, 30, 30), 13, "#334155", 600, Qt.AlignHCenter)
        mark.setStyleSheet("background: rgba(255,255,255,219); border: 1px solid rgba(215,224,239,204); border-radius: 9px; font-size: 13px; font-weight: 600; color: #334155;")
        label(header, "Voice Protect", (64, 18, 140, 19), 15, "#192434", 600)
        label(header, "딥보이스 방어", (64, 38, 110, 14), 10, "#65758b")
        button(header, "설정", (885, 16, 55, 37)).clicked.connect(self.show_settings)
        button(header, "도움말", (948, 16, 66, 37)).clicked.connect(self.show_help)

        label(self.glass, "음성·영상 보호", (25, 89, 320, 31), 24, "#172033", 600)
        label(self.glass, "파일을 추가하면 AI가 자동으로 딥보이스 방어 처리를 적용해요.", (25, 123, 610, 17), 12, "#667085")
        self.upload = panel(self.glass, (25, 152, 610, 312))
        self.add_button = button(self.upload, "파일 추가", (17, 19, 180, 18), "uploadTitle")
        self.add_button.setStyleSheet("text-align: left; font-size: 14px; font-weight: 600; color: #1e293b; background: transparent; border: none;")
        self.add_button.clicked.connect(self.choose_files)
        self.file_count = label(self.upload, "", (360, 21, 233, 15), 11, "#6b7280", 500, Qt.AlignRight)
        self.drop_button = button(self.upload, "", (17, 48, 576, 92), "drop")
        self.drop_button.setAccessibleName("파일 추가: WAV, MP3, MP4")
        self.drop_button.clicked.connect(self.choose_files)
        title = label(self.drop_button, "여기에 파일을 끌어오세요", (0, 10, 576, 20), 13, "#344054", 500, Qt.AlignHCenter)
        subtitle = label(self.drop_button, "WAV · MP3 · MP4 지원", (0, 32, 576, 15), 10, "#7b8798", 400, Qt.AlignHCenter)
        title.setAttribute(Qt.WA_TransparentForMouseEvents)
        subtitle.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.file_picker_button = button(self.drop_button, "+ 파일 추가", (236, 54, 104, 28), "uploadPicker")
        self.file_picker_button.setAccessibleName("파일 추가: 파일 선택 창 열기")
        self.file_picker_button.clicked.connect(self.choose_files)

        progress = panel(self.glass, (25, 478, 610, 119))
        label(progress, "현재 처리", (17, 17, 190, 18), 13, "#263449", 600)
        self.progress_percent = label(progress, "0%", (530, 17, 63, 18), 12, "#4f67ff", 600, Qt.AlignRight)
        self.current_filename = label(progress, "파일을 선택해주세요", (17, 45, 240, 15), 11, "#536176", 500)
        self.progress_bar = QProgressBar(progress)
        self.progress_bar.setGeometry(17, 70, 576, 8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setValue(0)
        self.processing_status = label(progress, "대기", (17, 90, 200, 14), 10, "#7a8798")
        self.processing_order = label(progress, "0 / 0", (523, 90, 70, 14), 10, "#6f7d90", 500, Qt.AlignRight)

        action = panel(self.glass, (653, 106, 368, 174))
        label(action, "보호 실행", (17, 17, 250, 19), 14, "#223047", 600)
        label(action, "선택한 파일에 딥보이스 방어 처리를 적용합니다.", (17, 44, 334, 14), 10, "#7c8796")
        self.selected_count = None
        for i, (name, value) in enumerate((("선택 파일", "3개"), ("지원 형식", "WAV · MP3 · MP4"), ("처리 방식", "로컬 처리"))):
            label(action, name, (17, 66 + 18*i, 100, 14), 10, "#8290a2")
            field = label(action, value, (157, 66 + 18*i, 194, 14), 10, "#42516a", 500, Qt.AlignRight)
            if i == 0:
                self.selected_count = field
        self.protect_button = button(action, "모든 파일 보호 시작", (17, 124, 334, 38), "primary")
        self.protect_button.clicked.connect(self.protect)

        result = panel(self.glass, (653, 292, 368, 350))
        self.result_card = result
        label(result, "결과", (16, 19, 150, 19), 14, "#223047", 600)
        self.result_badge = self.badge(result, "대기", (311, 16, 39, 23))
        self.result_filename = label(result, "처리 결과가 없습니다", (16, 51, 260, 15), 11, "#536176", 500)
        self.result_names = []
        self.play_buttons = {}
        for i, name in enumerate(("원본", "보호 적용")):
            audio = panel(result, (16, 76 + i*56, 334, 48), "audio")
            play = button(audio, "▶", (10, 10, 28, 28), "play")
            play.setAccessibleName(f"{name} 재생")
            role = "original" if i == 0 else "protected"
            self.play_buttons[role] = play
            play.clicked.connect(lambda checked=False, track=role: self.preview_audio(track))
            label(audio, name, (46, 10, 140, 16), 11, "#344054", 500)
            self.result_names.append(label(audio, "—", (46, 25, 240, 13), 9, "#8491a4"))
            label(audio, "재생", (294, 17, 30, 15), 10, "#68758a", 500, Qt.AlignRight)
        self.result_metrics = []
        for i, (name, value) in enumerate((("처리 시간", "—"), ("보호 상태", "대기"), ("신뢰도", "—"))):
            metric = panel(result, (16 + 114*i, 192, 106, 48), "metric")
            label(metric, name, (10, 9, 86, 14), 9, "#8290a2")
            self.result_metrics.append(label(metric, value, (10, 23, 86, 17), 12, "#314057", 600))
        self.save_button = button(result, "결과 저장", (16, 252, 334, 38))
        self.save_button.clicked.connect(self.save_result)
        self.reset_result()
        self.refresh_files()
        self.processor.running_changed.connect(self.set_processing_controls)
        self.processor.file_started.connect(self.on_file_started)
        self.processor.file_finished.connect(self.on_file_finished)
        self.processor.file_failed.connect(self.on_file_failed)
        self.processor.progress_changed.connect(self.update_progress)
        self.processor.finished.connect(self.on_processing_finished)

    @staticmethod
    def badge(parent, text, rect):
        badge = label(parent, text, rect, 11, "#667085", 500, Qt.AlignHCenter)
        MainPage.set_badge_status(badge, text)
        return badge

    @staticmethod
    def set_badge_status(badge, text):
        color, background = {"완료": ("#277a57", "rgba(232,247,239,219)"), "처리 중": ("#4f67ff", "rgba(238,243,255,219)"), "대기": ("#667085", "rgba(243,245,247,219)"), "실패": ("#b54747", "rgba(255,235,235,219)")}[text]
        badge.setText(text)
        style = f"font-size: 11px; font-weight: 500; color: {color}; background: {background}; border: none; border-radius: 11px;"
        if hasattr(badge, "_design_style"):
            badge._design_style = style
        badge.setStyleSheet(style)

    @property
    def files(self):
        return self.selection.files

    def refresh_files(self):
        for row in self.rows:
            row.hide()
            row.deleteLater()
        self.rows.clear()
        total = sum(item.mb for item in self.files)
        self.file_count.setText(f"{len(self.files)} / 10개 · {total:g}MB")
        self.selected_count.setText(f"{len(self.files)}개")
        # The design shows three rows. Mouse wheel pages through larger queues.
        offset = getattr(self, "file_offset", 0)
        offset = max(0, min(offset, len(self.files) - 3))
        self.file_offset = offset
        for visible_index, item in enumerate(self.files[offset:offset+3]):
            index = offset + visible_index
            row = panel(self.upload, (17, 152 + visible_index*51, 576, 39), "row")
            title = button(row, item.name, (10, 11, 245, 18), "fileName")
            title.setStyleSheet("text-align: left; font-size: 12px; font-weight: 500; color: #253246; background: transparent; border: none;")
            title.clicked.connect(lambda checked=False, path=item.path: self.select_result(path))
            title.setEnabled(not self.processor.running)
            title.setToolTip(str(item.path) + (f"\n{item.error_message}" if item.error_message else ""))
            extension = label(row, item.extension, (265, 12, 72, 16), 11, "#728095")
            extension.setToolTip(f"{item.mb:g} MB")
            status = item.status
            badge = self.badge(row, status, (347, 8, 52 if status == "처리 중" else 39, 23))
            badge.setToolTip(item.error_message)
            remove = button(row, "×", (549, 6, 24, 27), "remove")
            remove.setAccessibleName(f'{item.name} 삭제')
            remove.setEnabled(not self.processor.running)
            remove.clicked.connect(lambda checked=False, idx=index: self.remove_file(idx))
            row.show()
            self.rows.append(row)
        self.adapt_layout()

    def wheelEvent(self, event):
        self.file_offset += -1 if event.angleDelta().y() > 0 else 1
        self.refresh_files()
        event.accept()

    def remove_file(self, index):
        if self.processor.running:
            return
        self.selection.remove(index)
        self.refresh_files()
        self.reset_progress()

    def choose_files(self):
        if self.processor.running:
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "파일 추가", "", "음성·영상 (*.wav *.mp3 *.mp4 *.WAV *.MP3 *.MP4)")
        if paths:
            self.add_files(paths)

    def add_files(self, paths):
        if self.processor.running:
            return
        before = self.files
        rejected = self.selection.add(paths)
        self.file_offset = max(0, len(self.files) - 3)
        self.refresh_files()
        if self.files != before:
            self.reset_progress()
        if rejected:
            details = "\n".join(f"{name}: {reason}" for name, reason in rejected)
            QMessageBox.information(self, "파일 추가", "추가하지 못한 파일:\n" + details)

    def dragEnterEvent(self, event):
        if not self.processor.running and event.mimeData().hasUrls() and any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        if self.processor.running:
            event.ignore()
            return
        self.add_files([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
        event.acceptProposedAction()

    def protect(self):
        if self.processor.running:
            return
        if not self.files:
            QMessageBox.warning(self, "보호 실행", "보호할 파일을 먼저 추가해주세요.")
            return
        for index in range(len(self.files)):
            self.selection.set_status(index, "대기")
        self.processor.start(self.files)

    def reset_progress(self):
        self.reset_result()
        self.update_progress(0)
        self.current_filename.setText("시작 대기" if self.files else "파일을 선택해주세요")
        self.current_filename.setToolTip("")
        self.processing_status.setText("대기")
        self.processing_order.setText(f"0 / {len(self.files)}")

    def update_progress(self, percent):
        self.progress_bar.setValue(percent)
        self.progress_percent.setText(f"{percent}%")

    def set_processing_controls(self, running):
        if running:
            self.reset_result(processing=True)
        self.add_button.setEnabled(not running)
        self.drop_button.setEnabled(not running)
        self.protect_button.setEnabled(not running)
        self.protect_button.setText("보호 처리 중" if running else "모든 파일 보호 시작")
        self.refresh_files()

    def on_file_started(self, index, total):
        self.selection.set_status(index, "처리 중")
        self.current_filename.setText(self.files[index].name)
        self.current_filename.setToolTip(str(self.files[index].path))
        self.processing_order.setText(f"{index + 1} / {total}")
        self.processing_status.setText("더미 보호 처리 중")
        # Keep the active file visible within the original three-row panel.
        self.file_offset = min(index, max(0, total - 3))
        self.refresh_files()

    def on_file_finished(self, index):
        self.selection.set_status(index, "완료")
        self.refresh_files()

    def on_file_failed(self, index, message):
        self.selection.set_status(index, "실패", message)
        self.refresh_files()

    def on_processing_finished(self):
        self.processing_status.setText("모든 파일 더미 처리 완료")
        if self.processor.results:
            self.show_result(self.processor.results[-1])
        if self.processor.failures:
            failures = self.processor.failures
            self.processing_status.setText(f"더미 처리 종료 · 완료 {len(self.processor.results)} · 실패 {len(failures)}")
            if not self.processor.results:
                self.result_filename.setText("완료된 결과가 없습니다")
                self.set_badge_status(self.result_badge, "실패")
                self.result_metrics[1].setText("실패")
                self.adapt_layout()
            details = "\n".join(f"{self.files[i].name}: {message}" for i, message in failures.items())
            QMessageBox.warning(self, "일부 파일 처리 실패", f"완료 {len(self.processor.results)}개, 실패 {len(failures)}개\n실패한 파일을 확인해주세요.\n\n{details}")

    def reset_result(self, processing=False):
        self.playback.set_sources()
        self.current_result = None
        self.last_save_path = None
        self.result_card.setEnabled(False)
        self.save_button.setEnabled(False)
        self.set_badge_status(self.result_badge, "대기")
        self.result_filename.setText("처리 완료 후 결과가 표시됩니다" if processing else "처리 결과가 없습니다")
        self.result_filename.setToolTip("")
        for field in self.result_names:
            field.setText("—")
            field.setToolTip("")
        for field, value in zip(self.result_metrics, ("—", "처리 중" if processing else "대기", "—")):
            field.setText(value)
        self.adapt_layout()

    def show_result(self, result):
        self.playback.set_sources(result.original_path, result.output_path)
        self.current_result = result
        self.last_save_path = None
        self.result_card.setEnabled(True)
        self.save_button.setEnabled(True)
        self.set_badge_status(self.result_badge, "완료")
        self.result_filename.setText(result.original_path.name)
        self.result_filename.setToolTip(str(result.original_path))
        for field, name in zip(self.result_names, (result.original_path.name, result.protected_name)):
            field.setText(name)
            field.setToolTip(name)
        for field, value in zip(self.result_metrics, (f"{result.elapsed_seconds:.1f}초", "완료", f"{result.confidence}%")):
            field.setText(value)
        self.result_metrics[2].setToolTip("고정 테스트값이며 실제 AI 분석 수치가 아닙니다.")
        for role, play in self.play_buttons.items():
            available = self.playback.available(role)
            play.setEnabled(available)
            play.setToolTip("재생" if available else "재생할 실제 파일이 아직 없습니다.")
        self.adapt_layout()

    def select_result(self, path):
        if self.processor.running or self.current_result is None:
            return
        result = next((item for item in self.processor.results if item.original_path == path), None)
        if result is not None:
            self.show_result(result)

    def save_result(self):
        if self.current_result is None or self.processor.running:
            return
        result = self.current_result
        suffix = result.original_path.suffix
        selected, _ = QFileDialog.getSaveFileName(
            self, "결과 저장", str(result.original_path.parent / result.protected_name),
            f"보호 결과 (*{suffix.lower()} *{suffix.upper()})",
            options=QFileDialog.Option.DontConfirmOverwrite,
        )
        if not selected:
            return
        try:
            destination = prepare_destination(result, selected)
            if result.output_path is not None and destination.exists():
                answer = QMessageBox.question(
                    self, "덮어쓰기 확인", f"이미 존재하는 파일입니다. 덮어쓸까요?\n{destination}",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
            outcome = export_result(result, destination)
        except Exception as error:
            logger.exception("Result export failed: result=%r destination=%r", result, selected)
            message = friendly_error(error, "결과를 저장하지 못했습니다. 저장 경로와 파일명을 확인해주세요.")
            QMessageBox.critical(self, "저장 실패", message)
            return
        self.last_save_path = outcome.destination
        if outcome.written:
            QMessageBox.information(self, "저장 완료", f"결과 파일을 저장했습니다.\n{outcome.destination}")
        else:
            QMessageBox.information(self, "테스트 저장 경로 확인", f"저장 경로를 선택했습니다.\n{outcome.destination}\n\n아직 실제 결과 파일이 없어 파일은 생성하지 않았습니다.")

    def preview_audio(self, role):
        if self.current_result is not None and not self.processor.running:
            self.playback.toggle(role)

    def update_playback_button(self, role, state):
        play = self.play_buttons[role]
        name = "원본" if role == "original" else "보호 적용"
        action = "일시정지" if state == "playing" else "재생"
        play.setText("Ⅱ" if state == "playing" else "▶")
        play.setToolTip(f"{name} {action}")
        play.setAccessibleName(f"{name} {action}")

    def on_playback_error(self, message):
        QMessageBox.warning(self, "오디오 재생", message)

    def hideEvent(self, event):
        self.playback.stop()
        super().hideEvent(event)

    def show_settings(self):
        QMessageBox.information(self, "설정", "처리 방식: 로컬 처리\n지원 형식: WAV · MP3 · MP4\n최대 파일 수: 10개\nAI 보호 모델: 미연결")

    def show_help(self):
        QMessageBox.information(self, "도움말", "파일 추가 버튼이나 영역을 클릭하거나 파일을 끌어 놓으세요.\n× 버튼으로 파일을 삭제할 수 있습니다.\n파일이 4개 이상이면 마우스 휠로 목록을 이동합니다.\n\n보호 시작은 파일당 약 1.5초의 더미 처리입니다.\n완료 후 목록의 파일명을 클릭하면 해당 결과를 확인할 수 있습니다.\n실제 파일은 생성하지 않으며 신뢰도는 고정 테스트값입니다.")
