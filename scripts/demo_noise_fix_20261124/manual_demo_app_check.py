"""Offscreen real app workflow: dialogs automated, real Qt/CLI/CUDA/export.
This does not claim human listening or native Windows audio-device playback.
"""
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

from prepare_demo import HERE,ROOT,OLD,save,sha,quality
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['SAFESPEECH_SPEC_REPOSITORY']=str(OLD)
os.environ['SAFESPEECH_DEMO_OUTPUT_ROOT']=str(HERE/'gui_jobs')
sys.path[:0]=[str(HERE/'app_snapshot/ui'),str(ROOT/'scripts'),str(OLD/'scripts')]
from PySide6.QtCore import QEventLoop,QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QFileDialog,QMessageBox
from PySide6.QtMultimedia import QMediaPlayer
from main import create_app,VoiceProtect

def main():
    app=create_app();window=VoiceProtect();page=window.main_page
    window.show();window.setCurrentWidget(page)
    messages=[];ticks=[]
    timer=QTimer();timer.setInterval(10);timer.timeout.connect(lambda:ticks.append(True));timer.start()
    source=HERE/'demo_wavs/00_original.wav';before=sha(source)
    with patch.object(QFileDialog,'getOpenFileNames',return_value=([str(source)],'WAV')),\
         patch.object(QMessageBox,'warning',side_effect=lambda *args:messages.append(str(args[-1]))),\
         patch.object(QMessageBox,'information',side_effect=lambda *args:messages.append(str(args[-1]))),\
         patch.object(QMessageBox,'critical',side_effect=lambda *args:messages.append(str(args[-1]))):
        page.choose_files()
        assert len(page.files)==1
        cfg=page.processor.client.config
        assert cfg.epsilon==2 and cfg.iterations==10 and cfg.seed==52
        jobs=[]
        for attempt in (1,2):
            page.protect()
            assert page.processor.running and page.current_result is None
            assert not page.save_button.isEnabled() and page.playback.sources=={'original':None,'protected':None}
            event=QCloseEvent();window.closeEvent(event);assert not event.isAccepted()
            deadline=time.monotonic()+240
            while page.processor.running:
                app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents,50)
                time.sleep(.004)
                if time.monotonic()>deadline:raise RuntimeError('GUI processing timeout')
            # Drain slots after thread finished.
            for _ in range(10):app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents,50)
            assert not page.processor.failures and page.current_result is not None
            result=page.current_result
            assert page.playback.sources['original']==source
            assert page.playback.sources['protected']==result.output_path
            assert page.play_buttons['original'].isEnabled() and page.play_buttons['protected'].isEnabled()
            assert page.result_metrics[2].text()=='미측정'
            assert result.confidence is None
            metadata=json.loads(result.metadata_path.read_text())
            assert (metadata['epsilon_units_255'],metadata['iterations'])==(2,10)
            assert all(str(result.output_path)!=r['output_path'] for r in jobs)
            export=HERE/f'demo_wavs/04_app_saved_eps2_attempt{attempt}.wav'
            with patch.object(QFileDialog,'getSaveFileName',return_value=(str(export),'WAV')):
                page.save_result()
            assert page.last_save_path==export and sha(export)==sha(result.output_path)
            q=quality(source,export,2)
            jobs.append(dict(attempt=attempt,output_path=str(result.output_path),saved_path=str(export),sha256=sha(export),quality=q))
        # Actual Qt player invocation; distinguish loading from physical playback.
        playback=[]
        for role in ('original','protected'):
            page.preview_audio(role)
            deadline=time.monotonic()+8
            while time.monotonic()<deadline:
                app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents,50);time.sleep(.01)
                if page.playback.player.mediaStatus() in (QMediaPlayer.MediaStatus.LoadedMedia,QMediaPlayer.MediaStatus.BufferedMedia,QMediaPlayer.MediaStatus.EndOfMedia,QMediaPlayer.MediaStatus.InvalidMedia):break
            playback.append(dict(role=role,source=page.playback.player.source().toLocalFile(),
                media_status=str(page.playback.player.mediaStatus()),playback_state=str(page.playback.player.playbackState()),
                qt_error=str(page.playback.player.error()),error_string=page.playback.player.errorString()))
            page.playback.stop()
    assert sha(source)==before
    timer.stop();window.close();app.processEvents()
    save(HERE/'app_workflow_test.json',dict(status='OFFSCREEN_WORKFLOW_PASS',source_commit='d678c2051e336b2be2a595cc37562030c7df1954',
        upload_dialog_exercised=True,real_gpu_protection=True,save_dialog_exercised=True,
        fresh_job_on_repeat=True,old_playback_cleared=True,close_blocked_during_processing=True,
        settings=dict(epsilon_units_255=2,iterations=10,seed=52),jobs=jobs,heartbeat_count=len(ticks),playback=playback,messages=messages,
        scope='Patched app snapshot, offscreen Qt under WSL. File dialog paths automated. QMediaPlayer invoked; native Windows GUI and physical audio output not verified.',
        human_listening_verified=False,app_team_branch_modified=False))
    print(json.dumps(dict(status='OFFSCREEN_WORKFLOW_PASS',heartbeat_count=len(ticks),playback=playback),ensure_ascii=False))

if __name__=='__main__':main()
