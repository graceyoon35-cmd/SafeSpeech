"""Exercise real UI snapshot data modules, Qt worker, SPEC GPU CLI and export."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def smoke(root, review):
    sys.path.insert(0,str(root/'scripts'))
    sys.path.insert(0,str(review/'app_snapshot/ui'))
    from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer
    from file_selection import FileSelection
    from result_saver import save_result
    from spec_desktop_bridge import SpecClient, SpecConfig, sha256
    from spec_desktop_processor import SpecProtectionProcessor
    source=root/'runs/final_validation_20261120/app_handoff_examples/input.wav'
    before=sha256(source)
    inputs=review/'qt_smoke_inputs'
    inputs.mkdir(exist_ok=False)
    selected=inputs/"한글 입력 '원본'.wav"
    shutil.copyfile(source,selected)
    mp3=inputs/'지원 확인.mp3'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-n','-i',str(source),'-b:a','128k',str(mp3)],check=True)
    selection=FileSelection()
    assert selection.add([mp3,selected])==[]
    application=QCoreApplication.instance() or QCoreApplication([])
    client=SpecClient(SpecConfig(root,review/'qt_smoke_jobs',execution='local'))
    processor=SpecProtectionProcessor(client)
    events=[];ticks=[];finished=[]
    def started(index,total):
        selection.set_status(index,'처리 중')
        events.append({'event':'file_started','index':index,'total':total})
    def succeeded(index):
        selection.set_status(index,'완료')
        events.append({'event':'file_finished','index':index})
    def failed(index,message):
        selection.set_status(index,'실패',message)
        events.append({'event':'file_failed','index':index,'message':message})
    processor.file_started.connect(started)
    processor.file_finished.connect(succeeded)
    processor.file_failed.connect(failed)
    processor.finished.connect(lambda:finished.append(True))
    timer=QTimer();timer.setInterval(10);timer.timeout.connect(lambda:ticks.append(True));timer.start()
    assert processor.start(selection.files)
    assert not processor.start(selection.files)
    limit=time.monotonic()+180
    while not finished and time.monotonic()<limit:
        application.processEvents(QEventLoop.ProcessEventsFlag.AllEvents,50)
        time.sleep(.003)
    timer.stop()
    if not finished:
        # Do not terminate a live CUDA child; leave this process until it exits.
        while processor.running:
            application.processEvents(QEventLoop.ProcessEventsFlag.AllEvents,50)
            time.sleep(.01)
        raise RuntimeError('SPEC smoke exceeded 180 seconds')
    assert not processor.running and len(processor.results)==1 and set(processor.failures)=={0}
    assert [r.status for r in selection.files]==['실패','완료']
    assert len(ticks)>5
    result=processor.results[0]
    export=review/'qt_exported_protected.wav'
    outcome=save_result(result,export)
    assert outcome.written and sha256(export)==sha256(result.output_path)
    assert result.confidence is None
    assert sha256(source)==before and sha256(selected)==before
    report={'status':'PASS','ui_commit':'d678c2051e336b2be2a595cc37562030c7df1954',
        'scope':'QCoreApplication only; actual file_selection/result_saver source, adapter Qt signals and real GPU CLI. No visual UI or audio-device playback test.',
        'batch':{'selected':2,'expected_unsupported_mp3':1,'wav_success':1,'continued_after_failure':True},
        'event_loop_heartbeat_count':len(ticks),'events':events,'result':result.as_dict(),
        'exported_path':str(export),'exported_sha256':sha256(export),'originals_unchanged':True,
        'quality':json.loads(result.metadata_path.read_text())}
    with (review/'qt_contract_smoke.json').open('x',encoding='utf-8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
    print(json.dumps({'status':'PASS','scope':report['scope'],'events':events,'ticks':len(ticks)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',type=Path,required=True)
    p.add_argument('--review-root',type=Path,required=True)
    a=p.parse_args()
    smoke(a.repository.resolve(),a.review_root.resolve())
