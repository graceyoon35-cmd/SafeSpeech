"""Prepare reviewable UI patch in our run, without touching feature/ui."""
import difflib
import io
import json
import sys
import zipfile
from pathlib import Path
from prepare_demo import HERE,OLD,save,sha
sys.path.insert(0,str(OLD/'scripts'))
from run_korean_melo_spec import git

revision='d678c2051e336b2be2a595cc37562030c7df1954'
snapshot=HERE/'app_snapshot';snapshot.mkdir(exist_ok=False)
payload=git('archive','--format=zip',revision,'ui')
with zipfile.ZipFile(io.BytesIO(payload)) as z:z.extractall(snapshot)
ui=snapshot/'ui'
changes=[]
def edit(relative,replacements):
    p=ui/relative;before=p.read_text(encoding='utf-8');after=before
    for old,new in replacements:
        if old not in after:raise ValueError(f'Expected UI source missing: {relative}: {old[:60]}')
        after=after.replace(old,new)
    p.write_text(after,encoding='utf-8')
    changes.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/ui/'+relative,tofile='b/ui/'+relative))

edit('ui/main_page.py',[
    ('from protection_processor import DummyProtectionProcessor','from spec_demo_app import build_processor'),
    ('self.processor = DummyProtectionProcessor(self)','self.processor = build_processor(self)'),
    ('WAV · MP3 · MP4','WAV (44.1kHz PCM16 mono)'),
    ('음성·영상 (*.wav *.mp3 *.mp4 *.WAV *.MP3 *.MP4)','WAV (*.wav *.WAV)'),
    ('더미 보호 처리 중','SPEC 보호 처리 중'),
    ('모든 파일 더미 처리 완료','모든 파일 처리 완료'),
    ('더미 처리 종료','처리 종료'),
    ('f"{result.confidence}%"','"미측정" if result.confidence is None else f"{result.confidence}%"'),
    ('고정 테스트값이며 실제 AI 분석 수치가 아닙니다.','복제 방어 확률은 측정하지 않았습니다.'),
    ('AI 보호 모델: 미연결','SPEC: ε=2/255, 10회, seed52'),
    ('보호 시작은 파일당 약 1.5초의 더미 처리입니다.','보호 시작은 실제 SPEC GPU 처리를 실행합니다.'),
    ('실제 파일은 생성하지 않으며 신뢰도는 고정 테스트값입니다.','실제 WAV를 생성하며 방어 확률은 표시하지 않습니다.'),
])
edit('file_selection.py',[( 'frozenset((".wav", ".mp3", ".mp4"))','frozenset((".wav",))')])
# Refuse close while the real CUDA child is active; do not kill training/jobs.
edit('main.py',[( '\n\ndef create_app():','\n\n    def closeEvent(self, event):\n        if self.main_page.processor.running:\n            event.ignore()\n            return\n        self.main_page.playback.stop()\n        super().closeEvent(event)\n\n\ndef create_app():')])
with (HERE/'ui_backend.patch').open('x',encoding='utf-8') as f:f.writelines(changes)
save(HERE/'ui_preparation.json',dict(source_commit=revision,source_branch='origin/feature/ui (locally available ref; not fetched)',
    output=str(ui),patch=str(HERE/'ui_backend.patch'),app_branch_modified=False,
    required_modules=[str(HERE.parent.parent/'scripts'/p) for p in ('spec_demo_profile.py','spec_demo_app.py','launch_spec_demo.py')],
    settings=dict(epsilon_units_255=2,iterations=10,seed=52),
    cache_policy='No persistent WAV cache; fresh UUID job each call, prior result/playback cleared at new processing. Never select optimized.wav or a globbed latest result.'))
print(json.dumps({'ui_snapshot':str(ui),'status':'PREPARED_FOR_TEST'},ensure_ascii=False))
