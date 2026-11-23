"""Check the UI patch against its exact source and verify playable demo copies."""
import io
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from prepare_demo import HERE,ROOT,OLD,save,sha
sys.path.insert(0,str(OLD/'scripts'))
from run_korean_melo_spec import git

pristine=HERE/'patch_check_source';pristine.mkdir(exist_ok=False)
with zipfile.ZipFile(io.BytesIO(git('archive','--format=zip','d678c2051e336b2be2a595cc37562030c7df1954','ui'))) as z:z.extractall(pristine)
patchpath=ROOT/'patches/spec_demo_ui_20261124.patch'
check=git('apply','--check','--directory='+pristine.relative_to(ROOT).as_posix(),str(patchpath),cwd=ROOT)
assert sha(patchpath)==sha(HERE/'ui_backend.patch')
checks=[]
for p in list((HERE/'demo_wavs').glob('*.wav'))+list((HERE/'final_demo_eps2').glob('*.wav')):
    rate,codes=wavfile.read(p)
    actual,sr=sf.read(p,dtype='float64')
    assert codes.dtype==np.int16 and sr==rate==44100
    assert np.array_equal(codes.astype(np.float64)/32768,actual)
    checks.append(dict(path=str(p),sha256=sha(p),independent_decoder_exact=True))
for name in ('01_existing_profile_eps8_iter10','02_candidate_eps2_iter10','03_candidate_eps4_iter10'):
    original,_=sf.read(HERE/'demo_wavs/00_original.wav',dtype='int16')
    protected,_=sf.read(HERE/'demo_wavs'/(name+'.wav'),dtype='int16')
    ab,_=sf.read(HERE/'demo_wavs'/('AB_original_then_'+name+'.wav'),dtype='int16')
    assert np.array_equal(ab[:len(original)],original)
    assert np.count_nonzero(ab[len(original):len(original)+26460])==0
    assert np.array_equal(ab[len(original)+26460:],protected)
save(HERE/'handoff_verification.json',dict(status='PASS',ui_patch_applies_to_exact_source=True,source_commit='d678c2051e336b2be2a595cc37562030c7df1954',
    patch_sha256=sha(patchpath),wav_checks=checks,AB_samples_preserved=True,old_app_branch_modified=False))
print(json.dumps(dict(status='PASS',checked_wavs=len(checks),patch_apply_check=True)))
