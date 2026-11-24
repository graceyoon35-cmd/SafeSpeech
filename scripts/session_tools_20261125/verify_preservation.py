"""Read-only audio/result/checkpoint checks; no training or audio generation."""
import csv
import gc
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import wave
import zipfile

import numpy as np
import soundfile as sf

from workspace_paths import ROOT, CLOSEOUT as OUT, STUDY, LISTENING as LISTEN
ATTACK=STUDY/'defense_validation_20261124'
CACHE={}


def sha(p):
    p=Path(p)
    if str(p) not in CACHE:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        CACHE[str(p)]=h.hexdigest()
    return CACHE[str(p)]


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def save(p,o):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)


def pcm(p):
    with wave.open(str(p),'rb') as f:
        assert f.getnchannels()==1 and f.getsampwidth()==2 and f.getframerate()==44100
        return f.readframes(f.getnframes())


def main():
    packet=read(LISTEN/'package_verification.json')
    zpath=LISTEN/'SafeSpeech_listening_20261125.zip'
    assert sha(zpath)==packet['zip_sha256']
    assert zpath.stat().st_size==packet['zip_bytes']
    mapping=rows(LISTEN/'organizer_private/answer_mapping.csv')
    assert len(mapping)==15 and len({r['utterance_id'] for r in mapping})==5
    expected={r['name']:r for r in packet['zip_members']}
    with zipfile.ZipFile(zpath) as z:
        assert z.testzip() is None and set(z.namelist())==set(expected) and len(z.namelist())==16
        assert sum(n.endswith('.wav') for n in z.namelist())==15
        for name,item in expected.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==item['sha256']
            assert sha(LISTEN/'evaluator'/name)==item['sha256']
    for r in mapping:
        src=Path(r['source_path']);dst=LISTEN/'evaluator'/r['filename']
        # Mapping contains Windows paths; convert for Linux without changing files.
        if str(src).startswith('C:'):src=Path('/mnt/c')/str(src)[3:].replace('\\','/')
        assert sha(src)==r['source_sha256'] and sha(dst)==r['copied_sha256']
        data=pcm(src);copy=pcm(dst);assert data==copy
        assert hashlib.sha256(data).hexdigest()==r['source_pcm_sha256']==r['copied_pcm_sha256']
        decode=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(dst),'-f','null','-'],capture_output=True)
        assert decode.returncode==0
    for uid in {r['utterance_id'] for r in mapping}:
        group=[r for r in mapping if r['utterance_id']==uid]
        assert {r['condition'] for r in group}=={'clean','eps2_iter10','silence_only_smooth'}
        assert len({r['samples'] for r in group})==1
    for p,h in read(LISTEN/'organizer_private/source_hashes.json').items():assert sha(p)==h
    print('Listening ZIP: 15 WAV + 1 guide; full CRC, SHA256, source/copy PCM and decode PASS',flush=True)

    artifacts=read(STUDY/'artifact_checksums.json')['files']
    for r in artifacts:
        p=STUDY/r['relative_path'];assert p.is_file() and p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
    q=rows(STUDY/'analysis/per_file.csv');clean={r['utterance_id']:r for r in q if r['condition']=='clean'}
    variants=('silence_only_smooth','silence_floor10_smooth','voice_adaptive_smooth')
    audio=[]
    for variant in variants:
        items=[r for r in q if r['condition']==variant];assert len(items)==140
        for r in items:
            p=Path(r['path']);assert sha(p)==r['sha256']
            x,sr=sf.read(clean[r['utterance_id']]['path'],dtype='int16');y,ysr=sf.read(p,dtype='int16')
            assert sr==ysr==44100 and x.shape==y.shape and y.ndim==1 and sf.info(p).subtype=='PCM_16'
            assert int(np.max(np.abs(y.astype(np.int32)-x.astype(np.int32))))<=257
            assert not np.any((y==-32768)|(y==32767))
            audio.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    assert len(list((STUDY/'candidates').glob('*/*.wav')))==len(audio)==420
    print('Candidates: 3 x 140 = 420 WAV, stored SHA256/strict PCM16 epsilon/clipping PASS',flush=True)
    preserved=read(ATTACK/'postrun_state.json')
    for r in preserved['small_artifacts_sha256']:
        p=ATTACK/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],str(p)
    evaluated=list((ATTACK/'runs').glob('melo_smoke_evaluation_*/evaluation.json'));assert len(evaluated)==1
    assert read(evaluated[0])['status']=='EVALUATION_PASS'
    per=rows(evaluated[0].parent/'per_file.csv');assert len(per)==150
    assert len({r['condition'] for r in per})==5
    for r in per:assert sha(r['output'])==r['output_sha256']
    assert (ATTACK/'comparison.csv').is_file() and (ATTACK/'results.json').is_file()
    print('SIM/CER: 5 conditions x 30 prompts; CSV/JSON and 150 synthesized WAV hashes PASS',flush=True)

    import torch
    checkpoints=[]
    for m in sorted(ATTACK.glob('checkpoints/*/checkpoints/step_001000/checkpoint.json')):
        committed=read(m);assert committed['step']==1000 and committed['schema_version']==2
        run=m.parents[2];cfg=read(run/'config.json');report=read(run/'training_report.json')
        assert report['completed_steps']==1000 and report['status']=='TRAINING_SMOKE_PASS'
        assert hashlib.sha256(json.dumps(cfg,sort_keys=True,ensure_ascii=False).encode()).hexdigest()==committed['config_sha256']
        assert sha(cfg['experiment']['dataset_manifest'])==committed['dataset_manifest_sha256']
        files=[]
        for kind,item in committed['files'].items():
            p=m.parent/item['path'];assert sha(p)==item['sha256']
            files.append(dict(kind=kind,path=str(p),sha256=sha(p),bytes=p.stat().st_size))
        state=torch.load(m.parent/committed['files']['state']['path'],map_location='cpu',weights_only=True)
        assert state['step']==1000 and len(state['history'])==1000
        assert {'discriminator','duration_discriminator','optimizers','rng','initial_probe'}<=set(state)
        assert len(state['optimizers'])==3 and {'torch','cuda','numpy','python'}<=set(state['rng'])
        assert state['rng']['cuda'] is not None
        optimizer_steps=[]
        for optimizer in state['optimizers']:
            steps=sorted({int(v['step']) for v in optimizer['state'].values() if 'step' in v})
            assert steps==[1000];optimizer_steps.append(steps)
        checkpoints.append(dict(condition=report['condition'],step=1000,optimizer_steps=optimizer_steps,
            rng_keys=list(state['rng']),files=files))
        del state;gc.collect()
    assert len(checkpoints)==2
    print('Two model checkpoints + three optimizer states each + Torch/CUDA/NumPy/Python RNG: PASS',flush=True)
    processes=subprocess.check_output(['ps','-eo','pid,comm,args'],text=True)
    active=[line for line in processes.splitlines() if any(name in line for name in
        ('train_melo_bounded.py','run_defense_validation.py','evaluate_paired_melo_smoke.py','summarize_defense_validation.py'))]
    assert not active,active
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu,temperature.gpu','--format=csv'],text=True)
    save(OUT/'preservation_verification.json',dict(status='PASS',zip=str(zpath),zip_sha256=sha(zpath),zip_crc_all_pass=True,
        listening_wavs=15,zip_members=16,mapping_csv=str(LISTEN/'organizer_private/answer_mapping.csv'),
        candidate_wavs=audio,checkpoints=checkpoints,sim_cer_evaluation=str(evaluated[0]),synthesized_wavs=len(per),
        active_training_evaluation_processes=active,nvidia_smi=gpu,verified_file_hashes=CACHE,
        no_training_or_experiments_started=True,no_files_deleted_moved_or_overwritten=True))
    print('Preservation verification saved; active training/evaluation processes: 0',flush=True)


if __name__=='__main__':main()
