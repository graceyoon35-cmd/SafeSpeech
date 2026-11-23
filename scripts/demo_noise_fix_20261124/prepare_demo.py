"""Phase 1: real app handoff input; explicit new epsilon profiles and exact ABs."""
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import soundfile as sf
from pystoi import stoi

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(os.environ.get('SAFESPEECH_DEMO_RUN', str(ROOT/'runs/demo_noise_fix_20261124'))).resolve()
OLD=ROOT.parent/'SafeSpeech-desktop-integration-20261118'
sys.path[:0]=[str(ROOT/'scripts'),str(OLD/'scripts')]
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from spec_demo_profile import create_client

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(p,x):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)

def quality(ref,p,epsilon):
    x,sr=sf.read(ref,dtype='float64');y,yr=sf.read(p,dtype='float64');info=sf.info(p)
    assert x.ndim==y.ndim==1 and x.shape==y.shape and sr==yr==44100
    assert info.subtype=='PCM_16' and np.isfinite(y).all()
    e=y-x;power=np.sum(e*e)
    q=dict(path=str(p),sha256=sha(p),sample_rate=sr,bit_depth=16,channels=1,samples=len(y),
        peak=float(np.max(np.abs(y))),rms=float(np.sqrt(np.mean(y*y))),delta_rms=float(np.sqrt(np.mean(e*e))),
        snr_db=float(10*np.log10(np.sum(x*x)/power)) if power else None,
        stoi=float(stoi(x,y,sr,extended=False)),clipping=int(np.count_nonzero((y<=-1)|(y>=32767/32768))),
        linf=float(np.max(np.abs(e))),epsilon=epsilon/255,strict_epsilon_pass=bool(np.max(np.abs(e))<=epsilon/255))
    assert q['clipping']==0 and q['strict_epsilon_pass']
    return q

def main():
    demo=HERE/'demo_wavs';demo.mkdir(exist_ok=False)
    source=OLD/'runs/final_validation_20261120/app_handoff_examples/input.wav'
    original=demo/'00_original.wav';shutil.copyfile(source,original)
    assert sha(source)==sha(original)=='1e6eafa93d90e2c132b03d14f76b7046f633ba77ff79b4de067fc7d53e8b2a8f'
    rows=[dict(condition='original',**quality(original,original,0))]
    paths={}
    for eps,name in [(8.,'01_existing_profile_eps8_iter10'),(2.,'02_candidate_eps2_iter10'),(4.,'03_candidate_eps4_iter10')]:
        client=create_client(OLD,HERE/'app_jobs',execution='local',epsilon=eps,iterations=10)
        result=client.protect(original)
        dest=demo/(name+'.wav');shutil.copyfile(result.output_path,dest)
        assert sha(dest)==sha(result.output_path)
        q=quality(original,dest,eps)
        q.update(condition=name,job=str(result.diagnostics_directory),generation_metadata=str(result.metadata_path))
        rows.append(q);paths[name]=str(dest)
        print(json.dumps(q,ensure_ascii=False),flush=True)
        x,sr=sf.read(original,dtype='int16');y,yr=sf.read(dest,dtype='int16')
        ab=demo/('AB_original_then_'+name+'.wav')
        with ab.open('xb') as f:sf.write(f,np.concatenate([x,np.zeros(round(.6*sr),dtype=np.int16),y]),sr,format='WAV',subtype='PCM_16')
        codes,_=sf.read(ab,dtype='int16')
        assert np.array_equal(codes[:len(x)],x) and np.array_equal(codes[len(x)+round(.6*sr):],y)
    with (HERE/'demo_quality.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
    save(HERE/'demo_result.json',dict(status='QUALITY_CONTRACT_PASS_HUMAN_AND_DEFENSE_PENDING',
        input_origin=str(source),input_sha256=sha(source),text='우리는 서로 답을 비교해 보았다.',
        input_selection='Previously used actual app integration demo input; user has not supplied a replacement.',
        profiles=rows,preferred_first_candidate=paths['02_candidate_eps2_iter10'],human_listening_verified=False,
        defense_verified=False,volume_reduction=False,filters=False,cache_policy='Fresh UUID per request; explicit settings and input/output hashes validated by existing bridge.'))

if __name__=='__main__':main()
