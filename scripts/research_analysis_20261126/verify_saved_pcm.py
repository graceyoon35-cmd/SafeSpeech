"""Recheck existing historical PCM/thresholds without any synthesis or ML inference."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import soundfile as sf


def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def verify(workspace,code_root):
    root=Path(workspace).resolve();code=Path(code_root).resolve()
    sys.path.insert(0,str(code/'scripts/psychoacoustic_spec_pilot'))
    from auditory_masking import build_model,audit
    pilot=root/'runs/psychoacoustic_spec_pilot_20261125/pilot'
    refine=root/'runs/psychoacoustic_spec_refinement_20261126'
    pq=rows(pilot/'quality_per_file.csv');rq=rows(refine/'comparison/quality_per_file.csv')
    clean={r['id']:r for r in pq if r['condition']=='clean'}
    assert len(clean)==5
    checked=[];watched={}
    for uid,source in sorted(clean.items()):
        x,sr=sf.read(source['path'],dtype='int16');assert sr==44100 and x.ndim==1
        assert digest(source['path'])==source['sha256']
        watched[source['path']]=source['sha256']
        models=build_model(x.astype(np.float64)/32768)
        threshold=pilot/'thresholds'/(uid+'.npz');watched[str(threshold)]=digest(threshold)
        with np.load(threshold) as saved:
            radius=saved['radius_codes'].copy()
            for model in models:np.testing.assert_array_equal(model.raw_threshold,saved[f'raw_threshold_{model.n_fft}'])
        jobs=[(Path(r['path']),r['condition'],3 if '3db' in r['condition'] else 6,r['sha256'])
              for r in pq if r['id']==uid and r['condition'].startswith('mask')]
        jobs.extend((Path(r['path']),r['condition'],3,r['sha256'])
                    for r in rq if r['id']==uid and r['condition']!='old_mask3')
        diagnostic=refine/'stage_diagnosis/baseline_trace'/uid/'protected.wav'
        generation=json.loads((diagnostic.parent/'generation.json').read_text())
        jobs.append((diagnostic,'baseline_trace',3,generation['output_sha256']))
        assert len(jobs)==5
        for path,condition,margin,expected in jobs:
            sha=digest(path);assert sha==expected;watched[str(path)]=sha
            y,rate=sf.read(path,dtype='int16');info=sf.info(path)
            assert rate==sr and y.shape==x.shape and info.subtype=='PCM_16' and info.channels==1
            delta=y.astype(np.int32)-x.astype(np.int32)
            max_code=int(abs(delta).max());assert max_code<=257 and np.all(abs(delta)<=radius)
            clipping=int(np.count_nonzero((y==-32768)|(y==32767)));assert clipping==0
            masking=audit(delta/32768.,models,margin);assert all(m['pass_constraint'] for m in masking)
            decode=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-f','null','-'],capture_output=True,text=True)
            assert decode.returncode==0,decode.stderr
            checked.append(dict(id=uid,condition=condition,path=str(path),sha256=sha,margin_db=margin,
                                max_delta_codes=max_code,strict_epsilon_pass=True,radius_pass=True,clipping=clipping,
                                ffmpeg_decode_pass=True,masking=masking))
    assert len(checked)==25 and all(digest(p)==h for p,h in watched.items())
    return dict(status='PASS',existing_files_checked=25,source_ids=sorted(clean),checks=checked,
                inputs_unchanged=True,new_audio=False,new_model_inference=False,new_attack_training=False,
                defense_confirmed_by_this_check=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--code-root',type=Path,default=Path(__file__).resolve().parents[2])
    p.add_argument('--output',type=Path,required=True,help='New JSON path; existing files are refused')
    a=p.parse_args()
    if a.output.exists():p.error('Output already exists')
    result=verify(a.workspace,a.code_root)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print('PASS: 25 saved PCM files, source SHA, dual FFT masking, radius, epsilon, clipping, full decode')


if __name__=='__main__':main()
