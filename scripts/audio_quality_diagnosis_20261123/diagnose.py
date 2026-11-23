"""Read-only historical audit; fresh SPEC optimization, never clone training.
Run with existing WSL melo-training Python and -B. All writes stay in OUT.
"""
import csv
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import soundfile as sf
from pystoi import stoi

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('SAFESPEECH_DIAGNOSIS_RUN', str(ROOT/'runs/audio_quality_diagnosis_20261123'))).resolve()
OLD = ROOT.parent / 'SafeSpeech-desktop-integration-20261118'
sys.path.insert(0, str(OLD/'scripts'))
from run_korean_melo_spec import check_assets, load_backend
from protect_wav import project_pcm16

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def save(p, x):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(x, f, ensure_ascii=False, indent=2, allow_nan=False)

def table(p, rows):
    with Path(p).open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)

watched = {}
def watch(p):
    p = Path(p)
    if str(p) not in watched: watched[str(p)] = digest(p)
    return p

def load(p):
    watch(p)
    a, sr = sf.read(p, dtype='float64', always_2d=True)
    if not len(a) or not np.isfinite(a).all(): raise ValueError(f'Invalid audio: {p}')
    return a, sr

def metrics(p, ref):
    a, sr = load(p); x, xr = load(ref); info = sf.info(p)
    if a.shape != x.shape or sr != xr: raise ValueError('No resample, alignment, trimming or broadcasting permitted')
    e = a-x
    rms = lambda v: float(np.sqrt(np.mean(v*v)))
    db = lambda v: 20*math.log10(v) if v else None
    power = float(np.sum(e*e))
    # Lowest 20% clean-energy 20ms frames, explicitly a relative quiet mask.
    n = int(sr*.02); m = len(x)//n
    energy = np.mean(x[:m*n].reshape(m,n,-1)**2, axis=(1,2))
    mask = energy <= np.quantile(energy,.2)
    qe = e[:m*n].reshape(m,n,-1)[mask]
    spec = np.abs(np.fft.rfft(e[:,0]))**2
    high = np.fft.rfftfreq(len(e),1/sr) >= 8000
    lo, hi = -1., (32767/32768 if info.subtype == 'PCM_16' else 1.)
    return dict(path=str(p), sha256=digest(p), sample_rate=sr, bit_depth={'PCM_16':16,'PCM_24':24,'PCM_32':32,'FLOAT':32,'DOUBLE':64}.get(info.subtype),
        subtype=info.subtype, channels=info.channels, samples=len(a), seconds=len(a)/sr,
        peak=float(np.max(np.abs(a))), rms=rms(a), rms_dbfs=db(rms(a)),
        snr_db=10*math.log10(float(np.sum(x*x))/power) if power else None,
        snr_status='finite' if power else 'infinite_identical_reference',
        stoi=float(np.mean([stoi(x[:,c],a[:,c],sr,extended=False) for c in range(a.shape[1])])),
        rail_samples=int(np.count_nonzero((a<=lo)|(a>=hi))), out_of_range_samples=int(np.count_nonzero((a < -1)|(a>1))),
        linf=float(np.max(np.abs(e))), delta_rms=rms(e), delta_dbfs=db(rms(e)),
        quiet20_delta_rms=rms(qe), delta_energy_above_8khz=float(spec[high].sum()/spec.sum()) if spec.sum() else 0.)

def compare(a,b):
    x,sr=load(a); y,yr=load(b)
    same=x.shape==y.shape and sr==yr
    return dict(a=str(a),b=str(b),sha256_a=digest(a),sha256_b=digest(b),
        file_bytes_identical=digest(a)==digest(b), format_shape_match=same,
        decoded_samples_identical=bool(np.array_equal(x,y)) if same else False,
        different_samples=int(np.count_nonzero(x!=y)) if same else None,
        max_abs_difference=float(np.max(np.abs(x-y))) if same else None)

def write_audio(p,a,sr,subtype='PCM_16'):
    with Path(p).open('xb') as f: sf.write(f,a,sr,format='WAV',subtype=subtype)

def audit():
    review=OLD/'runs/app_integration_review_20261120'
    contract=json.loads(watch(review/'qt_contract_smoke.json').read_text(encoding='utf-8'))
    ref=Path(contract['result']['original_path']); direct=Path(contract['result']['output_path']); exported=Path(contract['exported_path'])
    trio=[dict(role=role, **metrics(p,ref)) for role,p in [('original',ref),('direct_SPEC',direct),('app_saved',exported)]]
    table(OUT/'app_trio.csv',trio)
    save(OUT/'app_comparison.json',compare(direct,exported))
    optimized=direct.parent/'optimized.wav'
    table(OUT/'serialization.csv',[dict(role=role,**metrics(p,ref)) for role,p in [('unprojected',optimized),('projected',direct)]])
    save(OUT/'projection_difference.json',compare(optimized,direct))
    manifest=json.loads(watch(OLD/'runs/korean_melo_spec_mqxs1q1s/handoff_manifest.json').read_text(encoding='utf-8'))
    rows=[]; references={}
    for i,r in enumerate(manifest['utterances']):
        uid=r['utterance_id']; clean=Path(r['clean_path']); references[uid]=clean
        if digest(watch(clean))!=r['clean_sha256'] or digest(watch(r['protected_path']))!=r['protected_sha256']: raise ValueError('Historical hash mismatch')
        rows.append(dict(utterance_id=uid,condition=f"SPEC_{r['iterations']}_historical",**metrics(r['protected_path'],clean)))
        if r['iterations']==10:
            corrected=OLD/'runs/final_validation_20261119/strict_corrected_inputs'/f'{uid}.wav'
            rows.append(dict(utterance_id=uid,condition='SPEC_10_corrected',**metrics(corrected,clean)))
            rows.append(dict(utterance_id=uid,condition='original',**metrics(clean,clean)))
        if i%50==0: print('audit',i,flush=True)
    table(OUT/'historical_files.csv',rows)
    summaries=[]
    for condition in sorted({r['condition'] for r in rows}):
        rr=[r for r in rows if r['condition']==condition]
        summaries.append(dict(condition=condition,n=len(rr),mean_snr_db=float(np.mean([r['snr_db'] for r in rr])) if rr[0]['snr_db'] is not None else None,
            mean_stoi=float(np.mean([r['stoi'] for r in rr])),mean_delta_rms=float(np.mean([r['delta_rms'] for r in rr])),
            rail_samples=sum(r['rail_samples'] for r in rr), max_linf=max(r['linf'] for r in rr)))
    save(OUT/'historical_summary.json',summaries)
    # Preselect five evenly ranked clean recordings by corrected SNR, BEFORE new optimization.
    ranked=sorted([r for r in rows if r['condition']=='SPEC_10_corrected'], key=lambda r:r['snr_db'])
    selected=[ranked[i]['utterance_id'] for i in (0,49,99,149,199)]
    save(OUT/'selection.json',dict(ids=selected,policy='Five SNR order statistics (0,49,99,149,199) of 200 historical corrected files; diagnostic purposive subset, not random or held-out defense evaluation.'))
    return references,selected,trio,summaries

def candidates(references,selected):
    folder=OUT/'fresh_SPEC'; folder.mkdir(exist_ok=False)
    assets,config,hashes=check_assets('/home/safespeech/models/melo-spec-pinned-20261118')
    for p in hashes: watch(assets/p)
    watch(assets/'MeloTTS/.git/HEAD')
    backend=folder/'backend'; backend.mkdir()
    generate=load_backend(assets,config,backend,'cuda')
    results=[]
    for uid in selected:
        clean=references[uid]; x,sr=load(clean)
        for eps in (2.,4.,8.):
            for it in (1,10):
                job=folder/uid/f'eps{int(eps)}_iter{it}'; job.mkdir(parents=True)
                raw=job/'optimized.wav'; protected=job/'protected.wav'
                settings=SimpleNamespace(epsilon=eps,step=None,iters=it,seed=52,mode='SPEC',beta=10.,alpha=.05,log_every=0,random_init=False)
                stats=generate(clean,raw,{'model_speaker':'KR'},settings)
                codes,adjusted=project_pcm16(x[:,0],sf.read(raw)[0],eps)
                write_audio(protected,codes,sr)
                q=metrics(protected,clean)
                if q['linf']>eps/255 or q['rail_samples']: raise ValueError('Candidate numerical contract failed')
                save(job/'generation.json',dict(settings=vars(settings),assets=hashes,stats=stats,adjusted_samples=adjusted,quality=q,
                    defense_status='NOT_EVALUATED; fresh optimization, no clone training or SIM/CER'))
                results.append(dict(utterance_id=uid,epsilon=eps,iterations=it,**q))
                print('candidate',uid,eps,it,q['snr_db'],q['stoi'],flush=True)
    table(OUT/'candidates.csv',results)
    summary=[]
    for eps in (2.,4.,8.):
        for it in (1,10):
            rr=[r for r in results if r['epsilon']==eps and r['iterations']==it]
            summary.append(dict(epsilon=eps,iterations=it,n=len(rr),mean_snr_db=float(np.mean([r['snr_db'] for r in rr])),
                mean_stoi=float(np.mean([r['stoi'] for r in rr])),min_stoi=min(r['stoi'] for r in rr),mean_delta_rms=float(np.mean([r['delta_rms'] for r in rr])),
                clipping_samples=sum(r['rail_samples'] for r in rr)))
    save(OUT/'candidate_summary.json',summary)
    return results,summary

def listening(references, selected, results):
    target=OUT/'listening'; target.mkdir()
    manifest=[]
    for uid in selected:
        dst=target/uid; dst.mkdir()
        clean=references[uid]; x,sr=load(clean)
        shutil.copyfile(clean,dst/'00_original.wav')
        paths=[('00_original',dst/'00_original.wav')]
        historical=OLD/'runs/final_validation_20261119/strict_corrected_inputs'/f'{uid}.wav'
        shutil.copyfile(historical,dst/'01_existing_corrected_SPEC10.wav')
        paths.append(('01_existing_corrected_SPEC10',dst/'01_existing_corrected_SPEC10.wav'))
        for r in [r for r in results if r['utterance_id']==uid]:
            name=f"eps{int(r['epsilon'])}_iter{r['iterations']}"
            p=dst/f'{name}.wav'; shutil.copyfile(r['path'],p); paths.append((name,p))
        for name,p in paths[1:]:
            y,_=load(p)
            # Exact same amplitude; no loudness matching, filtering, fading or resampling.
            ab=dst/f'AB_original_then_{name}.wav'
            write_audio(ab,np.concatenate([x,np.zeros((round(.6*sr),1)),y]),sr)
            residual=dst/f'residual_{name}_FLOAT.wav'
            write_audio(residual,y-x,sr,'FLOAT')
            manifest.append(dict(utterance_id=uid,condition=name,original=str(dst/'00_original.wav'),candidate=str(p),ab=str(ab),residual=str(residual),gain=1,order='original, 0.6s silence, candidate',human_rating='PENDING'))
    save(target/'manifest.json',manifest)
    table(target/'human_ratings.csv',[dict(utterance_id=r['utterance_id'],condition=r['condition'],noise_1_to_5='',naturalness_1_to_5='',preference='',notes='') for r in manifest])
    (target/'README.txt').write_text('동일 재생 음량으로 비교하세요. AB 파일: 원본 → 0.6초 무음 → 보호본. 볼륨 보정·필터·리샘플 없음. residual은 원래 크기의 차이 신호이며 보호 음성 자체가 아닙니다. 사람 청취 평가는 아직 수행하지 않았습니다. human_ratings.csv에 잡음(1 적음~5 큼), 자연스러움(1 낮음~5 높음), 선호도를 기록하세요.\n',encoding='utf-8')

def main():
    references,selected,trio,historical=audit()
    results,summary=candidates(references,selected)
    listening(references,selected,results)
    changed=[p for p,h in watched.items() if digest(p)!=h]
    save(OUT/'preservation.json',dict(watched_sha256=watched,changed=changed,scope='All read source WAVs, manifests, model/config hashes; untouched files are not claimed globally rehashed. Python -B prevents old-source pycache writes.'))
    if changed: raise RuntimeError('Source changed')
    save(OUT/'outcome.json',dict(status='DIAGNOSTIC_COMPLETE_DEFENSE_AND_HUMAN_LISTENING_PENDING',trio=trio,historical=historical,candidates=summary,
         actual_incident_file_identified=False,device_playback_tested=False,new_clone_training=False))
    print('COMPLETE',flush=True)

if __name__=='__main__': main()
