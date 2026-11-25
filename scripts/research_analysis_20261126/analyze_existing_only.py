"""CPU analysis of EXISTING audio/logs only; never writes WAVs or loads models."""
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import soundfile as sf

OUT=Path(__file__).resolve().parent
from research_bootstrap import setup
CODE_ROOT,ROOT,LEGACY=setup()
STUDY=ROOT/'runs/segment_noise_balance_20261124'
from segment_balance import segment,frames,SR
WATCH={}
CONDITIONS=('clean','eps8_iter10_strict','eps2_iter10','silence_only_smooth','perceptual05','voice_adaptive_smooth')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def watch(p,expected=None):
    p=Path(p);h=sha(p)
    if expected is not None:assert h==expected
    if str(p) in WATCH:assert WATCH[str(p)]==h
    WATCH[str(p)]=h;return p


def read(p):return json.loads(watch(p).read_text(encoding='utf-8'))


def csvread(p):
    with watch(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def save(p,o):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)


def table(p,rows):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def main():
    original=csvread(STUDY/'analysis/per_file.csv');by={}
    for r in original:
        if r['condition'] in CONDITIONS:by.setdefault(r['utterance_id'],{})[r['condition']]=r
    assert len(by)==140
    bins=(-np.inf,-50,-40,-30,-20,np.inf)
    accum={(c,i):dict(frames=0,clean_power_sum=0.,delta_power_sum=0.,utterances=set()) for c in CONDITIONS for i in range(5)}
    saturation=[];equality=[]
    for uid,variants in sorted(by.items()):
        source=watch(variants['clean']['path'],variants['clean']['sha256'])
        xcodes,sr=sf.read(source,dtype='int16');assert sr==SR
        x=xcodes.astype(np.float64)/32768
        sample_labels,flabels,_,_=segment(x)
        cp=np.mean(frames(x)**2,axis=1);levels=10*np.log10(np.maximum(cp,1e-30))
        codes={}
        for c in CONDITIONS:
            r=variants[c];p=watch(r['path'],r['sha256']);y,rate=sf.read(p,dtype='int16')
            assert rate==SR and y.shape==xcodes.shape
            codes[c]=y.astype(np.int32)-xcodes.astype(np.int32)
            limit=0 if c=='clean' else 1028 if c=='eps8_iter10_strict' else 257
            assert int(np.max(np.abs(codes[c])))<=limit
            delta=codes[c].astype(np.float64)/32768
            dp=np.mean(frames(delta)**2,axis=1)
            for i in range(5):
                m=(levels>=bins[i])&(levels<bins[i+1]);a=accum[c,i]
                a['frames']+=int(m.sum());a['clean_power_sum']+=float(cp[m].sum());a['delta_power_sum']+=float(dp[m].sum())
                if m.any():a['utterances'].add(uid)
            if c!='clean':
                for i,name in enumerate(('silence','weak','strong')):
                    m=sample_labels==i
                    saturation.append(dict(utterance_id=uid,condition=c,region=name,samples=int(m.sum()),
                        epsilon_boundary_samples=int(np.count_nonzero(np.abs(codes[c][m])==limit)),
                        epsilon_boundary_fraction=float(np.mean(np.abs(codes[c][m])==limit)) if m.any() else None))
        strong=sample_labels==2
        equality.append(dict(utterance_id=uid,strong_samples=int(strong.sum()),
            strong_residual_samples_changed=int(np.count_nonzero(codes['silence_only_smooth'][strong]!=codes['eps2_iter10'][strong]))))
    assert sum(r['strong_residual_samples_changed'] for r in equality)==0
    amplitude=[]
    for (c,i),a in accum.items():
        if not a['frames']:continue
        n=a['frames'];clean=a['clean_power_sum']/n;delta=a['delta_power_sum']/n
        amplitude.append(dict(condition=c,clean_frame_level_bin_dbfs=f'[{bins[i]},{bins[i+1]})',
            frames=n,utterances=len(a['utterances']),clean_rms=math.sqrt(clean),delta_rms=math.sqrt(delta),
            regional_snr_db=10*math.log10(clean/delta) if delta else None,
            weighting='20ms original-energy frames,10ms hop; pooled frame energy, not arithmetic per-utterance RMS'))
    region_summary=[]
    for c in CONDITIONS:
        rr=[r for r in original if r['condition']==c]
        for region in ('silence','weak','voiced'):
            def avg(k):
                values=[float(r[k]) for r in rr if r.get(k)]
                return float(np.mean(values)) if values else None
            region_summary.append(dict(condition=c,region=region,n=140,
                mean_clean_rms=avg(region+'_clean_rms'),mean_noise_rms=avg(region+'_noise_rms'),
                mean_regional_snr_db=avg(region+'_snr_db')))
    sat_summary=[]
    for c in CONDITIONS[1:]:
        for region in ('silence','weak','strong'):
            r=[a for a in saturation if a['condition']==c and a['region']==region]
            sat_summary.append(dict(condition=c,region=region,sample_count=sum(a['samples'] for a in r),
                boundary_sample_fraction=sum(a['epsilon_boundary_samples'] for a in r)/sum(a['samples'] for a in r)))
    timing=[]
    for c,folder in (
        ('eps2_iter10',ROOT/'runs/demo_noise_fix_20261124/defense_attack_140/SPEC_eps2_iter10/protected_train'),
        ('perceptual05',ROOT/'runs/perceptual_spec_20261124/attacks_followup/mask_eps2_iter10_s05/protected_train')):
        values=[];audio=[]
        for uid in sorted(by):
            gen=read(folder/uid/'generation.json');values.append(gen['stats']['seconds']);audio.append(gen['stats']['audio_seconds'])
        timing.append(dict(condition=c,n=140,generation_seconds_sum=float(sum(values)),
            median_seconds_per_file=float(np.median(values)),p90_seconds_per_file=float(np.quantile(values,.9)),
            audio_seconds_sum=float(sum(audio))))
    for p in (ROOT/'scripts/perceptual_spec_20261124/perceptual_objective.py',
        ROOT/'runs/perceptual_spec_20261124/attacks_followup/mask_eps2_iter10_s05/backend/perceptual_override.py',
        LEGACY/'scripts/train_melo_bounded.py'):
        watch(p)
    table(OUT/'amplitude_bins.csv',amplitude);table(OUT/'regional_noise.csv',region_summary)
    table(OUT/'epsilon_saturation.csv',sat_summary);table(OUT/'strong_speech_sample_equality.csv',equality)
    save(OUT/'timing_basis.json',timing)
    changed=[p for p,h in WATCH.items() if sha(p)!=h];assert not changed
    save(OUT/'analysis_audit.json',dict(status='READ_ONLY_ANALYSIS_COMPLETE',files_sha256=WATCH,changed=changed,
        waveform_files_created=0,training_started=False,model_inference=False,packages_installed=False,
        strong_speech_unchanged_in_all_140=True,segmentation='Original-only energy proxy; not human validated VAD',
        amplitude_bin_weighting='pooled overlapping frame powers; separate from per-utterance averages'))
    print(json.dumps(dict(regional_noise=region_summary,epsilon_saturation=sat_summary,timing=timing,
        strong_speech_changed_samples=0,old_files_changed=changed),ensure_ascii=False,indent=2))


if __name__=='__main__':
    from runner import cli
    cli('analyze_existing_only')
