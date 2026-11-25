"""CPU-only shaping of existing SPEC residuals; NEVER starts PGD or training.
Energy labels are a proxy, not validated semantic VAD/auditory masking.
"""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import soundfile as sf
from scipy.ndimage import gaussian_filter1d
from pystoi import stoi

from workspace_paths import ROOT, LEGACY as OLD, STUDY as OUT
HOP=441;FRAME=882;SR=44100
BANDS=((0,200),(200,1000),(1000,4000),(4000,8000),(8000,16000),(16000,22051))
VARIANTS=('silence_only_smooth','silence_floor10_smooth','voice_adaptive_smooth')
WATCH={}

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def watch(p):
    p=Path(p);WATCH.setdefault(str(p),sha(p));return p

def read(p):return json.loads(watch(p).read_text(encoding='utf-8'))

def save(p,o):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)

def table(p,rows):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def db(power):return float(10*np.log10(max(float(power),1e-30)))

def frames(x):
    padded=np.pad(x,(FRAME//2,FRAME//2),mode='edge')
    return np.lib.stride_tricks.sliding_window_view(padded,FRAME)[::HOP][:math.ceil(len(x)/HOP)]

def segment(x):
    power=np.mean(frames(x)**2,axis=1);levels=10*np.log10(np.maximum(power,1e-30))
    reference=float(np.quantile(levels,.9))
    low=max(-60.,min(-42.,reference-30.));high=low+6;weak_limit=reference-15
    labels=np.where(levels<=low,0,np.where(levels<weak_limit,1,2))
    state=False;active=[]
    for value in levels:
        if value>=high:state=True
        elif value<=low:state=False
        active.append(state)
    expanded=np.zeros(len(active),dtype=float)
    for i,on in enumerate(active):
        if on:expanded[max(0,i-3):min(len(active),i+9)]=1
    positions=np.arange(len(x))/HOP
    hard=np.interp(positions,np.arange(len(expanded)),expanded)
    smooth=gaussian_filter1d(hard,sigma=.008*SR,mode='nearest',truncate=4)
    amplitude=np.clip((np.sqrt(power)/max(10**(reference/20),1e-12))**.5,.35,1.)
    scaled=hard*np.interp(positions,np.arange(len(amplitude)),amplitude)
    adaptive=.05+.95*gaussian_filter1d(scaled,sigma=.008*SR,mode='nearest',truncate=4)
    gains=dict(silence_only_smooth=np.clip(smooth,0,1),silence_floor10_smooth=.1+.9*np.clip(smooth,0,1),
        voice_adaptive_smooth=np.clip(adaptive,.05,1))
    sample_labels=labels[np.minimum(np.rint(positions).astype(int),len(labels)-1)]
    meta=dict(method='Original-only 20ms RMS / 10ms hop; relative energy classes, not semantic VAD.',
        reference_p90_dbfs=reference,silence_threshold_dbfs=low,gate_open_dbfs=high,weak_speech_upper_dbfs=weak_limit,
        onset_padding_ms=30,hangover_ms=80,gaussian_sigma_ms=8,gaussian_support_ms=64,
        frame_counts={n:int(np.sum(labels==i)) for i,n in enumerate(('silence_proxy','weak_speech_proxy','strong_speech_proxy'))})
    return sample_labels,labels,gains,meta

def regions(labels):
    starts=np.r_[0,np.flatnonzero(np.diff(labels))+1];ends=np.r_[starts[1:],len(labels)]
    names=('silence_proxy','weak_speech_proxy','strong_speech_proxy')
    return [dict(start_seconds=float(a*.01),end_seconds=float(b*.01),region=names[int(labels[a])]) for a,b in zip(starts,ends)]

def metrics(x,y,labels,flabels,baseline_delta):
    delta=y-x;dp=np.mean(delta**2);sp=np.mean(x*x)
    r=dict(sample_rate=SR,subtype='PCM_16',channels=1,samples=len(x),seconds=len(x)/SR,
        snr_db=db(sp/dp) if dp else None,stoi=float(stoi(x,y,SR,extended=False)),
        clean_rms=float(np.sqrt(sp)),protected_rms=float(np.sqrt(np.mean(y*y))),
        noise_rms=float(np.sqrt(dp)),noise_dbfs=db(dp),linf=float(np.max(np.abs(delta))),
        clipping=int(np.count_nonzero((y<=-1)|(y>=32767/32768))))
    freq=np.fft.rfftfreq(FRAME,1/SR);window=np.hanning(FRAME)
    spectral=np.abs(np.fft.rfft(frames(delta)*window,axis=1))**2/(FRAME*np.sum(window**2))
    factors=np.full(len(freq),2.);factors[[0,-1]]=1;spectral*=factors
    for i,name in enumerate(('silence','weak','voiced')):
        mask=labels==i;fm=flabels==i;r[name+'_samples']=int(mask.sum())
        if not mask.any():
            r[name+'_noise_rms']=None;r[name+'_snr_db']=None;continue
        power=float(np.mean(delta[mask]**2));signal=float(np.mean(x[mask]**2))
        original_power=float(np.sum(baseline_delta[mask]**2))
        r.update({name+'_clean_rms':float(np.sqrt(signal)),name+'_noise_rms':float(np.sqrt(power)),
            name+'_noise_dbfs':db(power),name+'_snr_db':db(signal/power) if signal and power else None,
            name+'_residual_energy_retained':float(np.sum(delta[mask]**2)/original_power) if original_power else None})
        # Average individual Hann-window periodograms, never concatenate disjoint regions.
        for lo,hi in BANDS:
            p=float(spectral[fm][:,(freq>=lo)&(freq<hi)].sum(axis=1).mean()) if fm.any() else None
            r[f'{name}_band_{lo}_{hi}_power']=p
    N=len(delta);p=np.abs(np.fft.rfft(delta))**2/N**2;f=np.fft.rfftfreq(N,1/SR)
    mult=np.full(len(p),2.);mult[0]=1
    if N%2==0:mult[-1]=1
    p*=mult;assert abs(float(p.sum())-dp)<1e-12
    for lo,hi in BANDS:
        v=float(p[(f>=lo)&(f<hi)].sum());r[f'band_{lo}_{hi}_power']=v;r[f'band_{lo}_{hi}_dbfs']=db(v)
    return r

def write_audio(path,a):
    with Path(path).open('xb') as f:sf.write(f,a,SR,format='WAV',subtype='PCM_16')

def main():
    dataset=read(OLD/'runs/paired_melo_experiment_ong85l9i/clean/dataset.json')
    train=[r for r in dataset['utterances'] if r['split']=='train'];assert len(train)==140
    baseq=read(ROOT/'runs/demo_noise_fix_20261124/defense_attack_140/SPEC_eps2_iter10/quality.json')
    bm={r['utterance_id']:r for r in baseq['per_file']}
    with watch(OLD/'runs/final_validation_20261119/strict_corrected_inputs/quality.csv').open(encoding='utf-8-sig') as f:
        e8={r['utterance_id']:r for r in csv.DictReader(f)}
    (OUT/'candidates').mkdir(exist_ok=False);(OUT/'analysis').mkdir();(OUT/'listening').mkdir()
    pilot={'kss_hf_00078','kss_hf_00137','kss_hf_00347'};all_rows=[];all_regions=[];new_paths={v:{} for v in VARIANTS};pilot_data={}
    save(OUT/'method.json',dict(basis='CPU postprocessing of existing eps2/255 10-iteration residual only; no SPEC optimization or new model training.',
        variants={v:('smoothed conservative activity envelope' if v=='silence_only_smooth' else
            '0.1+0.9*activity envelope' if v=='silence_floor10_smooth' else
            '0.05+0.95*activity*clip(sqrt(frame_RMS/reference_RMS),0.35,1)') for v in VARIANTS},
        human_listening='Exploratory feedback only, no formal audibility approval.',defense='UNMEASURED for all new candidates; retained voiced residual is not defense evidence.'))
    for index,row in enumerate(train):
        uid=row['utterance_id'];clean=watch(row['wav']);assert sha(clean)==row['wav_sha256']
        x,sr=sf.read(clean);xcodes,_=sf.read(clean,dtype='int16');assert sr==SR and x.ndim==1
        base=watch(bm[uid]['path']);assert sha(base)==bm[uid]['sha256'];bcodes,br=sf.read(base,dtype='int16')
        assert br==SR and bcodes.shape==xcodes.shape
        y=bcodes.astype(np.float64)/32768;delta=y-x;dcode=bcodes.astype(np.int32)-xcodes.astype(np.int32)
        labels,flabels,gains,meta=segment(x)
        all_regions.extend(dict(utterance_id=uid,**r) for r in regions(flabels))
        save(OUT/'analysis'/(uid+'_segments.json'),dict(original=str(clean),sha256=sha(clean),**meta,intervals=regions(flabels)))
        sources={'clean':clean,'eps2_iter10':base,'eps8_iter10_strict':Path(e8[uid]['output']),
            'eps1_iter5':ROOT/'runs/perceptual_spec_20261124/attacks/eps1_iter5/protected_train'/uid/'protected.wav',
            'perceptual05':ROOT/'runs/perceptual_spec_20261124/attacks_followup/mask_eps2_iter10_s05/protected_train'/uid/'protected.wav'}
        assert sha(watch(sources['eps8_iter10_strict']))==e8[uid]['sha256']
        for condition,source in sources.items():
            watch(source);a,rate=sf.read(source);assert a.shape==x.shape and rate==SR
            eps=0 if condition=='clean' else 8 if condition=='eps8_iter10_strict' else 1 if condition=='eps1_iter5' else 2
            q=metrics(x,a,labels,flabels,delta);assert q['linf']<=eps/255 and q['clipping']==0
            all_rows.append(dict(utterance_id=uid,condition=condition,path=str(source),sha256=sha(source),epsilon=eps/255,strict_epsilon_pass=True,**q))
        for variant,gain in gains.items():
            folder=OUT/'candidates'/variant;folder.mkdir(exist_ok=True)
            newdelta=np.rint(gain*dcode).astype(np.int32)
            assert np.all(np.abs(newdelta)<=np.abs(dcode))
            codes=np.clip(xcodes.astype(np.int32)+newdelta,-32768,32767).astype(np.int16)
            path=folder/(uid+'.wav');write_audio(path,codes)
            actual,rate=sf.read(path);assert rate==SR and actual.shape==x.shape
            q=metrics(x,actual,labels,flabels,delta)
            assert q['clipping']==0 and q['linf']<=2/255
            modulation=float(np.max(np.abs(np.diff(gain)*delta[:-1])))
            record=dict(utterance_id=uid,condition=variant,path=str(path),sha256=sha(path),epsilon=2/255,strict_epsilon_pass=True,
                gain_min=float(gain.min()),gain_max=float(gain.max()),max_gain_step=float(np.max(np.abs(np.diff(gain)))),
                max_gate_modulation_step=modulation,**q)
            all_rows.append(record);new_paths[variant][uid]=path
            if uid in pilot:
                original_codes=xcodes;silence=np.zeros(26460,dtype=np.int16)
                output=OUT/'listening'/(uid+'__original_baseline_'+variant+'.wav')
                expected=np.concatenate([original_codes,silence,bcodes,silence,codes]);write_audio(output,expected)
                decoded,_=sf.read(output,dtype='int16');assert np.array_equal(decoded,expected)
                decode=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(output),'-f','null','-'],capture_output=True)
                assert decode.returncode==0
        if uid in pilot:pilot_data[uid]=dict(x=x,y=y,labels=labels,gains=gains,sr=sr)
        if (index+1)%20==0:print('CPU analysis/candidate generation',index+1,140,flush=True)
    table(OUT/'analysis/per_file.csv',all_rows);table(OUT/'analysis/segments.csv',all_regions)
    summary=[]
    for condition in sorted({r['condition'] for r in all_rows}):
        items=[r for r in all_rows if r['condition']==condition]
        keys=('snr_db','stoi','noise_rms','silence_noise_rms','weak_noise_rms','voiced_residual_energy_retained',
              'band_8000_16000_power','band_16000_22051_power','max_gate_modulation_step')
        summary.append(dict(condition=condition,n=len(items),clipping=sum(r['clipping'] for r in items),
            epsilon_failures=sum(not r['strict_epsilon_pass'] for r in items),
            **{k:float(np.mean([r[k] for r in items if r.get(k) is not None])) if any(r.get(k) is not None for r in items) else None for k in keys}))
    save(OUT/'analysis/summary.json',summary);table(OUT/'analysis/summary.csv',summary)
    plot(pilot_data)
    make_plans(dataset,new_paths,summary)
    changed=[p for p,h in WATCH.items() if sha(p)!=h];assert not changed
    save(OUT/'preservation.json',dict(original_sha256=WATCH,changed=changed))
    save(OUT/'outcome.json',dict(status='CPU_QUALITY_VALIDATED_DEFENSE_AND_HUMAN_LISTENING_PENDING',
        source_utterances=140,new_candidates=len(VARIANTS)*140,listening_files=len(pilot)*len(VARIANTS),
        no_new_1000_step_training=True,no_spec_optimization=True,no_gpu_model_inference=True,
        full_audibility_approval=False,defense_approved=False,git_commit_or_push=False,summary=summary))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

def plot(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for uid,d in data.items():
        fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True);t=np.arange(len(d['x']))/SR
        axes[0].plot(t,d['x'],linewidth=.5);axes[0].set_ylabel('Original amplitude')
        axes[1].plot(t,d['y']-d['x'],linewidth=.4);axes[1].set_ylabel('Original SPEC residual')
        for label,gain in d['gains'].items():axes[2].plot(t,gain,label=label)
        axes[2].set(xlabel='Time (s)',ylabel='Residual-only gain',ylim=(-.03,1.03));axes[2].legend(fontsize=8)
        for ax in axes:
            for region in regions(d['labels'][::HOP]):
                if region['region']=='silence_proxy':ax.axvspan(region['start_seconds'],region['end_seconds'],color='gray',alpha=.12)
        fig.tight_layout();fig.savefig(OUT/'analysis'/(uid+'_envelope.png'),dpi=140);plt.close(fig)

def make_plans(dataset,new_paths,summary):
    selected=('silence_only_smooth','voice_adaptive_smooth')
    base=read(OLD/'runs/paired_melo_experiment_ong85l9i/clean/config.json')
    (OUT/'plans').mkdir()
    for label in selected:
        quality=next(r for r in summary if r['condition']==label)
        assert quality['clipping']==quality['epsilon_failures']==0
        folder=OUT/'plans'/label;folder.mkdir()
        updated=copy.deepcopy(dataset)
        for r in updated['utterances']:
            if r['split']=='train':r.update(wav=str(new_paths[label][r['utterance_id']]),wav_sha256=sha(new_paths[label][r['utterance_id']]))
        manifest=folder/'dataset.json';save(manifest,updated)
        cfg=copy.deepcopy(base);cfg['experiment'].update(condition=label,dataset_manifest=str(manifest),defense_measured=False)
        metadata=folder/'metadata.list.cleaned'
        with metadata.open('x',encoding='utf-8') as f:
            for line in watch(base['data']['training_files']).read_text().splitlines():
                fields=line.split('|');fields[0]=str(new_paths[label][Path(fields[0]).stem]);f.write('|'.join(fields)+'\n')
        cfg['data']['training_files']=str(metadata);save(folder/'config.json',cfg)
    save(OUT/'future_attack_plan.json',dict(status='PLAN_ONLY_DO_NOT_EXECUTE_IN_THIS_REQUEST',
        selected_candidates=selected,selection='Strict epsilon/clipping pass; compare conservative silence-only shaping against amplitude-adaptive shaping. Selection is not a defense claim.',
        training=dict(train_utterances=140,validation_utterances=30,test_prompts=30,steps=1000,seed=52,batch_size=1,
            trainer=str(OLD/'scripts/train_melo_bounded.py'),trainer_sha256=sha(watch(OLD/'scripts/train_melo_bounded.py')),
            generator_initialization='same pinned official checkpoint, not an archived defended attack model',
            checkpoint_every=1000,config_paths=[str(OUT/'plans'/c/'config.json') for c in selected]),
        evaluation=dict(inference_seed='2026+i',speaker_metric='same ECAPA cosine against fixed five clean references',
            asr='same Whisper small ko temperature0 and original normalization',archived_references=['Clean','SPEC eps2/255 10 iterations']),
        runtime_estimate=dict(gpu='Existing RTX 5060 Ti 8GB',training_minutes_per_candidate=[9,25],
            evaluation_minutes=[2,6],total_two_candidate_minutes=[22,62],recommended_reserved_minutes_including_overhead=[25,70],
            evaluation_minutes_scope='per candidate; total = two training runs plus two evaluations',
            basis='Historical 534s/508s/523s runs and older 1213s/1488s runs; not a measurement of these new candidates'),
        disk_estimate=dict(generator_bytes_per_candidate=207822652,state_bytes_per_candidate=984079669,
            recommended_free_bytes_two_candidates=8*1024**3,backup_extra_not_included=True),
        validation_limits='Energy proxy is not manual VAD. Existing SIM/CER cannot prove defense of these modified WAVs. No candidate is approved as inaudible or defense-preserving.',
        hypotheses=['Silence-only suppression preserves voiced perturbation more closely than global loss penalties; defense may nevertheless rely on silent context.',
            'Amplitude-adaptive suppression should reduce weak-speech residual but can weaken defense further.',
            'Future masked SPEC optimization should keep original loss as primary and audit relative loss/gradient changes, rather than accept lower noise alone. A surrogate-loss guard would still not establish cloning defense.']))

if __name__=='__main__':main()
