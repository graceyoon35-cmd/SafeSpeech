"""Matched original WAV grid, objective variants, spectra and exact listening assets."""
import gc
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import numpy as np
import soundfile as sf
import torch
from experiment_io import ROOT,OUT,LEGACY,ASSETS,PROFILES,save,sha,table
from audio_analysis import analyze
from perceptual_objective import install
from run_korean_melo_spec import check_assets,load_backend
from protect_wav import project_pcm16

def write_audio(path,a,sr,subtype='PCM_16'):
    with path.open('xb') as f:sf.write(f,a,sr,format='WAV',subtype=subtype)

def main():
    folder=OUT/'quality';folder.mkdir(exist_ok=False)
    demo=ROOT/'runs/demo_noise_fix_20261124/final_demo_eps2/original.wav'
    dataset=json.loads((LEGACY/'runs/paired_melo_experiment_ong85l9i/clean/dataset.json').read_text())['utterances']
    # Fixed diagnostic selection established before inspecting new objective results.
    ids=['kss_hf_00078','kss_hf_04258','kss_hf_03746']
    sources={r['utterance_id']:Path(r['wav']) for r in dataset if r['utterance_id'] in ids}
    assert len(sources)==3 and sha(sources['kss_hf_00078'])==sha(demo)
    assets,config,hashes=check_assets(ASSETS)
    watch={str(p):sha(p) for p in sources.values()}
    watch.update({str(assets/p):sha(assets/p) for p in hashes})
    backend=folder/'backend';backend.mkdir()
    generate=load_backend(assets,config,backend,'cuda')
    install(sys.modules['protect_melo'],backend)
    save(folder/'plan.json',dict(ids=ids,profiles=PROFILES,seed=52,step='epsilon/255/10',
        surrogate='Relative local spectral power budget, low/high frequency weighting. Not a calibrated auditory masking model.',
        human_listening='PENDING',defense='PENDING',source_sha256=watch,
        literature='https://proceedings.mlr.press/v97/qin19a.html',literature_scope='ASR adversarial audio; not validation of anti-cloning efficacy.'))
    rows=[]
    for uid in ids:
        source=sources[uid];target=folder/uid;target.mkdir()
        original=target/'original.wav';shutil.copyfile(source,original)
        for label,profile in PROFILES.items():
            job=target/label;job.mkdir()
            args=SimpleNamespace(epsilon=profile['epsilon'],step=None,iters=profile['iterations'],seed=52,
                mode='SPEC',beta=10.,alpha=.05,log_every=1,random_init=False,perceptual_strength=profile['strength'])
            raw=job/'optimized.wav';output=job/'protected.wav'
            stats=generate(source,raw,{'model_speaker':'KR'},args)
            x,sr=sf.read(source);y,_=sf.read(raw)
            codes,adjusted=project_pcm16(x,y,args.epsilon);write_audio(output,codes,sr)
            q=analyze(source,output)
            assert q['linf']<=args.epsilon/255 and q['clipping']==0
            row=dict(utterance_id=uid,profile=label,epsilon=profile['epsilon'],iterations=profile['iterations'],strength=profile['strength'],**q)
            rows.append(row)
            save(job/'generation.json',dict(settings=vars(args),stats=stats,quality=q,pcm16_adjusted_samples=adjusted))
            xcodes,_=sf.read(source,dtype='int16');ycodes,_=sf.read(output,dtype='int16')
            write_audio(job/'AB_original_then_protected.wav',np.concatenate([xcodes,np.zeros(26460,dtype=np.int16),ycodes]),sr)
            write_audio(job/'residual_original_amplitude_FLOAT.wav',ycodes.astype(np.float64)/32768-xcodes.astype(np.float64)/32768,sr,'FLOAT')
            print(json.dumps(dict(id=uid,profile=label,snr=q['snr_db'],stoi=q['stoi'],quiet_dbfs=q['quiet20_delta_dbfs'])),flush=True)
        if uid=='kss_hf_00078':
            old=ROOT/'runs/demo_noise_fix_20261124/final_demo_eps2/protected_eps2_iter10.wav'
            rows.append(dict(utterance_id=uid,profile='archived_eps2_iter10',epsilon=2.,iterations=10,strength=0.,**analyze(source,old)))
    del generate;gc.collect();torch.cuda.empty_cache()
    table(folder/'per_file.csv',rows)
    summary=[]
    keys=['snr_db','stoi','estoi','delta_dbfs','quiet20_delta_dbfs','band_0_200_dbfs','band_4000_8000_dbfs','band_8000_16000_dbfs','band_16000_22051_dbfs']
    for label in PROFILES:
        subset=[r for r in rows if r['profile']==label]
        summary.append(dict(profile=label,n=len(subset),**{k:float(np.mean([r[k] for r in subset])) for k in keys}))
    table(folder/'summary.csv',summary);save(folder/'summary.json',summary)
    plot(rows,folder)
    blind_trials(rows,folder)
    changed=[p for p,h in watch.items() if sha(p)!=h]
    save(folder/'preservation.json',dict(source_sha256=watch,changed=changed));assert not changed
    print('QUALITY_COMPLETE',flush=True)

def plot(rows,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    chosen=['eps2_iter10','eps1_iter1','eps1_iter5','mask_eps2_iter10_s05','mask_eps2_iter10_s2']
    fig,axes=plt.subplots(2,1,figsize=(11,8))
    for label in chosen:
        r=next(r for r in rows if r['utterance_id']=='kss_hf_00078' and r['profile']==label)
        y,sr=sf.read(r['path']);x,_=sf.read(folder/'kss_hf_00078/original.wav');delta=y-x
        freq,psd=welch(delta,sr,nperseg=2048)
        axes[0].plot(freq,10*np.log10(np.maximum(psd,1e-20)),label=label)
        n=882;m=len(x)//n;frame=np.mean(delta[:m*n].reshape(m,n)**2,axis=1)
        axes[1].plot(np.arange(m)*.02,10*np.log10(np.maximum(frame,1e-20)),label=label)
    axes[0].set(xlim=(0,22050),xlabel='Frequency (Hz)',ylabel='Residual PSD (dBFS/Hz)')
    axes[1].set(xlabel='Time (s)',ylabel='20ms residual RMS (dBFS)')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(folder/'residual_spectrum_and_time.png',dpi=160);plt.close(fig)

def blind_trials(rows,folder):
    # Hidden A/B identity and X assignment. No normalization, filters or fades.
    target=folder/'blind_listening';target.mkdir()
    x,sr=sf.read(folder/'kss_hf_00078/original.wav',dtype='int16')
    rng=np.random.default_rng(20261124);key=[];answers=[]
    for label in ['eps2_iter10','eps1_iter5','mask_eps2_iter10_s05','mask_eps2_iter10_s2']:
        r=next(r for r in rows if r['utterance_id']=='kss_hf_00078' and r['profile']==label)
        y,_=sf.read(r['path'],dtype='int16')
        sub=target/label;sub.mkdir()
        for i in range(1,21):
            swapped=bool(rng.integers(2));A,B=(y,x) if swapped else (x,y)
            answer='A' if rng.integers(2)==0 else 'B';X=A if answer=='A' else B
            name=f'trial_{i:02d}.wav'
            write_audio(sub/name,np.concatenate([A,np.zeros(26460,dtype=np.int16),B,np.zeros(26460,dtype=np.int16),X]),sr)
            key.append(dict(profile=label,trial=i,answer=answer,A_is_protected=swapped))
            answers.append(dict(profile=label,trial=i,X_matches_A_or_B='',noise_1_to_5='',naturalness_1_to_5='',notes=''))
    save(target/'answer_key_DO_NOT_OPEN_BEFORE_LISTENING.json',key)
    table(target/'responses.csv',answers)
    with (target/'README.txt').open('x',encoding='utf-8') as f:f.write('각 trial: A → 0.6초 무음 → B → 0.6초 무음 → X. X가 A/B 중 어느 것과 같은지 기록하세요. 동일 장치·동일 재생 음량을 유지하세요. 정답은 청취 후에만 열어보세요. 미청취·무응답은 불가청 판정이 아닙니다. 20회 시험도 모든 사람·장치의 불가청을 보장하지 않습니다.\n')

if __name__=='__main__':main()
