"""Five fixed KSS train utterances; masking-SPEC generation and quality only."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from research_paths import configure_paths
import random
import shutil
import subprocess
import sys
import time
import traceback
import wave
import zipfile
from types import SimpleNamespace
import numpy as np
import soundfile as sf
import torch
from auditory_masking import build_model,audit
from masked_spec import MaskedSpecPerturb

IDS=('kss_hf_06460','kss_hf_07448','kss_hf_00347','kss_hf_11187','kss_hf_01638')
WATCH={}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def watch(p,expected=None):
    p=Path(p);actual=sha(p)
    if expected is not None:assert actual==expected,(str(p),'source hash mismatch')
    if str(p) in WATCH:assert WATCH[str(p)]==actual
    WATCH[str(p)]=actual;return p


def read(p):return json.loads(watch(p).read_text(encoding='utf-8'))


def save(p,o):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False)


def table(p,rows):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def loadcsv(p):
    with watch(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def git(root,*args):
    win=subprocess.check_output(['wslpath','-w',str(root)],text=True).strip()
    return subprocess.check_output(['/mnt/c/Program Files/Git/cmd/git.exe','-C',win,*args],text=True).strip()


def main(args):
    root=args.workspace.resolve();out=args.output.resolve();out.mkdir(exist_ok=False)
    started=time.perf_counter()
    study=root/'runs/segment_noise_balance_20261124';legacy=configure_paths(root)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    from segment_balance import segment,metrics
    from run_korean_melo_spec import check_assets,load_backend
    def progress(**kw):
        line=json.dumps(dict(time=time.time(),**kw),ensure_ascii=False)
        with (out/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(line+'\n')
        print(line,flush=True)
    branch=git(root,'branch','--show-current');head=git(root,'rev-parse','HEAD')
    refs_before=git(root,'for-each-ref','--format=%(refname) %(objectname)','refs/heads','refs/remotes')
    assert not git(root,'status','--porcelain','--untracked-files=all')
    disk=shutil.disk_usage(root);assert disk.free>2*1024**3
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free,utilization.gpu,temperature.gpu','--format=csv'],text=True)
    assert torch.cuda.is_available();free,total=torch.cuda.mem_get_info();assert free>4*1024**3
    torch.set_num_threads(4)
    dataset=read(legacy/'runs/paired_melo_experiment_ong85l9i/clean/dataset.json')
    original={r['utterance_id']:r for r in dataset['utterances']}
    source_metrics=loadcsv(study/'analysis/per_file.csv');sources={}
    for uid in IDS:
        assert original[uid]['split']=='train'
        sources[uid]={r['condition']:r for r in source_metrics if r['utterance_id']==uid and r['condition'] in ('clean','eps2_iter10','silence_only_smooth')}
        assert len(sources[uid])==3
        for r in sources[uid].values():watch(r['path'],r['sha256'])
        assert sources[uid]['clean']['sha256']==original[uid]['wav_sha256']
    assets=Path('/home/safespeech/models/melo-spec-pinned-20261118')
    assets,cfg,hashes=check_assets(assets)
    for relative,h in hashes.items():watch(assets/relative,h)
    for p in Path(__file__).parent.glob('*.py'):watch(p)
    watch(root/'runs/algorithm_design_20261125/ALGORITHM_PLAN_KO.txt')
    save(out/'plan.json',dict(ids=IDS,split='train',margins_db=[3,6],epsilon=2/255,iters=10,seed=52,beta=10,
        n_fft=[2048,1024],normalization='Clean utterance global spectral max96dB, not calibrated SPL',
        model='All-tonal Qin2019-style maskers,0.5Bark separation,two-slope spreading,ATH; no temporal masking',
        source_model_sha256=hashes,source_branch=branch,source_head=head,free_disk=disk.free,free_cuda=free,nvidia_smi=gpu,
        no_attack_training=True,no_sim_cer_evaluation=True,no_package_install=True,
        literal_spec_function='Mel reconstruction +10*(noise reconstruction+mel KL); input-only gradients',
        limitations='Five train recordings, no human validation; model threshold not an absolute hearing guarantee'))
    backend=out/'backend';backend.mkdir()
    generate=load_backend(assets,cfg,backend,'cuda')
    import protect_melo as P
    records=[];mask_rows=[];objective_rows=[];threshold_records=[];all_sources={}
    (out/'thresholds').mkdir();(out/'candidates').mkdir()
    for uid in IDS:
        progress(stage='utterance_start',id=uid)
        refs={c:r['path'] for c,r in sources[uid].items()}
        x,sr=sf.read(refs['clean']);xcodes,_=sf.read(refs['clean'],dtype='int16');assert sr==44100
        labels,flabels,gains,segmentation=segment(x)
        models=build_model(x)
        radius=np.floor(257*gains['silence_only_smooth']+1e-10).astype(np.int32)
        tpath=out/'thresholds'/(uid+'.npz')
        with tpath.open('xb') as f:
            np.savez_compressed(f,**{f'raw_threshold_{m.n_fft}':m.raw_threshold for m in models},
                **{f'masker_count_{m.n_fft}':m.masker_counts for m in models},radius_codes=radius)
        threshold_records.append(dict(id=uid,path=str(tpath),sha256=sha(tpath),segmentation=segmentation,
            scales=[dict(n_fft=m.n_fft,clean_peak_power=m.clean_peak_power,frames=m.raw_threshold.shape[1],
                mean_maskers=float(np.mean(m.masker_counts))) for m in models]))
        wave_paths=dict(refs);objectives=None
        for margin in (3,6):
            label=f'mask{margin}db_iter10';job=out/'candidates'/label/uid;job.mkdir(parents=True)
            optimizer=MaskedSpecPerturb(models,radius,refs,margin,lambda **kw:progress(stage='update',id=uid,candidate=label,**kw))
            P.perturb=optimizer
            settings=SimpleNamespace(epsilon=2.,iters=10,step=None,seed=52,mode='SPEC',beta=10.,alpha=.05,log_every=1,random_init=False)
            target=job/'protected.wav'
            stats=generate(Path(refs['clean']),target,{'model_speaker':'KR'},settings)
            actual,saved_sr=sf.read(target,dtype='int16');assert saved_sr==sr and actual.shape==xcodes.shape
            final=optimizer.measure_file(target)
            assert abs(final['loss_SPEC']-stats['final_pcm_objective']['loss_SPEC'])<.01
            stats['reread_saved_pcm_objective']=final
            save(job/'generation.json',dict(settings=vars(settings),margin_db=margin,stats=stats,
                source_original=refs['clean'],source_sha256=sources[uid]['clean']['sha256'],output_sha256=sha(target)))
            wave_paths[label]=str(target);objectives=stats['objective_references']
            objective_rows.append(dict(id=uid,condition=label,**final,
                reduction_vs_clean=objectives['clean']['loss_SPEC']-final['loss_SPEC'],
                retained_vs_eps2=stats['reference_improvement_retained']['eps2_iter10'],
                retained_vs_silence=stats['reference_improvement_retained']['silence_only_smooth'],
                guard95_eps2=stats['guard_95pct_baseline'],guard95_silence=stats['guard_95pct_silence'],
                accepted_steps=stats['accepted_steps'],clean_fallback=stats['clean_fallback'],
                residual_gain_fallback=(stats['selected_repair'] or {}).get('residual_gain_fallback',0)))
            progress(stage='candidate_complete',id=uid,candidate=label,loss_SPEC=final['loss_SPEC'],guard95=stats['guard_95pct_silence'])
        for c,r in objectives.items():objective_rows.append(dict(id=uid,condition=c,**r,reduction_vs_clean=objectives['clean']['loss_SPEC']-r['loss_SPEC']))
        baseline=sf.read(refs['eps2_iter10'])[0]-x
        for label,path in wave_paths.items():
            y,rate=sf.read(path);assert rate==sr and y.shape==x.shape
            q=metrics(x,y,labels,flabels,baseline);assert q['clipping']==0
            assert q['linf']<=2/255 and (label!='clean' or q['linf']==0)
            decoded=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',path,'-f','null','-'],capture_output=True)
            assert decoded.returncode==0
            records.append(dict(id=uid,condition=label,path=path,sha256=sha(path),strict_epsilon_pass=True,decode_pass=True,**q))
            for margin in (3,6):
                checks=audit(y-x,models,margin)
                for row in checks:mask_rows.append(dict(id=uid,condition=label,**row))
                if label==f'mask{margin}db_iter10':assert all(row['pass_constraint'] for row in checks)
        all_sources[uid]=wave_paths
    table(out/'quality_per_file.csv',records);table(out/'masking_per_file.csv',mask_rows);table(out/'objective_per_file.csv',objective_rows)
    save(out/'threshold_manifest.json',threshold_records)
    summaries=[]
    for label in ('clean','eps2_iter10','silence_only_smooth','mask3db_iter10','mask6db_iter10'):
        a=[r for r in records if r['condition']==label]
        summary=dict(condition=label,n=len(a),clipping=sum(r['clipping'] for r in a),epsilon_failures=0)
        for key in ('snr_db','stoi','noise_rms','silence_noise_rms','weak_noise_rms','voiced_noise_rms'):
            values=[r[key] for r in a if r.get(key) is not None];summary[key]=float(np.mean(values)) if values else None
        obj=[r for r in objective_rows if r['condition']==label]
        summary['mean_loss_SPEC']=float(np.mean([r['loss_SPEC'] for r in obj]))
        if label.startswith('mask'):
            summary['guard95_silence_pass_count']=sum(r['guard95_silence'] for r in obj)
            summary['mean_retained_vs_silence']=float(np.mean([r['retained_vs_silence'] for r in obj]))
            summary['mean_retained_vs_eps2']=float(np.mean([r['retained_vs_eps2'] for r in obj]))
            summary['clean_fallback_count']=sum(r['clean_fallback'] for r in obj)
            margin=int(label[4]);ms=[r for r in mask_rows if r['condition']==label and r['margin_db']==margin]
            summary['masking_violation_bins']=sum(r['violating_bins'] for r in ms)
            summary['masking_max_excess_db']=max(r['max_excess_db'] for r in ms)
        summaries.append(summary)
    save(out/'summary.json',summaries);table(out/'summary.csv',summaries)
    make_blind(out,all_sources)
    changed=[p for p,h in WATCH.items() if sha(p)!=h];assert not changed
    assert git(root,'branch','--show-current')==branch and git(root,'rev-parse','HEAD')==head
    assert git(root,'for-each-ref','--format=%(refname) %(objectname)','refs/heads','refs/remotes')==refs_before
    assert not git(root,'status','--porcelain','--untracked-files=all')
    save(out/'preservation.json',dict(source_sha256=WATCH,changed=changed))
    save(out/'complete.json',dict(status='FIVE_UTTERANCE_PILOT_COMPLETE',ids=IDS,new_protected_wavs=10,blind_wavs=25,
        no_1000_step_training=True,no_sim_cer_evaluation=True,defense_confirmed=False,human_listening_pending=True,
        masking_constraints='2048+1024 independent FFT final-PCM check',seconds=time.perf_counter()-started,
        no_commit_push=True,original_branch=branch,original_head=head))
    print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)


def make_blind(out,paths):
    public=out/'blind_listening';private=out/'organizer_private';public.mkdir();private.mkdir()
    rng=random.Random(2026112503);ids=list(IDS);rng.shuffle(ids);mapping=[];filenames=[]
    for group,uid in enumerate(ids,1):
        conditions=list(paths[uid]);rng.shuffle(conditions)
        for position,c in enumerate(conditions,1):
            source=Path(paths[uid][c]);target=public/(f'clip_{rng.getrandbits(32):08X}.wav')
            with wave.open(str(source),'rb') as f:params=f.getparams();data=f.readframes(f.getnframes())
            with target.open('xb') as stream:
                with wave.open(stream,'wb') as f:
                    f.setnchannels(params.nchannels);f.setsampwidth(params.sampwidth);f.setframerate(params.framerate);f.writeframes(data)
            with wave.open(str(target),'rb') as f:assert f.readframes(f.getnframes())==data
            mapping.append(dict(group=group,position=position,filename=target.name,id=uid,condition=c,
                source=str(source),source_sha256=sha(source),copied_sha256=sha(target),pcm_sha256=hashlib.sha256(data).hexdigest()))
            filenames.append(target.name)
    table(private/'answer_mapping.csv',mapping);save(private/'answer_mapping.json',mapping)
    guide='한국어 음성 품질 비교\n\n같은 발화5개씩 한 묶음입니다. 이어폰/헤드폰과 기기 음량을 유지하고 듣습니다.\n파일별 잡음 거슬림/자연스러움/내용 이해를 각각1~5점으로 기록하고, 발화 중·쉼·경계의 소리를 구분해 적어 주세요.\n순서는 아래와 같으며 조건을 추측하거나 다른 평가자와 상의하지 마세요. 자유롭게 재청취할 수 있습니다.\n\n'
    for r in mapping:guide+=f"묶음{r['group']} / {r['position']}: {r['filename']}\n"
    with (public/'README_KO.txt').open('x',encoding='utf-8') as f:f.write(guide)
    zp=out/'SafeSpeech_masking_pilot_blind_20261125.zip'
    with zipfile.ZipFile(zp,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ['README_KO.txt']+filenames:
            info=zipfile.ZipInfo(name,date_time=(2026,11,25,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,(public/name).read_bytes())
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None and len(z.namelist())==26
        for name in z.namelist():assert hashlib.sha256(z.read(name)).hexdigest()==sha(public/name)
    save(out/'blind_verification.json',dict(wavs=25,zip_members=26,zip_crc_pass=True,zip_sha256=sha(zp),zip=str(zp),mapping=str(private/'answer_mapping.csv'),human_responses=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:main(a)
    except BaseException as e:
        if a.output.exists():save(a.output/('failure_'+str(time.time_ns())+'.json'),dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
        raise
