"""Fresh attacks of two already-generated WAV sets; all old files read-only.

Uses unchanged bounded Melo trainer and unchanged evaluator with output locations
redirected in memory. Never generates SPEC audio or resumes an old attack model.
"""
import copy
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from unittest.mock import patch

import numpy as np
import soundfile as sf
from pystoi import stoi

from workspace_paths import ROOT, LEGACY, STUDY as BASE
TARGET=BASE/'defense_validation_20261124'
ASSETS=Path('/home/safespeech/models/melo-spec-pinned-20261118')
EVALMODELS=Path('/home/safespeech/models/melo-evaluation-20261118')
LABELS=('silence_only_smooth','voice_adaptive_smooth')
WATCH={}
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
sys.path.insert(0,str(LEGACY/'scripts'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def watch(p,expected=None):
    p=Path(p);actual=sha(p)
    if expected is not None:assert actual==expected,(str(p),'hash mismatch')
    if str(p) in WATCH:assert WATCH[str(p)]==actual
    WATCH[str(p)]=actual
    return p


def read(p):return json.loads(watch(p).read_text(encoding='utf-8'))


def save(p,o):
    with Path(p).open('x',encoding='utf-8') as f:
        json.dump(o,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def table(p,rows):
    with Path(p).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def progress(**kw):
    line=json.dumps(dict(time=time.time(),**kw),ensure_ascii=False)
    with (TARGET/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(line+'\n')
    print(line,flush=True)


def main():
    TARGET.mkdir(exist_ok=False)
    from datetime import datetime,timezone
    started=datetime.now(timezone.utc).isoformat()
    disk=shutil.disk_usage(ROOT)
    assert disk.free>=8*1024**3,'Less than planned 8 GiB available'
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu','--format=csv'],text=True)
    processes=subprocess.check_output(['ps','-eo','pid,comm,args'],text=True)
    save(TARGET/'resource_check.json',dict(started_utc=started,disk_total=disk.total,disk_free=disk.free,
        nvidia_smi=gpu,processes=processes,planned_candidates=LABELS,estimated_total_minutes=[25,70],recommended_free_bytes=8*1024**3))
    import torch
    assert torch.cuda.is_available()
    free,total=torch.cuda.mem_get_info()
    assert free>=5*1024**3,'Insufficient free GPU memory for historical trainer footprint'
    progress(stage='resource_check_pass',disk_free_gib=disk.free/1024**3,gpu=torch.cuda.get_device_name(),cuda_free_gib=free/1024**3)
    old=read(ROOT/'runs/demo_noise_fix_20261124/defense_attack_140/training_summary.json')
    previous={r['condition']:r for r in old['training']}
    experiment=Path(old['experiment']);original=read(experiment/'clean/dataset.json')
    basecfg=read(experiment/'clean/config.json')
    cleanreport=read(previous['clean']['report'])
    eps2report=read(previous['SPEC_eps2_iter10']['report'])
    assert cleanreport['completed_steps']==eps2report['completed_steps']==1000
    assert cleanreport['initial_checkpoint_sha256']==eps2report['initial_checkpoint_sha256']
    trainer=watch(LEGACY/'scripts/train_melo_bounded.py',cleanreport['trainer_sha256'])
    assert sha(trainer)==eps2report['trainer_sha256']
    for r in previous.values():
        if r['condition'] in ('clean','SPEC_eps2_iter10'):
            watch(r['checkpoint']);watch(r['report'])
    assert len(original['utterances'])==200
    for row in original['utterances']:
        for k in ('wav','features'):watch(row[k],row[k+'_sha256'])
    initial=watch(basecfg['experiment']['generator_initialization'],cleanreport['initial_checkpoint_sha256'])
    assert sha(initial)=='48e3ff3fd0b5348e095f0468e60ae727507564100f58142ef3a922ead6e0a4d0'
    approved=read(BASE/'plans/verified_inputs.json')
    approved_by={r['candidate']:r for r in approved['candidates']}
    quality=[];case_configs={}
    for label in LABELS:
        cfgpath=watch(BASE/'plans'/label/'config.json',approved_by[label]['config_sha256'])
        cfg=read(cfgpath);ds=read(watch(cfg['experiment']['dataset_manifest'],approved_by[label]['dataset_sha256']))
        assert cfg['train']==basecfg['train'] and cfg['model']==basecfg['model']
        assert cfg['train']['seed']==52 and cfg['train']['batch_size']==1
        assert cfg['experiment']['precision']==basecfg['experiment']['precision']
        assert cfg['experiment']['generator_initialization']==str(initial)
        a=copy.deepcopy(cfg['data']);b=copy.deepcopy(basecfg['data']);a.pop('training_files');b.pop('training_files');assert a==b
        assert len(ds['utterances'])==200
        count=0
        for row,clean in zip(ds['utterances'],original['utterances']):
            assert all(row[k]==clean[k] for k in ('utterance_id','split','text','features','features_sha256','input_kind'))
            watch(row['features'],row['features_sha256'])
            if row['split']!='train':assert row==clean;continue
            p=watch(row['wav'],row['wav_sha256'])
            info=sf.info(p);ci=sf.info(clean['wav'])
            assert info.samplerate==ci.samplerate==44100 and info.channels==ci.channels==1
            assert info.subtype==ci.subtype=='PCM_16' and info.frames==ci.frames
            x,sr=sf.read(clean['wav'],dtype='int16');y,_=sf.read(p,dtype='int16')
            d=y.astype(np.int32)-x.astype(np.int32);linf=int(np.max(np.abs(d)))
            assert linf<=int(np.floor((2/255)*32768))==257
            clip=int(np.count_nonzero((y==-32768)|(y==32767)));assert clip==0
            xf=x.astype(np.float64)/32768;yf=y.astype(np.float64)/32768
            power=float(np.mean((yf-xf)**2))
            quality.append(dict(condition=label,utterance_id=row['utterance_id'],original=clean['wav'],path=str(p),
                original_sha256=clean['wav_sha256'],sha256=row['wav_sha256'],sample_rate=sr,subtype=info.subtype,
                samples=len(x),linf_codes=linf,epsilon=2/255,strict_pcm16_pass=True,clipping=clip,
                snr_db=float(10*np.log10(np.mean(xf*xf)/power)),stoi=float(stoi(xf,yf,sr,extended=False))))
            count+=1
        assert count==140
        assert [r['utterance_id'] for r in ds['utterances'] if r['split']=='train']==[r['utterance_id'] for r in original['utterances'] if r['split']=='train']
        watch(cfg['data']['training_files'])
        case_configs[label]=cfgpath
        progress(stage='candidate_pcm16_preflight_pass',candidate=label,utterances=count,max_delta_codes=max(r['linf_codes'] for r in quality if r['condition']==label))
    provenance=read(EVALMODELS/'model_provenance.json')
    for relative,expected in provenance['sha256'].items():watch(EVALMODELS/relative,expected)
    for name in ('melo_training_core.py','melo_checkpoint.py','evaluate_paired_melo_smoke.py','evaluation_statistics.py','run_korean_melo_spec.py'):
        watch(LEGACY/'scripts'/name)
    watch(BASE/'analysis/per_file.csv');watch(BASE/'analysis/summary.json');watch(BASE/'analysis/preselection.json')
    priorpreservation=read(BASE/'preservation.json')
    for p,h in priorpreservation['original_sha256'].items():watch(p,h)
    table(TARGET/'quality_recheck.csv',quality)
    save(TARGET/'preflight.json',dict(status='PASS',candidates=LABELS,each_train=140,test_prompts=30,steps=1000,seed=52,
        initial_sha256=sha(initial),trainer_sha256=sha(trainer),free_cuda_bytes=free,free_disk_bytes=disk.free,
        all_strict_pcm16_pass=True,all_same_original=True,clipping=0,
        quality_summary=[dict(condition=label,mean_snr_db=float(np.mean([r['snr_db'] for r in quality if r['condition']==label])),
            mean_stoi=float(np.mean([r['stoi'] for r in quality if r['condition']==label]))) for label in LABELS]))
    save(TARGET/'source_hashes_before.json',WATCH)
    # Release preflight CUDA context cache; training runs in a separate process.
    torch.cuda.empty_cache()
    records=[]
    for label in LABELS:
        folder=TARGET/label;folder.mkdir()
        command=[sys.executable,'-B','-u',str(trainer),'--config',str(case_configs[label]),'--assets',str(ASSETS),
            '--steps','1000','--checkpoint-every','1000','--output-root',str(TARGET/'checkpoints')]
        save(folder/'command.json',command)
        progress(stage='training_start',candidate=label,steps=1000)
        with (folder/'training.log').open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        assert result.returncode==0,f'{label} trainer exit {result.returncode}; see training.log'
        last=json.loads((folder/'training.log').read_text().splitlines()[-1]);run=Path(last['run'])
        report=read(run/'training_report.json')
        assert report['status']=='TRAINING_SMOKE_PASS' and report['completed_steps']==1000 and report['start_step']==0
        assert report['initial_checkpoint_sha256']==cleanreport['initial_checkpoint_sha256']
        assert report['trainer_sha256']==cleanreport['trainer_sha256']
        assert [r['utterance_id'] for r in report['losses']]==[r['utterance_id'] for r in cleanreport['losses']]
        assert [r['utterance_id'] for r in report['losses']]==[r['utterance_id'] for r in eps2report['losses']]
        from melo_checkpoint import verify_checkpoint
        committed=verify_checkpoint(report['checkpoint_directory'],read(case_configs[label]))
        assert committed['step']==1000
        watch(report['checkpoint'],report['checkpoint_sha256'])
        records.append(dict(condition=label,run=str(run),checkpoint=report['checkpoint'],report=str(run/'training_report.json')))
        save(folder/'complete.json',dict(status='FRESH_1000_STEP_ATTACK_PASS',record=records[-1],seconds=report['seconds'],
            checkpoint_optimizer_rng_integrity_verified=True))
        progress(stage='training_complete',candidate=label,seconds=report['seconds'])
    summary=copy.deepcopy(old)
    summary['training']=[previous[k] for k in ('clean','SPEC_eps2_iter10')]+records
    summary['comparison_note']='Two fresh 1000-step attacks on exactly these new WAVs; archived Clean and eps2 models directly reevaluated as references.'
    summaryfile=TARGET/'training_summary.json';save(summaryfile,summary)
    import evaluate_paired_melo_smoke as E
    E.ROOT=TARGET;(TARGET/'runs').mkdir()
    from speechbrain.inference.speaker import SpeakerRecognition
    loader=SpeakerRecognition.from_hparams
    def isolated_loader(*a,**kw):
        kw['savedir']=str(TARGET/'speaker_runtime');return loader(*a,**kw)
    save(TARGET/'evaluation_adaptations.json',dict(evaluator_sha256=sha(LEGACY/'scripts/evaluate_paired_melo_smoke.py'),
        changes=['ROOT redirected to new validation directory','SpeechBrain savedir redirected to new validation directory'],
        metric_or_inference_settings_changed=False,conditions=['pretrained','clean','SPEC_eps2_iter10',*LABELS],
        prompts=30,inference_seed='2026+i',training_seed=52))
    progress(stage='evaluation_start',conditions=5,prompts_per_condition=30)
    sys.argv=['evaluate_paired_melo_smoke.py','--training-summary',str(summaryfile),'--assets',str(ASSETS)]
    with patch.object(SpeakerRecognition,'from_hparams',side_effect=isolated_loader):E.main()
    changed=[p for p,h in WATCH.items() if sha(p)!=h]
    save(TARGET/'preservation.json',dict(source_sha256=WATCH,changed=changed));assert not changed
    save(TARGET/'complete.json',dict(status='TWO_FRESH_ATTACKS_AND_EVALUATION_COMPLETE',old_files_changed=0,new_training_steps=2000,
        source_utterances_per_candidate=140,test_prompts=30,human_listening_verified=False,
        scope='Single speaker, single seed, fixed reused 30 prompts. No population/inaudibility or baseline non-inferiority claim.'))
    progress(stage='complete',old_files_changed=0)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        if TARGET.exists():
            failure=TARGET/('failure_'+str(time.time_ns())+'.json')
            save(failure,dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),
                completed_models=[str(p) for p in TARGET.glob('*/complete.json')]))
        raise
