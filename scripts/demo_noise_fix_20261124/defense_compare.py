"""Bounded fresh-input attack comparison. No writes outside this new run.

Same historical KSS train order/features/seed/trainer/1000 steps and 30 held-out
prompts. Historical Clean and eps8 checkpoints are reference models, never
evidence for eps2/eps4. Newly protect only train WAVs; tests stay held out.
"""
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from prepare_demo import HERE,OLD,ROOT,sha,save,quality
sys.path.insert(0,str(OLD/'scripts'))
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from run_korean_melo_spec import check_assets,load_backend
from protect_wav import project_pcm16

ASSETS=Path('/home/safespeech/models/melo-spec-pinned-20261118')
RESUME='--resume-after-eps2' in sys.argv
TARGET=HERE/'defense_attack_140';TARGET.mkdir(exist_ok=RESUME)
SUMMARY=OLD/'runs/paired_melo_training_rhp4a8t2/summary.json'
EXPERIMENT=OLD/'runs/paired_melo_experiment_ong85l9i'
watched={}
def watch(p):
    p=Path(p);watched[str(p)]=sha(p);return p

def progress(**kwargs):
    record=dict(time=time.time(),**kwargs)
    with (TARGET/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    print(json.dumps(record,ensure_ascii=False),flush=True)

def main():
    import torch
    original=json.loads(watch(SUMMARY).read_text())
    base=json.loads(watch(EXPERIMENT/'clean/config.json').read_text())
    dataset=json.loads(watch(EXPERIMENT/'clean/dataset.json').read_text())
    historical={r['condition']:r for r in original['training']}
    for row in original['training']:watch(row['checkpoint']);watch(row['report'])
    for row in dataset['utterances']:
        for key in ('wav','features'):
            if sha(watch(row[key]))!=row[key+'_sha256']:raise ValueError('Source hash mismatch')
    train=[r for r in dataset['utterances'] if r['split']=='train']
    tests=[r for r in dataset['utterances'] if r['split']=='test']
    assert len(train)==140 and len(tests)==30
    assert original['steps_per_condition']==1000 and base['train']['seed']==52
    free=shutil.disk_usage(TARGET).free
    if free<12*1024**3:raise RuntimeError('Need at least 12GiB free for isolated checkpoints')
    plan=dict(train_count=len(train),test_count=len(tests),conditions=['eps2_iter10','eps4_iter10'],
        steps=1000,training_seed=52,perturbation_seed=52,initial_checkpoint_sha256=original['initial_checkpoint_sha256'],
        historical_baselines={c:historical[c] for c in ('clean','SPEC_10')},
        clean_baseline_retrained=False,defense_evaluation='Fresh-input finetuning attacks then same held-out inference/ECAPA/Whisper settings; one speaker, one seed.',
        old_trainer=str(OLD/'scripts/train_melo_bounded.py'),old_trainer_sha256=sha(watch(OLD/'scripts/train_melo_bounded.py')),
        baseline_source_summary_sha256=sha(SUMMARY),free_bytes=free)
    if RESUME:
        previous=json.loads((TARGET/'plan.json').read_text())
        assert previous['baseline_source_summary_sha256']==sha(SUMMARY)
        assert previous['old_trainer_sha256']==sha(OLD/'scripts/train_melo_bounded.py')
        save(TARGET/'resume_plan.json',dict(reason='Resume after safe module-path guard stopped second backend load',
            completed_condition='SPEC_eps2_iter10',repeat_completed_training=False,plan=plan))
    else:save(TARGET/'plan.json',plan)
    reports=[]
    for epsilon in (2.,4.):
        label=f'SPEC_eps{int(epsilon)}_iter10'
        if RESUME and epsilon==2:
            log=TARGET/label/'training.log'
            reply=json.loads(log.read_text().splitlines()[-1]);run=Path(reply['run'])
            report=json.loads((run/'training_report.json').read_text())
            assert report['status']=='TRAINING_SMOKE_PASS' and report['completed_steps']==1000
            assert sha(report['checkpoint'])==report['checkpoint_sha256']
            reports.append(dict(condition=label,run=str(run),checkpoint=report['checkpoint'],report=str(run/'training_report.json')))
            progress(stage='resume_kept_complete_model',condition=label,checkpoint=report['checkpoint'])
            continue
        folder=TARGET/label;folder.mkdir(exist_ok=RESUME)
        assets,cfg,hashes=check_assets(ASSETS)
        for name in hashes:watch(assets/name)
        backend=folder/('backend_resume' if RESUME else 'backend');backend.mkdir()
        # Each provenance check requires modules imported from this job's exact source copy.
        for module in ('common','protect_melo'):sys.modules.pop(module,None)
        generate=load_backend(assets,cfg,backend,'cuda')
        changed=copy.deepcopy(dataset)
        qrows=[]
        for i,row in enumerate(changed['utterances']):
            if row['split']!='train':continue
            job=folder/'protected_train'/row['utterance_id'];job.mkdir(parents=True)
            settings=SimpleNamespace(epsilon=epsilon,step=None,iters=10,seed=52,mode='SPEC',beta=10.,alpha=.05,log_every=0,random_init=False)
            raw=job/'optimized.wav';out=job/'protected.wav'
            stats=generate(Path(row['wav']),raw,{'model_speaker':'KR'},settings)
            x,sr=sf.read(row['wav']);y,yr=sf.read(raw)
            codes,adjusted=project_pcm16(x,y,epsilon)
            with out.open('xb') as f:sf.write(f,codes,sr,format='WAV',subtype='PCM_16')
            q=quality(row['wav'],out,epsilon)
            save(job/'generation.json',dict(settings=vars(settings),stats=stats,quality=q,pcm16_adjusted_samples=adjusted,assets_sha256=hashes))
            qrows.append(dict(utterance_id=row['utterance_id'],**q))
            row.update(wav=str(out),wav_sha256=sha(out))
            if len(qrows)%10==0:progress(stage='protect',condition=label,completed=len(qrows),total=len(train))
        del generate;gc.collect();torch.cuda.empty_cache()
        save(folder/'quality.json',dict(n=len(qrows),mean_snr_db=float(np.mean([r['snr_db'] for r in qrows])),
            mean_stoi=float(np.mean([r['stoi'] for r in qrows])),clipping=sum(r['clipping'] for r in qrows),per_file=qrows))
        manifest=folder/'dataset.json';save(manifest,changed)
        config=copy.deepcopy(base)
        config['experiment'].update(condition=label,dataset_manifest=str(manifest),candidate_epsilon=epsilon/255,candidate_iterations=10)
        # Same order, phones and metadata; only train WAV path is substituted.
        metadata=folder/'metadata.list.cleaned'
        mapping={r['utterance_id']:r['wav'] for r in changed['utterances'] if r['split']=='train'}
        old_lines=watch(Path(base['data']['training_files'])).read_text().splitlines()
        with metadata.open('x',encoding='utf-8') as f:
            for line in old_lines:
                parts=line.split('|');uid=Path(parts[0]).stem;parts[0]=mapping[uid];f.write('|'.join(parts)+'\n')
        config['data']['training_files']=str(metadata)
        configpath=folder/'config.json';save(configpath,config)
        # The original trainer is called unchanged. output_root and -B isolate all writes.
        command=[sys.executable,'-B','-u',str(OLD/'scripts/train_melo_bounded.py'),'--config',str(configpath),'--assets',str(ASSETS),
            '--steps','1000','--checkpoint-every','1000','--output-root',str(TARGET/'checkpoints')]
        save(folder/'command.json',command)
        progress(stage='training_start',condition=label,steps=1000)
        with (folder/'training.log').open('x') as f:
            result=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
        if result.returncode:raise RuntimeError(f'{label} training failed: {result.returncode}')
        reply=json.loads((folder/'training.log').read_text().splitlines()[-1]);run=Path(reply['run'])
        report=json.loads((run/'training_report.json').read_text())
        assert report['status']=='TRAINING_SMOKE_PASS' and report['completed_steps']==1000
        assert report['initial_checkpoint_sha256']==original['initial_checkpoint_sha256']
        clean_report=json.loads(Path(historical['clean']['report']).read_text())
        assert report['trainer_sha256']==clean_report['trainer_sha256']
        assert [r['utterance_id'] for r in report['losses']]==[r['utterance_id'] for r in clean_report['losses']]
        reports.append(dict(condition=label,run=str(run),checkpoint=report['checkpoint'],report=str(run/'training_report.json')))
        progress(stage='training_complete',condition=label,seconds=report['seconds'])
    summary=copy.deepcopy(original)
    summary['training']=[dict(historical[c]) for c in ('clean','SPEC_10')]+reports
    summary.update(source_summary=str(SUMMARY),source_summary_sha256=sha(SUMMARY),
        comparison_note='Clean/SPEC_10 are archived same-seed/steps references. Only eps2/eps4 have freshly generated attack training inputs.')
    summary_path=TARGET/'training_summary.json';save(summary_path,summary)
    progress(stage='evaluation_start',prompts=30,conditions=5)
    import evaluate_paired_melo_smoke as E
    E.ROOT=TARGET  # Existing evaluator creates runs/ below this explicitly new root.
    (TARGET/'runs').mkdir()
    from speechbrain.inference.speaker import SpeakerRecognition
    original_loader=SpeakerRecognition.from_hparams
    def isolated_loader(*args,**kwargs):
        kwargs['savedir']=str(TARGET/'speaker_runtime')
        return original_loader(*args,**kwargs)
    save(TARGET/'evaluation_adaptations.json',dict(evaluator_sha256=sha(watch(OLD/'scripts/evaluate_paired_melo_smoke.py')),
        changes=['ROOT redirected to new defense folder','SpeechBrain runtime savedir redirected to new defense folder'],
        metric_or_inference_settings_changed=False))
    sys.argv=['evaluate_paired_melo_smoke.py','--training-summary',str(summary_path),'--assets',str(ASSETS)]
    with patch.object(SpeakerRecognition,'from_hparams',side_effect=isolated_loader):E.main()
    changed_sources=[p for p,h in watched.items() if sha(p)!=h]
    save(TARGET/'preservation.json',dict(files=watched,changed=changed_sources))
    assert not changed_sources
    progress(stage='complete',changed_source_files=0)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        save(TARGET/('failure_resume.json' if RESUME else 'failure.json'),dict(type=type(e).__name__,message=str(e)))
        raise
