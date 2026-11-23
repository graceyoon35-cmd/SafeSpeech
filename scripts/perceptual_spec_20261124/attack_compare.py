"""Two fresh same-protocol attacks; archived Clean/eps2 are references only."""
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
from experiment_io import ROOT,OUT,LEGACY,ASSETS,PROFILES,save,sha
from audio_analysis import analyze
from perceptual_objective import install
from run_korean_melo_spec import check_assets,load_backend
from protect_wav import project_pcm16

TARGET=OUT/'attacks'
CANDIDATES=('eps1_iter5','mask_eps2_iter10_s2')
EXTRA_REFERENCES=[]
SELECTION_NOTE='eps1_iter5 represents reduced radius/updates; weighted strength2 showed lower high-frequency and quiet residual than eps2_iter10 in the diagnostic pilot.'
def progress(**fields):
    data=dict(time=time.time(),**fields)
    with (TARGET/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(data)+'\n')
    print(json.dumps(data),flush=True)

def main():
    TARGET.mkdir(exist_ok=False)
    previous=ROOT/'runs/demo_noise_fix_20261124/defense_attack_140/training_summary.json'
    old=json.loads(previous.read_text());by={r['condition']:r for r in old['training']}
    experiment=Path(old['experiment'])
    base=json.loads((experiment/'clean/config.json').read_text())
    data=json.loads((experiment/'clean/dataset.json').read_text())
    source={str(previous):sha(previous)}
    for r in data['utterances']:
        for key in ('wav','features'):
            assert sha(r[key])==r[key+'_sha256'];source[r[key]]=sha(r[key])
    for reference in [by[label] for label in ('clean','SPEC_eps2_iter10')]+EXTRA_REFERENCES:
        source[reference['checkpoint']]=sha(reference['checkpoint'])
        source[reference['report']]=sha(reference['report'])
    source[str(LEGACY/'scripts/train_melo_bounded.py')]=sha(LEGACY/'scripts/train_melo_bounded.py')
    save(TARGET/'plan.json',dict(candidates=CANDIDATES,selection=SELECTION_NOTE,extra_references=EXTRA_REFERENCES,
        train_count=140,steps=1000,seed=52,test_prompts=30,source_sha256=source,
        human_listening='PENDING, no imperceptibility claim',scope='Same architecture/initial weights/trainer/seed/order/held-out data; one speaker and seed.'))
    records=[]
    baseline_report=json.loads(Path(by['clean']['report']).read_text())
    for label in CANDIDATES:
        profile=PROFILES[label];folder=TARGET/label;folder.mkdir()
        for name in ('common','protect_melo'):sys.modules.pop(name,None)
        assets,cfg,hashes=check_assets(ASSETS)
        source.update({str(assets/p):sha(assets/p) for p in hashes})
        backend=folder/'backend';backend.mkdir()
        generate=load_backend(assets,cfg,backend,'cuda')
        install(sys.modules['protect_melo'],backend)
        updated=copy.deepcopy(data);qualities=[]
        for row in updated['utterances']:
            if row['split']!='train':continue
            original=Path(row['wav']);job=folder/'protected_train'/row['utterance_id'];job.mkdir(parents=True)
            args=SimpleNamespace(epsilon=profile['epsilon'],step=None,iters=profile['iterations'],seed=52,mode='SPEC',
                beta=10.,alpha=.05,log_every=1,random_init=False,perceptual_strength=profile['strength'])
            raw=job/'optimized.wav';output=job/'protected.wav'
            stats=generate(original,raw,{'model_speaker':'KR'},args)
            x,sr=sf.read(original);y,_=sf.read(raw);codes,adjusted=project_pcm16(x,y,args.epsilon)
            with output.open('xb') as f:sf.write(f,codes,sr,format='WAV',subtype='PCM_16')
            q=analyze(original,output);assert q['linf']<=args.epsilon/255 and not q['clipping']
            qualities.append(dict(id=row['utterance_id'],**q))
            save(job/'generation.json',dict(settings=vars(args),stats=stats,quality=q,adjusted_samples=adjusted))
            row.update(wav=str(output),wav_sha256=sha(output))
            if len(qualities)%10==0:progress(stage='protect',candidate=label,completed=len(qualities),total=140)
        del generate;gc.collect();torch.cuda.empty_cache()
        assert len(qualities)==140
        save(folder/'quality.json',dict(n=140,mean_snr_db=float(np.mean([r['snr_db'] for r in qualities])),
            mean_stoi=float(np.mean([r['stoi'] for r in qualities])),clipping=sum(r['clipping'] for r in qualities),per_file=qualities))
        dataset=folder/'dataset.json';save(dataset,updated)
        config=copy.deepcopy(base)
        config['experiment'].update(condition=label,dataset_manifest=str(dataset),profile=profile)
        mapping={r['utterance_id']:r['wav'] for r in updated['utterances'] if r['split']=='train'}
        metadata=folder/'metadata.list.cleaned'
        with metadata.open('x',encoding='utf-8') as f:
            for line in Path(base['data']['training_files']).read_text().splitlines():
                fields=line.split('|');fields[0]=mapping[Path(fields[0]).stem];f.write('|'.join(fields)+'\n')
        config['data']['training_files']=str(metadata)
        configfile=folder/'config.json';save(configfile,config)
        command=[sys.executable,'-B','-u',str(LEGACY/'scripts/train_melo_bounded.py'),'--config',str(configfile),
            '--assets',str(ASSETS),'--steps','1000','--checkpoint-every','1000','--output-root',str(TARGET/'checkpoints')]
        save(folder/'command.json',command)
        progress(stage='training_start',candidate=label,steps=1000)
        with (folder/'training.log').open('x') as log:
            proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
        if proc.returncode:raise RuntimeError(f'Attack failed: {label} code {proc.returncode}')
        last=json.loads((folder/'training.log').read_text().splitlines()[-1]);run=Path(last['run'])
        report=json.loads((run/'training_report.json').read_text())
        assert report['status']=='TRAINING_SMOKE_PASS' and report['completed_steps']==1000
        assert report['trainer_sha256']==baseline_report['trainer_sha256']
        assert report['initial_checkpoint_sha256']==baseline_report['initial_checkpoint_sha256']
        assert [r['utterance_id'] for r in report['losses']]==[r['utterance_id'] for r in baseline_report['losses']]
        records.append(dict(condition=label,run=str(run),checkpoint=report['checkpoint'],report=str(run/'training_report.json')))
        progress(stage='training_complete',candidate=label,seconds=report['seconds'])
    summary=copy.deepcopy(old);summary['training']=[by[c] for c in ('clean','SPEC_eps2_iter10')]+EXTRA_REFERENCES+records
    summary.update(comparison_note='Archived Clean/eps2 re-evaluated as their own references; new attack data and models for both new candidates.')
    summaryfile=TARGET/'training_summary.json';save(summaryfile,summary)
    # Verify metric weights before using the unchanged evaluator, without downloads.
    evalroot=Path('/home/safespeech/models/melo-evaluation-20261118')
    provenance=json.loads((evalroot/'model_provenance.json').read_text())
    for relative,expected in provenance['sha256'].items():
        assert sha(evalroot/relative)==expected;source[str(evalroot/relative)]=expected
    import evaluate_paired_melo_smoke as E
    E.ROOT=TARGET;(TARGET/'runs').mkdir()
    from speechbrain.inference.speaker import SpeakerRecognition
    original_loader=SpeakerRecognition.from_hparams
    def loader(*a,**kw):
        kw['savedir']=str(TARGET/'speaker_runtime');return original_loader(*a,**kw)
    progress(stage='evaluation_start',test_prompts=30,conditions=len(summary['training'])+1)
    sys.argv=['evaluate_paired_melo_smoke.py','--training-summary',str(summaryfile),'--assets',str(ASSETS)]
    with patch.object(SpeakerRecognition,'from_hparams',side_effect=loader):E.main()
    changed=[p for p,h in source.items() if sha(p)!=h]
    save(TARGET/'preservation.json',dict(source_sha256=source,changed=changed));assert not changed
    progress(stage='complete',old_files_changed=0)

if __name__=='__main__':
    try:main()
    except BaseException as e:
        if TARGET.exists():save(TARGET/'failure.json',dict(type=type(e).__name__,message=str(e)))
        raise
