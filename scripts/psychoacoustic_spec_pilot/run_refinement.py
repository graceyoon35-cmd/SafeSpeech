"""Diagnose the preserved 3dB pipeline, then run exactly two fixed repairs.

Only the same five train WAVs are used. No training/SIM/CER/installation/Git
mutation. Staged replays are distinguished from the immutable historical PCM;
runtime numerical differences are recorded, never overwritten or concealed.
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from research_paths import configure_paths
from types import SimpleNamespace
import numpy as np
import soundfile as sf
import torch
import run_pilot as R
from auditory_masking import build_model,audit
from projection_refinement import RefinementPerturb

MODES=('anchor_backtrack','local_caps_anchor')
sha=R.sha;save=R.save;table=R.table;read=R.read;watch=R.watch


def main(args):
    root=args.workspace.resolve();base=root/'runs/psychoacoustic_spec_refinement_20261126'
    out=base/('stage_diagnosis' if args.phase=='diagnose' else 'comparison')
    if out.exists():
        assert args.resume_diagnosis and args.phase=='diagnose' and not (out/'complete.json').exists()
    else:out.mkdir(exist_ok=False)
    wt=Path(__file__).resolve().parents[2]
    assert R.git(wt,'branch','--show-current')=='work/psychoacoustic-spec-pilot-20261125'
    preservation=read(base/'preservation_before.json')
    for e in preservation['entries']:watch(e['source'],e['sha256'])
    for p in Path(__file__).parent.glob('*'):
        if p.is_file():watch(p)
    assert R.git(root,'for-each-ref','--format=%(refname) %(objectname)','refs/heads','refs/remotes')==preservation['refs_before']
    assert not R.git(root,'status','--porcelain','--untracked-files=all')
    assert R.git(wt,'rev-parse','HEAD')==preservation['head']
    assert shutil.disk_usage(root).free>2*1024**3
    torch.set_num_threads(4);assert torch.cuda.is_available()
    free,_=torch.cuda.mem_get_info();assert free>4*1024**3
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free,utilization.gpu,temperature.gpu','--format=csv'],text=True)
    study=root/'runs/segment_noise_balance_20261124';old=root/'runs/psychoacoustic_spec_pilot_20261125/pilot'
    legacy=configure_paths(root)
    from segment_balance import segment,metrics
    from run_korean_melo_spec import check_assets,load_backend
    def progress(**kw):
        line=json.dumps(dict(time=time.time(),phase=args.phase,**kw),ensure_ascii=False)
        with (out/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(line+'\n')
        if kw.get('iteration') in (None,1,5,10):print(line,flush=True)
    dataset=read(legacy/'runs/paired_melo_experiment_ong85l9i/clean/dataset.json')
    ds={r['utterance_id']:r for r in dataset['utterances']}
    prior_metrics=R.loadcsv(study/'analysis/per_file.csv')
    references={}
    for uid in R.IDS:
        assert ds[uid]['split']=='train'
        refs={r['condition']:r['path'] for r in prior_metrics if r['utterance_id']==uid and r['condition'] in ('clean','eps2_iter10','silence_only_smooth')}
        refs['old_mask3']=str(old/'candidates/mask3db_iter10'/uid/'protected.wav')
        for p in refs.values():watch(p)
        assert sha(refs['clean'])==ds[uid]['wav_sha256'];references[uid]=refs
    if args.phase=='compare':
        diag=read(base/'stage_diagnosis/complete.json')
        assert len(read(base/'stage_diagnosis/baseline_reproduction.json'))==5
    planpath=out/('resume_plan_'+str(time.time_ns())+'.json' if args.resume_diagnosis else 'plan.json')
    save(planpath,dict(phase=args.phase,ids=R.IDS,epsilon=2/255,margin_db=3,iters=10,seed=52,beta=10,
        variants=['baseline_trace'] if args.phase=='diagnose' else MODES,
        change_scope='Only feasible-PCM repair of the same continuous proposal; fixed spectral model/radii/step/acceptance/RNG',
        max_improvement_variants=2,no_adaptive_followup=True,no_140_expansion=True,no_melo_training=True,
        no_sim_cer=True,nvidia_smi=gpu,free_cuda_bytes=free,head=preservation['head']))
    assets,cfg,hashes=check_assets(Path('/home/safespeech/models/melo-spec-pinned-20261118'))
    for p,h in hashes.items():watch(assets/p,h)
    backend=out/('backend_resume_'+str(time.time_ns()) if args.resume_diagnosis else 'backend')
    backend.mkdir();generate=load_backend(assets,cfg,backend,'cuda')
    import protect_melo as P
    stages=[];quality=[];objectives=[];mask_rows=[];replayed=[];diagnostics=[]
    modes=('baseline_trace',) if args.phase=='diagnose' else MODES
    started=time.perf_counter()
    for uid in R.IDS:
        refs=references[uid];x,sr=sf.read(refs['clean']);xc,_=sf.read(refs['clean'],dtype='int16')
        labels,flabels,_,_=segment(x);models=build_model(x)
        stored=np.load(watch(old/'thresholds'/(uid+'.npz')))
        for m in models:np.testing.assert_array_equal(m.raw_threshold,stored[f'raw_threshold_{m.n_fft}'])
        radius=stored['radius_codes'];baseline=sf.read(refs['eps2_iter10'])[0]-x
        newpaths={}
        for mode in modes:
            job=out/mode/uid
            target=job/'protected.wav'
            settings=SimpleNamespace(epsilon=2.,iters=10,step=None,seed=52,mode='SPEC',beta=10.,alpha=.05,log_every=1,random_init=False)
            progress(stage='case_start',id=uid,mode=mode)
            if (job/'generation.json').exists():
                assert args.resume_diagnosis
                previous=read(job/'generation.json');assert sha(target)==previous['output_sha256']
                assert previous['source_sha256']==sha(refs['clean'])
                stats=previous['stats'];final=previous['reread_final_pcm_objective']
                progress(stage='resume_existing_case',id=uid,mode=mode)
            else:
                job.mkdir(parents=True)
                optimizer=RefinementPerturb(models,radius,refs,mode,lambda **kw:progress(stage='update',id=uid,**kw))
                P.perturb=optimizer
                stats=generate(Path(refs['clean']),target,{'model_speaker':'KR'},settings)
                final=optimizer.measure_file(target);assert abs(final['loss_SPEC']-stats['final_pcm_objective']['loss_SPEC'])<.01
                save(job/'generation.json',dict(settings=vars(settings),stats=stats,reread_final_pcm_objective=final,
                    source=refs['clean'],source_sha256=sha(refs['clean']),output_sha256=sha(target)))
            y,rate=sf.read(target,dtype='int16');delta=y.astype(np.int32)-xc.astype(np.int32)
            assert rate==sr and y.shape==xc.shape and abs(delta).max()<=257
            assert not np.any((y==-32768)|(y==32767))
            checks=audit(delta/32768.,models,3);assert all(r['pass_constraint'] for r in checks)
            if args.phase=='diagnose':
                original,osr=sf.read(refs['old_mask3'],dtype='int16')
                equal=bool(np.array_equal(original,y))
                difference=y.astype(np.int32)-original.astype(np.int32)
                replayed.append(dict(id=uid,pcm_equal=equal,changed_samples=int(np.count_nonzero(difference)),
                    max_sample_difference_codes=int(np.abs(difference).max()),old_sha256=sha(refs['old_mask3']),replay_sha256=sha(target),
                    old_pcm_loss=stats['objective_references']['old_mask3']['loss_SPEC'],replay_pcm_loss=final['loss_SPEC'],
                    interpretation='Diagnostic runtime replay, not a replacement for immutable historical comparison WAV'))
            for r in stats['stage_measurements']:stages.append(dict(id=uid,**r))
            objectives.append(dict(id=uid,condition=mode,**final,
                reduction_vs_clean=stats['objective_references']['clean']['loss_SPEC']-final['loss_SPEC'],
                retained_vs_eps2=stats['reference_improvement_retained']['eps2_iter10'],
                retained_vs_silence=stats['reference_improvement_retained']['silence_only_smooth'],
                improvement_vs_old=stats['objective_references']['old_mask3']['loss_SPEC']-final['loss_SPEC'],
                accepted_steps=stats['accepted_steps'],clean_fallback=stats['clean_fallback']))
            diagnostics.append(dict(id=uid,condition=mode,selected_repair=stats['selected_repair']))
            newpaths[mode]=str(target)
            progress(stage='case_complete',id=uid,mode=mode,retained_vs_eps2=stats['reference_improvement_retained']['eps2_iter10'])
        if args.phase=='compare':
            allpaths={'old_mask3':refs['old_mask3'],**newpaths}
            for name,path in allpaths.items():
                y,rate=sf.read(path);assert rate==sr and y.shape==x.shape
                q=metrics(x,y,labels,flabels,baseline);assert q['clipping']==0 and q['linf']<=2/255
                check=audit(y-x,models,3);assert all(r['pass_constraint'] for r in check)
                decoded=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',path,'-f','null','-'],capture_output=True)
                assert decoded.returncode==0
                quality.append(dict(id=uid,condition=name,path=path,sha256=sha(path),strict_epsilon_pass=True,**q))
                mask_rows.extend(dict(id=uid,condition=name,**c) for c in check)
            refstats=stats['objective_references']
            r=refstats['old_mask3'];denom=refstats['clean']['loss_SPEC']-refstats['eps2_iter10']['loss_SPEC']
            objectives.append(dict(id=uid,condition='old_mask3',**r,reduction_vs_clean=refstats['clean']['loss_SPEC']-r['loss_SPEC'],
                retained_vs_eps2=(refstats['clean']['loss_SPEC']-r['loss_SPEC'])/denom))
    if stages:table(out/'stage_measurements.csv',stages)
    if quality:table(out/'quality_per_file.csv',quality)
    if mask_rows:table(out/'masking_per_file.csv',mask_rows)
    table(out/'objective_per_file.csv',objectives);save(out/'repair_diagnostics.json',diagnostics)
    if replayed:save(out/'baseline_reproduction.json',replayed)
    if args.phase=='compare':
        summaries=[]
        for mode in ('old_mask3',*MODES):
            a=[r for r in quality if r['condition']==mode];o=[r for r in objectives if r['condition']==mode]
            masks=[r for r in mask_rows if r['condition']==mode]
            summary=dict(condition=mode,n=5,clipping=sum(r['clipping'] for r in a),epsilon_failures=0,
                masking_violating_bins=sum(r['violating_bins'] for r in masks),
                masking_bins=sum(r['bins'] for r in masks),max_mask_excess_db=max(r['max_excess_db'] for r in masks),
                mean_loss_SPEC=float(np.mean([r['loss_SPEC'] for r in o])),
                mean_retained_vs_eps2=float(np.mean([r['retained_vs_eps2'] for r in o])))
            for key in ('snr_db','stoi','silence_noise_rms','weak_noise_rms','voiced_noise_rms'):
                values=[r[key] for r in a if r[key] is not None]
                summary[key]=float(np.mean(values)) if values else None
            summary['snr_unavailable_count']=sum(r['snr_db'] is None for r in a)
            if mode!='old_mask3':
                summary['improved_objective_files']=sum(r['improvement_vs_old']>0 for r in o)
                summary['clean_fallback_count']=sum(r['clean_fallback'] for r in o)
            summaries.append(summary)
        table(out/'summary.csv',summaries);save(out/'summary.json',summaries)
    changed=[p for p,h in R.WATCH.items() if sha(p)!=h];assert not changed
    assert R.git(root,'for-each-ref','--format=%(refname) %(objectname)','refs/heads','refs/remotes')==preservation['refs_before']
    assert not R.git(root,'status','--porcelain','--untracked-files=all')
    save(out/'preservation.json',dict(source_sha256=R.WATCH,changed=changed))
    save(out/'complete.json',dict(status='COMPLETE',phase=args.phase,ids=R.IDS,
        baseline_replay_equal_to_original_count=sum(r['pcm_equal'] for r in replayed),improvement_variants=0 if args.phase=='diagnose' else 2,
        protected_wavs=5 if args.phase=='diagnose' else 10,seconds=time.perf_counter()-started,
        no_training=True,no_sim_cer=True,no_commit_push=True,no_adaptive_followup=True))
    print(json.dumps(dict(phase=args.phase,status='COMPLETE',seconds=time.perf_counter()-started),ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True,type=Path);p.add_argument('--phase',choices=('diagnose','compare'),required=True)
    p.add_argument('--resume-diagnosis',action='store_true',help='Reuse completed diagnostic cases only; never overwrite prior files')
    a=p.parse_args()
    try:main(a)
    except BaseException as e:
        out=a.workspace/'runs/psychoacoustic_spec_refinement_20261126'/('stage_diagnosis' if a.phase=='diagnose' else 'comparison')
        if out.exists():save(out/('failure_'+str(time.time_ns())+'.json'),dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
        raise
