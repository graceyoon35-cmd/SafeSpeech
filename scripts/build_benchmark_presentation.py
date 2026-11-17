"""Verified 100-step tables, conditional runtime estimates and presentation figures."""
import argparse
import csv
import json
from pathlib import Path
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_korean_melo_spec import save,sha256,validate_handoff


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_csv(path,rows):
    with Path(path).open('x',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--robustness',type=Path,required=True)
    p.add_argument('--storage',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    bench=load(args.benchmark);evaluation=load(args.evaluation);robust=load(args.robustness);storage=load(args.storage)
    if bench['status']!='PAIRED_SYNTHETIC_TRAINING_SMOKE_PASS' or bench['steps_per_condition']!=100:
        raise ValueError('Completed 100-step benchmark required')
    if evaluation['test_utterances']!=12 or evaluation['status']!='SYNTHETIC_SMOKE_EVALUATION_PASS':
        raise ValueError('Completed 12-prompt synthetic evaluation required')
    if robust['status']!='PAIRED_SYNTHETIC_QUALITY_PASS':raise ValueError('Robustness quality run incomplete')
    args.output.mkdir(parents=True,exist_ok=False)
    timings=[];histories={};phases=[(row,load(row['report'])) for row in bench['phases']]
    resume_results=[]
    for trained in bench['training']:
        report=load(trained['report']);condition=trained['condition'];history=report['losses']
        if sha256(report['checkpoint'])!=report['checkpoint_sha256'] or [r['step'] for r in history]!=list(range(1,101)):
            raise ValueError('Checkpoint or step history inconsistency')
        if report['restored_optimizer_steps']!=[[50],[50],[50]]:raise ValueError('Resume optimizer state mismatch')
        cfg=load(Path(trained['run'])/'config.json')
        if cfg['train']['seed']!=52:raise ValueError('Unexpected training seed')
        values=[r['step_seconds_excluding_checkpoint'] for r in history]
        relevant=[r for meta,r in phases if meta['condition']==condition]
        saves=[c['save_seconds'] for r in relevant for c in r['checkpoints']]
        warmup_filtered=[r['step_seconds_excluding_checkpoint'] for r in history if r['step'] not in list(range(1,6))+list(range(51,56))]
        timings.append({'condition':condition,'steps':100,'train_utterances':2,'training_seed':52,
            'mean_step_seconds':float(np.mean(values)),'median_step_seconds':float(np.median(values)),
            'p90_step_seconds':float(np.quantile(values,.9)),'steady_mean_step_seconds':float(np.mean(warmup_filtered)),
            'total_step_seconds':sum(values),'checkpoint_save_seconds_total':sum(saves),
            'timed_training_sessions_seconds':sum(r['seconds'] for r in relevant),
            'peak_allocated_gib':max(r['peak_allocated_bytes'] for r in relevant)/1024**3,
            'peak_reserved_gib':max(r['peak_reserved_bytes'] for r in relevant)/1024**3,
            'generator_loss_mean_first10':float(np.mean([r['generator'] for r in history[:10]])),
            'generator_loss_mean_last10':float(np.mean([r['generator'] for r in history[-10:]])),
            'checkpoint':report['checkpoint'],'resume_verified':True})
        histories[condition]=history
        resume_results.append({'condition':condition,'restored_steps':report['restored_optimizer_steps'],'start_step':report['start_step'],
            'end_step':report['completed_steps'],'checkpoint_hash_verified':True})
    # Independent fresh-process replays from step 50: compare losses, not timings.
    native=load(storage['native_training_report']);reference=histories['clean']
    fields=['generator','discriminator','duration_discriminator','mel','kl','duration','feature','adversarial','duration_adversarial','target_rms','prediction_rms']
    replay_delta=max(abs(native['losses'][i][key]-reference[i][key]) for i in (50,51) for key in fields)
    estimates=[]
    by_condition={r['condition']:r for r in timings}
    for steps in (1000,5000,10000):
        for conditions in [('clean','SPEC_10'),('clean','SPEC_1','SPEC_10')]:
            compute=sum(by_condition[c]['mean_step_seconds']*steps for c in conditions)
            save_count=math.ceil(steps/500)*len(conditions)
            seconds=compute+save_count*storage['native_checkpoint_save_seconds']
            estimates.append({'steps_per_condition':steps,'conditions':'+'.join(conditions),'checkpoint_interval':500,
                'estimated_training_and_save_minutes':seconds/60,'planning_minutes_with_50pct_margin':seconds*1.5/60,
                'estimated_epochs_if_140_train_batch1':steps/140,
                'basis':'Synthetic 2-utterance measured mean; native WSL save measured once',
                'excludes':'KSS permission/review, SPEC, preprocessing, startup/preflight, synthesis/evaluation',
                'warning':'Conditional estimate, not a measured KSS training time or convergence guarantee'})
    write_csv(args.output/'training_timing.csv',timings)
    write_csv(args.output/'training_time_estimates.csv',estimates)
    write_csv(args.output/'evaluation_summary.csv',evaluation['measurements'])
    # Verify every synthesized evaluation artifact before plotting its measurements.
    with (args.evaluation.parent/'per_file.csv').open(encoding='utf-8-sig',newline='') as stream:
        per_file=list(csv.DictReader(stream))
    if len(per_file)!=48:raise ValueError('Expected 4 conditions x 12 prompts')
    for row in per_file:
        if sha256(row['output'])!=row['output_sha256'] or sha256(row['reference'])!=row['checkpoint_sha256']:
            raise ValueError('Evaluation source/output hash changed')
    write_csv(args.output/'evaluation_per_file.csv',per_file)
    experiment=load(Path(bench['experiment'])/'experiment.json')
    handoff=load(Path(experiment['source_spec_run'])/'handoff_manifest.json')['utterances'];validate_handoff(handoff)
    write_csv(args.output/'spec_technical_results.csv',[{'utterance_id':r['utterance_id'],'split':r['split'],'iterations':r['iterations'],
        'epsilon':r['epsilon'],'stoi':r['stoi'],'input_kind':r['input_kind']} for r in handoff])
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(14,4.6))
    for condition,history in histories.items():
        x=np.arange(1,101);y=np.array([r['generator'] for r in history]);kernel=np.ones(10)/10
        axes[0].plot(x[9:],np.convolve(y,kernel,mode='valid'),label=condition)
    axes[0].set(title='Generator loss (10-step moving mean)',xlabel='Optimizer step',ylabel='Loss');axes[0].legend()
    axes[1].bar([r['condition'] for r in timings],[r['mean_step_seconds'] for r in timings]);axes[1].set(title='Measured training step',ylabel='Seconds (checkpoint excluded)')
    axes[2].bar([r['condition'] for r in timings],[r['peak_allocated_gib'] for r in timings]);axes[2].set(title='PyTorch peak allocated',ylabel='GiB',ylim=(0,8))
    fig.suptitle('100-step synthetic benchmark: 2 training utterances, one training seed; not KSS')
    fig.tight_layout();fig.savefig(args.output/'training_100step.png',dpi=180);fig.savefig(args.output/'training_100step.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    measurements=evaluation['measurements'];labels=[r['condition'] for r in measurements]
    for ax,metric,scale,title in [(axes[0],'sim',1.,'ECAPA cosine SIM'),(axes[1],'cer',100.,'Mean Korean CER (%)')]:
        means=np.array([r['mean_'+metric]*scale for r in measurements])
        low=np.array([r[metric+'_ci95_low']*scale for r in measurements]);high=np.array([r[metric+'_ci95_high']*scale for r in measurements])
        ax.bar(labels,means,yerr=np.vstack([means-low,high-means]),capsize=5,color=['#718096','#3179ab','#ef9e42','#c95446'])
        ax.set(title=title);ax.tick_params(axis='x',rotation=20)
    axes[0].set_ylim(0,1)
    fig.suptitle('100 steps, 12 fixed prompts: 95% bootstrap over prompts only\nSynthetic experiment; one speaker/seed, not human-defense validation',fontsize=11)
    fig.tight_layout();fig.savefig(args.output/'evaluation_12prompts.png',dpi=180);fig.savefig(args.output/'evaluation_12prompts.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(14,5))
    for count in (1,10):
        scores=[np.mean([r['paired_channel_stoi'] for r in robust['rows'] if r['iterations']==count and r['condition_index']==i]) for i in range(15)]
        ax.plot(range(15),scores,marker='o',label=f'SPEC {count}')
    labels=[]
    for c in robust['conditions']:
        labels.append(f"Gauss {c['std']} / s{c['seed']}" if c['name']=='gaussian_noise' else f"MP3 {c['bitrate_kbps']}k" if c['name']=='mp3' else f"Resample {c['target_hz']}Hz")
    ax.set_xticks(range(15),labels,rotation=40,ha='right');ax.set(ylabel='STOI of paired transformed clean/protected',ylim=(0,1.02))
    ax.legend();ax.set_title('Synthetic held-out input: 15 channel conditions, 2 deterministic repeats\nQuality only; no retraining/attack-defense robustness measured')
    fig.tight_layout();fig.savefig(args.output/'robustness_quality.png',dpi=180);fig.savefig(args.output/'robustness_quality.svg');plt.close(fig)
    write_csv(args.output/'robustness_quality.csv',robust['rows'])
    old_integrity=load(args.benchmark.parents[1]/'training_20261118/integrity/integrity.json')
    preserved=[]
    for path,expected in old_integrity['checkpoints_and_training_state'].items():
        if sha256(path)!=expected['sha256']:raise ValueError('Prior smoke artifact changed')
        preserved.append(path)
    root=args.benchmark.parents[2]
    kss=root/'runs/desktop_20261118/data/data_set/kss_part_200/spec_input.json'
    current=load(kss)['utterances']
    if sha256(kss)!=old_integrity['kss_manifest_sha256']:raise ValueError('KSS source manifest changed')
    save(args.output/'final_report.json',{'status':'VERIFIED_SYNTHETIC_100_STEP_BENCHMARK','training':timings,
        'evaluation':evaluation['measurements'],'estimates':estimates,'checkpoint_recovery':resume_results,
        'replayed_step51_52_max_loss_difference':replay_delta,
        'replay_scope':'Two fresh-process resumes from same step50 state, not uninterrupted or cross-device reproducibility',
        'storage':storage,'old_smoke_artifacts_preserved':preserved,
        'kss':{'utterances':len(current),'transcript_verified_count':sum(r['transcript_verified'] is True for r in current),
               'research_use_permitted_count':sum(r['research_use_permitted'] is True for r in current),
               'human_spec_steps':0,'human_training_steps':0},
        'robustness':{'wav_count':robust['wav_count'],'mp3_count':robust['mp3_count'],
            'source_utterances':robust['distinct_original_utterances'],'scope':robust['limitations']},
        'limitations':['Synthetic training data only','One training seed','2 train utterances',
            'Bootstrap intervals capture prompt variability only','KSS time is projected, not measured',
            'Session seconds exclude preflight validation; step timers use CUDA synchronization',
            'No human-defense efficacy or channel-trained attack experiment']})
    print(json.dumps({'directory':str(args.output.resolve()),'status':'VERIFIED'}),flush=True)


if __name__=='__main__':main()
