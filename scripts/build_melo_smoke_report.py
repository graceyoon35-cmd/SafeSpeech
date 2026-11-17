"""Publish local tables/figures from verified measurements; never infer efficacy."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from run_korean_melo_spec import save,sha256,validate_handoff


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--kss-review',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    training=json.loads(args.training.read_text())
    evaluation=json.loads(args.evaluation.read_text())
    if training['status']!='PAIRED_SYNTHETIC_TRAINING_SMOKE_PASS' or evaluation['status']!='SYNTHETIC_SMOKE_EVALUATION_PASS':
        raise ValueError('This report is explicitly for completed synthetic smoke runs')
    args.output.mkdir(parents=True,exist_ok=False)
    experiment=json.loads((Path(training['experiment'])/'experiment.json').read_text())
    handoff=json.loads((Path(experiment['source_spec_run'])/'handoff_manifest.json').read_text())['utterances']
    validate_handoff(handoff)
    table=[];histories={}
    for row in training['training']:
        result=json.loads(Path(row['report']).read_text())
        if sha256(result['checkpoint'])!=result['checkpoint_sha256']:
            raise ValueError('Changed trained checkpoint')
        score=next(r for r in evaluation['measurements'] if r['condition']==row['condition'])
        table.append({'condition':row['condition'],'training_steps':result['completed_steps'],
            'seconds_including_checkpoint':result['seconds'],
            'peak_gpu_allocated_gib':result['peak_allocated_bytes']/1024**3,
            'peak_gpu_reserved_gib':result['peak_reserved_bytes']/1024**3,
            'parameter_probe_max_delta':result['parameter_probe_max_delta'],
            'final_generator_loss':result['losses'][-1]['generator'],
            'heldout_test_count':score['test_count'],'sim':score['mean_sim'],'cer':score['mean_cer'],
            'scope':'Synthetic 3-step smoke only; not KSS or defense efficacy'})
        histories[row['condition']]=result['losses']
    for filename,rows in [('training_evaluation.csv',table),
        ('spec_input_quality.csv',[{'utterance_id':r['utterance_id'],'split':r['split'],
          'iterations':r['iterations'],'stoi':r['stoi'],'input_kind':r['input_kind']} for r in handoff])]:
        with (args.output/filename).open('x',encoding='utf-8-sig',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Verify outputs directly against per-file evaluation records, not copied numbers.
    per_file=list(csv.DictReader((args.evaluation.parent/'per_file.csv').open(encoding='utf-8-sig')))
    for row in per_file:
        if sha256(row['output'])!=row['output_sha256'] or sha256(row['reference'])!=row['checkpoint_sha256']:
            raise ValueError('Evaluation WAV/checkpoint hash changed')
    fig,axes=plt.subplots(1,3,figsize=(13,4.3))
    for condition,losses in histories.items():
        axes[0].plot([r['step'] for r in losses],[r['generator'] for r in losses],marker='o',label=condition)
    axes[0].set(xlabel='Optimizer step',ylabel='Generator loss',title='Actual bounded training');axes[0].legend(fontsize=8)
    measurements=evaluation['measurements'];labels=[r['condition'] for r in measurements]
    axes[1].bar(labels,[r['mean_sim'] for r in measurements],color='#4084ad')
    axes[1].set(ylabel='ECAPA cosine SIM',ylim=(0,1),title='One held-out synthetic prompt')
    axes[2].bar(labels,[100*r['mean_cer'] for r in measurements],color='#b37439')
    axes[2].set(ylabel='Korean CER (%)',title='Whisper small, language=ko')
    for ax in axes[1:]:
        ax.tick_params(axis='x',rotation=25)
    fig.suptitle('Synthetic training smoke only: 3 steps/condition; no KSS or protection-efficacy claim',fontsize=11)
    fig.tight_layout();fig.savefig(args.output/'synthetic_smoke_results.png',dpi=180)
    fig.savefig(args.output/'synthetic_smoke_results.svg');plt.close(fig)
    save(args.output/'report.json',{'scope':'synthetic_training_smoke','status':'ARTIFACTS_VERIFIED',
        'training_summary':str(args.training.resolve()),'evaluation':str(args.evaluation.resolve()),
        'kss_review':json.loads((args.kss_review/'review_status.json').read_text()),
        'training_evaluation':table,'test_count_per_condition':1,
        'limitations':['No human KSS training','No listening review','No statistical efficacy conclusion',
          'No bitwise reproducibility claim across GPU/platforms','Random discriminator initialization is identical by seed across conditions']})
    print(json.dumps({'directory':str(args.output.resolve()),'status':'ARTIFACTS_VERIFIED'}))


if __name__=='__main__':
    main()
