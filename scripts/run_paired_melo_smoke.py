"""Run matching clean/SPEC_1/SPEC_10 bounded training in separate GPU processes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from run_korean_melo_spec import ROOT, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--checkpoint-every',type=int,default=50)
    p.add_argument('--output-root',type=Path)
    p.add_argument('--conditions',nargs='+')
    args=p.parse_args()
    experiment=json.loads((args.experiment/'experiment.json').read_text())
    if args.conditions:
        by_condition={r['condition']:r for r in experiment['conditions']}
        if len(args.conditions)!=len(set(args.conditions)) or set(args.conditions)!=set(by_condition):
            raise ValueError('Requested order must include each prepared condition exactly once')
        experiment['conditions']=[by_condition[c] for c in args.conditions]
    configs=[json.loads(Path(row['config']).read_text()) for row in experiment['conditions']]
    baseline=configs[0]
    signatures=[]
    for cfg in configs:
        if cfg['train']!=baseline['train'] or cfg['model']!=baseline['model']:
            raise ValueError('Paired training hyperparameters differ')
        rows=json.loads(Path(cfg['experiment']['dataset_manifest']).read_text())['utterances']
        signatures.append([(r['utterance_id'],r['split'],r['text'],r['features_sha256']) for r in rows])
    if any(sig!=signatures[0] for sig in signatures):
        raise ValueError('Paired IDs/splits/transcripts/features differ')
    folder=Path(tempfile.mkdtemp(prefix='paired_melo_training_',dir=ROOT/'runs'))
    save(folder/'plan.json',{'experiment':str(args.experiment.resolve()),'steps_per_condition':args.steps,
        'conditions':[r['condition'] for r in experiment['conditions']],'scope':experiment['scope']})
    report={'status':'FAIL','scope':experiment['scope'],'steps_per_condition':args.steps,
            'experiment':str(args.experiment.resolve()),'training':[]}
    try:
        for condition in experiment['conditions']:
            log=folder/(condition['condition']+'.log')
            command=[sys.executable,str(ROOT/'scripts/train_melo_bounded.py'),
                '--config',condition['config'],'--assets',str(args.assets),'--steps',str(args.steps),
                '--checkpoint-every',str(args.checkpoint_every)]
            if args.output_root:
                command+=['--output-root',str(args.output_root)]
            with log.open('x') as stream:
                proc=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
            if proc.returncode:
                raise RuntimeError(f"Training failed: {condition['condition']}; see {log}")
            final=json.loads(log.read_text().splitlines()[-1])
            result=json.loads((Path(final['run'])/'training_report.json').read_text())
            if result['status']!='TRAINING_SMOKE_PASS' or result['completed_steps']!=args.steps:
                raise ValueError('Training did not complete requested steps')
            report['training'].append({'condition':condition['condition'],'run':final['run'],
                'checkpoint':result['checkpoint'],'report':str(Path(final['run'])/'training_report.json')})
            print(json.dumps(report['training'][-1]),flush=True)
        initial_hashes={json.loads(Path(r['report']).read_text())['initial_checkpoint_sha256'] for r in report['training']}
        if len(initial_hashes)!=1:
            raise ValueError('Different pretrained initialization')
        report.update(status='PAIRED_SYNTHETIC_TRAINING_SMOKE_PASS' if experiment['scope']=='synthetic_training_smoke'
                      else 'PAIRED_BOUNDED_TRAINING_PASS',initial_checkpoint_sha256=initial_hashes.pop())
    except Exception as error:
        report['error']=str(error)
        raise
    finally:
        save(folder/'summary.json',report)
        print(json.dumps({'directory':str(folder),'status':report['status']}),flush=True)


if __name__=='__main__':
    main()
