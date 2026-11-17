"""50 -> 100 steps per condition, including controlled interruption/recovery."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from run_korean_melo_spec import ROOT,save,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    args=p.parse_args()
    experiment=json.loads((args.experiment/'experiment.json').read_text())
    if experiment['scope']!='synthetic_training_smoke':
        raise ValueError('Timing benchmark accepts authorized synthetic tests only')
    configs=[json.loads(Path(row['config']).read_text()) for row in experiment['conditions']]
    identities=[]
    for cfg in configs:
        if cfg['train']!=configs[0]['train'] or cfg['model']!=configs[0]['model']:
            raise ValueError('Unequal paired settings')
        data=json.loads(Path(cfg['experiment']['dataset_manifest']).read_text())['utterances']
        identities.append([(r['utterance_id'],r['split'],r['text'],r['features_sha256']) for r in data])
    if any(ids!=identities[0] for ids in identities):
        raise ValueError('Paired data identities differ')
    folder=Path(tempfile.mkdtemp(prefix='melo_timing_100_',dir=ROOT/'runs'))
    summary={'status':'FAIL','scope':'synthetic_training_smoke','steps_per_condition':100,
        'experiment':str(args.experiment.resolve()),'training':[],'phases':[],
        'controlled_interruption_condition':'clean','initial_checkpoint_sha256':sha256(configs[0]['experiment']['generator_initialization'])}
    try:
        for condition in experiment['conditions']:
            previous=None
            for steps in (50,100):
                interrupted=condition['condition']=='clean' and steps==50
                command=[sys.executable,str(ROOT/'scripts/train_melo_bounded.py'),
                    '--config',condition['config'],'--assets',str(args.assets),
                    '--steps',str(steps),'--checkpoint-every','50']
                if previous:
                    command+=['--resume',previous['checkpoint_directory']]
                if interrupted:
                    command+=['--interrupt-after-checkpoint','50']
                log=folder/f"{condition['condition']}_{steps}.log"
                with log.open('x') as stream:
                    process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
                records=[]
                for line in log.read_text().splitlines():
                    try:
                        value=json.loads(line)
                        if isinstance(value,dict) and 'run' in value:
                            records.append(value)
                    except json.JSONDecodeError:
                        pass
                if not records:
                    raise RuntimeError(f'No training report: {log}')
                run=Path(records[-1]['run'])
                report=json.loads((run/'training_report.json').read_text())
                expected='INTERRUPTED_TEST_AFTER_COMMITTED_CHECKPOINT' if interrupted else 'TRAINING_SMOKE_PASS'
                if report['status']!=expected or (process.returncode==0)==interrupted or report['completed_steps']!=steps:
                    raise RuntimeError(f'Unexpected training outcome: {log}')
                if steps==100 and (report.get('restored_optimizer_steps')!=[[50],[50],[50]] or not report.get('resume_integrity_verified')):
                    raise ValueError('Optimizer/checkpoint restoration not verified')
                summary['phases'].append({'condition':condition['condition'],'target_step':steps,
                    'report':str(run/'training_report.json'),'process_returncode':process.returncode})
                previous=report
                print(json.dumps({'condition':condition['condition'],'step':steps,'status':report['status'],
                                  'mean_step_seconds':report['step_timing']['mean_seconds']}),flush=True)
            summary['training'].append({'condition':condition['condition'],'run':str(run),
                'checkpoint':previous['checkpoint'],'report':str(run/'training_report.json')})
        summary['status']='PAIRED_SYNTHETIC_TRAINING_SMOKE_PASS'
        summary['resume_test']='PASS: committed step 50 -> fresh process -> optimizer/RNG restore -> step 100'
    except Exception as error:
        summary['error']=str(error)
        raise
    finally:
        save(folder/'summary.json',summary)
        print(json.dumps({'directory':str(folder),'status':summary['status']}),flush=True)


if __name__=='__main__':
    main()
