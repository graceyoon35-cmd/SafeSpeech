"""Read-only progress for the current study; never infer completion from a WAV."""
import argparse
import json
from pathlib import Path


def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))


def records(path):
    if not path.exists():return []
    result=[]
    for line in path.read_text(encoding='utf-8',errors='replace').splitlines():
        try:
            item=json.loads(line)
            if isinstance(item,dict):result.append(item)
        except json.JSONDecodeError:pass
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);args=p.parse_args()
    study=args.study.resolve();runs=study.parent;status={'study':str(study),'phase':'SPEC'}
    spec=[r for r in records(study/'spec.log') if 'progress' in r]
    if spec:status.update(spec_pairs=spec[-1]['progress'],spec_pairs_total=spec[-1]['total'])
    if (study/'spec_complete.json').exists():
        status['phase']='PREPARING_MATCHED_INPUTS';status['spec_run']=read(study/'spec_complete.json')['spec_run']
    if (study/'inputs_complete.json').exists():
        experiment=read(study/'inputs_complete.json')['experiment'];status['phase']='TRAINING'
        for plan_path in runs.glob('paired_melo_training_*/plan.json'):
            plan=read(plan_path)
            if plan['experiment']!=experiment:continue
            progress=[]
            for condition in plan['conditions']:
                items=records(plan_path.parent/(condition+'.log'));steps=[r for r in items if 'step' in r]
                row={'condition':condition,'step':steps[-1]['step'] if steps else 0,'target':plan['steps_per_condition']}
                row['latest_checkpoint_step']=max((r['checkpoint_step'] for r in items if 'checkpoint_step' in r),default=0)
                if steps:row.update(loss=steps[-1].get('generator'),seconds_per_latest_step=steps[-1].get('step_seconds_excluding_checkpoint'))
                completed=[r for r in items if r.get('status')=='TRAINING_SMOKE_PASS']
                row['completed']=bool(completed);progress.append(row)
            status['training']=progress
    if (study/'training_complete.json').exists():
        status['phase']='EVALUATION';summary=read(study/'training_complete.json')['training_summary']
        for plan_path in runs.glob('melo_smoke_evaluation_*/plan.json'):
            plan=read(plan_path)
            if plan['training_summary']!=summary:continue
            status['evaluation_wavs_generated']=len(list(plan_path.parent.glob('*/*.wav')))
            status['evaluation_wavs_target']=plan['test_utterances']*len(plan['conditions'])
        status['evaluation_scores_logged']=sum('sim' in r and 'cer' in r for r in records(study/'evaluation.log'))
    if (study/'study.json').exists():status['final']=read(study/'study.json')
    print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
