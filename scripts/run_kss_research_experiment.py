"""KSS scoped research: preserved unreviewed flags -> SPEC -> paired training -> evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from run_korean_melo_spec import ROOT,save,check_manifest


def run(command,log):
    with log.open('x',encoding='utf-8') as stream:
        proc=subprocess.run([sys.executable,*map(str,command)],stdout=stream,stderr=subprocess.STDOUT)
    if proc.returncode:raise RuntimeError(f'Command failed ({proc.returncode}); see {log}')
    return log.read_text(encoding='utf-8').splitlines()[-1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1000);args=p.parse_args()
    check_manifest(args.manifest,allow_licensed_research=True)
    folder=Path(tempfile.mkdtemp(prefix='kss_research_study_',dir=ROOT/'runs'))
    report={'status':'FAIL','scope':'kss_noncommercial_research','manifest':str(args.manifest.resolve()),
        'steps_per_condition':args.steps,'condition_order':['clean','SPEC_10','SPEC_1'],
        'manual_listening_verified':False,'separate_speaker_consent':'not_verified','external_publication':'not_approved_by_this_record'}
    try:
        record=json.loads(run([ROOT/'scripts/run_korean_melo_spec.py','--manifest',args.manifest,'--assets',args.assets,
            '--allow-licensed-research','--iterations','1','10','--device','cuda'],folder/'spec.log'))
        report['spec_run']=record['run'];save(folder/'spec_complete.json',{'spec_run':record['run']})
        experiment=run([ROOT/'scripts/prepare_paired_melo_experiment.py','--spec-run',record['run'],'--assets',args.assets],folder/'prepare.log')
        report['experiment']=experiment;save(folder/'inputs_complete.json',{'experiment':experiment})
        trained=json.loads(run([ROOT/'scripts/run_paired_melo_smoke.py','--experiment',experiment,'--assets',args.assets,
            '--steps',args.steps,'--checkpoint-every','250','--output-root','/home/safespeech/experiments/SafeSpeech-KSS',
            '--conditions','clean','SPEC_10','SPEC_1'],folder/'training.log'))
        report['training_summary']=str(Path(trained['directory'])/'summary.json')
        save(folder/'training_complete.json',{'training_summary':report['training_summary']})
        evaluated=json.loads(run([ROOT/'scripts/evaluate_paired_melo_smoke.py','--training-summary',report['training_summary'],
            '--assets',args.assets],folder/'evaluation.log'))
        report['evaluation']=str(Path(evaluated['directory'])/'evaluation.json')
        report['status']='KSS_RESEARCH_TRAINING_AND_EVALUATION_PASS'
    except Exception as error:
        report['error']=str(error)
        raise
    finally:
        save(folder/'study.json',report)
        print(json.dumps({'directory':str(folder),'status':report['status']}),flush=True)


if __name__=='__main__':main()
