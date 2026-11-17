"""Execute reviewed input -> features -> SPEC -> paired inputs -> bounded training.

Requires genuine reviewed/authorized input declarations before any model work.
Never edits the submitted input or accepts a bypass flag.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import torch

from melo_training_core import KoreanFrontend, training_config
from prepare_paired_melo_experiment import prepare
from run_korean_melo_spec import ROOT, check_manifest, execute, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--bert',type=Path,required=True)
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--checkpoint-every',type=int,default=500)
    p.add_argument('--training-output-root',type=Path,default=Path('/home/safespeech/experiments/SafeSpeech-spec-perturbation'))
    args=p.parse_args()
    rows=check_manifest(args.manifest)
    if any(r['input_kind']!='human_recording' for r in rows):
        raise ValueError('This workflow requires authorized human recordings')
    if {r['split'] for r in rows}!={'train','validation','test'}:
        raise ValueError('Provide reviewed inputs covering fixed train/validation/test splits')
    folder=Path(tempfile.mkdtemp(prefix='authorized_korean_training_',dir=ROOT/'runs'))
    frontend=None
    enriched=[]
    for row in rows:
        current=copy.deepcopy(row)
        target=folder/(row['utterance_id']+'.features.pt')
        if row.get('features_path') and row.get('features_sha256'):
            from run_korean_melo_spec import sha256
            if sha256(row['features_path'])!=row['features_sha256']:
                raise ValueError('Prepared Korean feature hash mismatch')
            import shutil
            shutil.copyfile(row['features_path'],target)
            if torch.load(target,map_location='cpu',weights_only=True)['text']!=row['text']:
                raise ValueError('Prepared feature transcript mismatch')
        else:
            if frontend is None:
                frontend=KoreanFrontend(args.bert,training_config(args.assets))
            torch.save(frontend.features(row['text']),target)
        current['features_path']=str(target)
        enriched.append(current)
    save(folder/'input.json',{'schema_version':1,'utterances':enriched})
    spec=execute(folder/'input.json',args.assets,device='cuda',iterations=(1,10))
    experiment=prepare(spec,args.assets)
    command=[sys.executable,str(ROOT/'scripts/run_paired_melo_smoke.py'),
        '--experiment',str(experiment),'--assets',str(args.assets),'--steps',str(args.steps),
        '--checkpoint-every',str(args.checkpoint_every),'--output-root',str(args.training_output_root)]
    with (folder/'training.log').open('x') as stream:
        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    final=json.loads((folder/'training.log').read_text().splitlines()[-1])
    training_summary=Path(final['directory'])/'summary.json'
    evaluation_command=[sys.executable,str(ROOT/'scripts/evaluate_paired_melo_smoke.py'),
        '--training-summary',str(training_summary),'--assets',str(args.assets)]
    with (folder/'evaluation.log').open('x') as stream:
        subprocess.run(evaluation_command,stdout=stream,stderr=subprocess.STDOUT,check=True)
    evaluated=json.loads((folder/'evaluation.log').read_text().splitlines()[-1])
    save(folder/'workflow.json',{'status':'BOUNDED_TRAINING_AND_EVALUATION_COMPLETED',
        'spec_run':str(spec),'experiment':str(experiment),'training_log':str(folder/'training.log'),
        'training_summary':str(training_summary),'evaluation':str(Path(evaluated['directory'])/'evaluation.json')})
    print(str(folder))


if __name__=='__main__':
    main()
