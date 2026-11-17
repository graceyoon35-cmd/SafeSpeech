"""Measure one real checkpoint save on native WSL storage, preserve all inputs."""
import argparse
import json
from pathlib import Path
from train_melo_bounded import train
from run_korean_melo_spec import ROOT,save

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--benchmark-summary',type=Path,required=True)
p.add_argument('--assets',type=Path,required=True)
args=p.parse_args()
summary=json.loads(args.benchmark_summary.read_text())
phase=next(r for r in summary['phases'] if r['condition']=='clean' and r['target_step']==50)
old=json.loads(Path(phase['report']).read_text())
config=Path(phase['report']).parent/'config.json'
native=train(config,args.assets,52,resume=Path(old['checkpoint_directory']),checkpoint_every=50,
             output_root=Path('/home/safespeech/experiments/SafeSpeech-spec-perturbation'))
result=json.loads((native/'training_report.json').read_text())
save(ROOT/'runs/benchmark_20261118/native_storage_comparison.json',{
    'scope':'Storage comparison only; 2 resumed steps, not the 100-step evaluation checkpoints',
    'windows_checkpoint_save_seconds':old['checkpoints'][-1]['save_seconds'],
    'native_checkpoint_save_seconds':result['checkpoints'][-1]['save_seconds'],
    'native_training_report':str(native/'training_report.json'),
    'native_checkpoint':result['checkpoint'],'status':result['status']})
