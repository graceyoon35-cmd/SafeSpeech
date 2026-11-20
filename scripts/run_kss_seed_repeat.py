"""Repeat frozen KSS inputs at a new training seed, preserving every old artifact."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from run_korean_melo_spec import ROOT, save, sha256


def repeat(summary, assets, output_root, seed, steps=1000):
    original = json.loads(Path(summary).read_text())
    if original['scope'] != 'kss_noncommercial_research' or original['status'] != 'PAIRED_BOUNDED_TRAINING_PASS':
        raise ValueError('Require completed KSS paired study')
    by_condition = {r['condition']:r for r in original['training']}
    folder = Path(tempfile.mkdtemp(prefix=f'kss_seed_{seed}_', dir=ROOT/'runs'))
    report = {'status':'FAIL', 'scope':original['scope'], 'experiment':original['experiment'],
              'steps_per_condition':steps, 'training':[], 'training_seed':seed,
              'source_summary':str(Path(summary).resolve()), 'source_summary_sha256':sha256(summary),
              'perturbation_seed':52, 'perturbations_reused':True,
              'replication_scope':'Training seed changes only; fixed SPEC waveforms and split, same inference seeds.'}
    save(folder/'plan.json', report)
    try:
        for condition in ('clean','SPEC_10'):
            row = by_condition[condition]
            cfg = json.loads((Path(row['run'])/'config.json').read_text())
            if cfg['train']['seed'] == seed:
                raise ValueError('Choose a seed different from original')
            cfg = copy.deepcopy(cfg)
            cfg['train']['seed'] = seed
            config = folder/(condition+'_config.json')
            save(config, cfg)
            command = [sys.executable,str(ROOT/'scripts/train_melo_bounded.py'),
                       '--config',str(config),'--assets',str(assets),'--steps',str(steps),
                       '--checkpoint-every','1000','--output-root',str(output_root)]
            with (folder/(condition+'.log')).open('x') as stream:
                proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            if proc.returncode:
                raise RuntimeError(f'{condition} training failed with exit {proc.returncode}; see {folder}')
            final = json.loads((folder/(condition+'.log')).read_text().splitlines()[-1])
            run = Path(final['run'])
            result = json.loads((run/'training_report.json').read_text())
            if result['status'] != 'TRAINING_SMOKE_PASS' or result['completed_steps'] != steps:
                raise ValueError('Incomplete seed repeat')
            report['training'].append({'condition':condition, 'run':str(run),
                'checkpoint':result['checkpoint'], 'report':str(run/'training_report.json')})
            print(json.dumps({'completed_condition':condition,'directory':str(folder)}),flush=True)
        hashes = {json.loads(Path(r['report']).read_text())['initial_checkpoint_sha256'] for r in report['training']}
        if hashes != {original['initial_checkpoint_sha256']}:
            raise ValueError('Initialization mismatch')
        report.update(status='PAIRED_BOUNDED_TRAINING_PASS', initial_checkpoint_sha256=hashes.pop())
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        save(folder/'summary.json', report)
        print(json.dumps({'directory':str(folder),'status':report['status']}),flush=True)
    return folder


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-summary',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--seed',type=int,default=53)
    p.add_argument('--steps',type=int,default=1000)
    args = p.parse_args()
    if not 0 <= args.seed < 2**32:
        p.error('seed must be uint32')
    repeat(args.training_summary,args.assets,args.output_root,args.seed,args.steps)
