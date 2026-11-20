"""Train a separate strict-PCM16 KSS condition; reuse the completed clean model."""
import argparse
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys

from audit_kss_final import digest, load, write
from run_korean_melo_spec import ROOT


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    original = load(ROOT/'runs/paired_melo_training_rhp4a8t2/summary.json')
    source = next(r for r in original['training'] if r['condition'] == 'SPEC_10')
    cfg = load(Path(source['run'])/'config.json')
    original_cfg = copy.deepcopy(cfg)
    dataset = load(cfg['experiment']['dataset_manifest'])
    corrected = ROOT/'runs/final_validation_20261119/strict_corrected_inputs'
    with (corrected/'quality.csv').open(encoding='utf-8-sig') as stream:
        quality = {r['utterance_id']: r for r in csv.DictReader(stream)}
    provenance = []
    for row in dataset['utterances']:
        q = quality[row['utterance_id']]
        target = Path(q['output'])
        if digest(target) != q['sha256'] or digest(row['wav']) != row['wav_sha256']:
            raise ValueError('Source or corrected WAV hash mismatch')
        if digest(row['features']) != row['features_sha256']:
            raise ValueError('Text feature hash mismatch')
        provenance.append({'utterance_id': row['utterance_id'], 'split': row['split'],
                           'old_wav': row['wav'], 'old_sha256': row['wav_sha256'],
                           'new_wav': str(target), 'new_sha256': q['sha256']})
        row.update(wav=str(target), wav_sha256=q['sha256'])
    write(out/'dataset.json', dataset)
    # The bounded trainer consumes dataset.json directly. Keep metadata lists in
    # agreement too, so downstream tools cannot accidentally use historical WAVs.
    replacements = {r['old_wav']: r['new_wav'] for r in provenance}
    for key in ('training_files', 'validation_files'):
        content = Path(cfg['data'][key]).read_text()
        for old, new in replacements.items():
            content = content.replace(old, new)
        path = out/(key+'.list.cleaned')
        with path.open('x', encoding='utf-8') as stream:
            stream.write(content)
        cfg['data'][key] = str(path)
    cfg['experiment']['dataset_manifest'] = str(out/'dataset.json')
    cfg['experiment']['condition'] = 'SPEC_10_strict'
    write(out/'config.json', cfg)
    write(out/'input_provenance.json', {'training_seed': 52, 'steps': 1000,
          'source_config': original_cfg, 'actual_config': cfg, 'files': provenance,
          'comparison_limit': 'CUDA training is not bitwise deterministic; one corrected training seed only.'})
    command = [sys.executable, str(ROOT/'scripts/train_melo_bounded.py'),
               '--config', str(out/'config.json'), '--assets', str(args.assets),
               '--steps', '1000', '--checkpoint-every', '1000',
               '--output-root', str(args.checkpoint_root)]
    write(out/'command.json', {'training': command})
    with (out/'training.log').open('x') as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    final = json.loads((out/'training.log').read_text().splitlines()[-1])
    folder = Path(final['run'])
    report = load(folder/'training_report.json')
    if report['status'] != 'TRAINING_SMOKE_PASS' or report['completed_steps'] != 1000:
        raise ValueError('Incomplete strict training')
    if report['initial_checkpoint_sha256'] != original['initial_checkpoint_sha256']:
        raise ValueError('Wrong initialization')
    if digest(report['checkpoint']) != report['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    summary = copy.deepcopy(original)
    summary['training'] = [next(r for r in original['training'] if r['condition'] == 'clean'),
        {'condition': 'SPEC_10_strict', 'run': str(folder),
         'checkpoint': report['checkpoint'], 'report': str(folder/'training_report.json')}]
    summary['clean_training_reused'] = True
    summary['strict_input_provenance'] = str(out/'input_provenance.json')
    write(out/'summary.json', summary)
    with (out/'evaluation.log').open('x') as stream:
        subprocess.run([sys.executable, str(ROOT/'scripts/evaluate_paired_melo_smoke.py'),
            '--training-summary', str(out/'summary.json'), '--assets', str(args.assets)],
            stdout=stream, stderr=subprocess.STDOUT, check=True)
    location = json.loads((out/'evaluation.log').read_text().splitlines()[-1])
    write(out/'evaluation_location.json', location)
    print(json.dumps({'status': 'PASS', 'output': str(out), 'evaluation': location}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint-root', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    run(p.parse_args())
