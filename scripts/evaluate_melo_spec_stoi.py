"""Revalidate Melo artifacts, then reuse strict CPU STOI. No transforms/models."""
import argparse
import csv
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
from tempfile import mkdtemp

from audio_metrics import load_audio, score_pair
from validate_melo_spec_pair import sha256, validate


def evaluate_validation(path):
    path = Path(path).resolve()
    before = sha256(path)
    stored = json.loads(path.read_text(encoding='utf-8'))
    if stored.get('schema_version') != 1 or stored.get('model_kind') != 'MeloTTS-Korean':
        raise ValueError('Require Melo validator schema v1')
    expected_status = 'MOCK_PASS' if stored['input_kind'] == 'mock_contract' else 'PASS'
    if stored.get('status') != expected_status or stored.get('input_files_unchanged') is not True:
        raise ValueError('Require a successful matching artifact validation')
    for artifact, digest in stored['files_sha256'].items():
        if sha256(artifact) != digest:
            raise ValueError('Artifact changed since validation')
    current = validate(stored['generation_report'], stored['original_path'], stored['protected_path'],
                       expected_original_sha256=stored['expected_original_sha256'],
                       execution_id=stored['execution_id'], input_kind=stored['input_kind'],
                       record_index=stored['record_index'])
    if stored['files_sha256'] != current['files_sha256']:
        raise ValueError('Validation input bindings differ')
    clean, rate = load_audio(current['original_path'])
    candidate, candidate_rate = load_audio(current['protected_path'])
    identity = score_pair(clean, clean, rate, rate)['stoi']
    if abs(identity - 1) > 1e-6:
        raise ValueError('Identity STOI sanity check failed')
    scores = score_pair(clean, candidate, rate, candidate_rate)
    for artifact, digest in current['files_sha256'].items():
        if sha256(artifact) != digest:
            raise RuntimeError('An input changed during STOI scoring')
    if sha256(path) != before:
        raise RuntimeError('Validation JSON changed during STOI scoring')
    return {'status': current['status'],
            'scope': 'mock_input_contract_STOI_only' if current['input_kind'] == 'mock_contract' else 'melo_artifact_STOI_only',
            'validation_scope': current['scope'],
            'model_kind': current['model_kind'], 'input_kind': current['input_kind'],
            'execution_id': current['execution_id'], 'seed': current['seed'],
            'iterations': current['iterations'], 'model_sampling_rate': current['model_sampling_rate'],
            'epsilon': current['epsilon'], 'sample_rate': rate, 'samples': len(clean),
            'identity_stoi': identity, **scores,
            'original_sha256': current['files_sha256'][current['original_path']],
            'protected_sha256': current['files_sha256'][current['protected_path']],
            'generation_report_sha256': current['files_sha256'][current['generation_report']],
            'validation_path': str(path), 'validation_sha256': before,
            'input_files_unchanged': True, 'model_weights_verified': False,
            'transform_applied': None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parents[1] / 'runs')
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix='melo_stoi_', dir=args.output_root))
    report = {'status': 'FAIL', 'model_kind': 'MeloTTS-Korean', 'device': 'cpu',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'python': platform.python_version(), 'platform': platform.platform(),
              'versions': {n: version(n) for n in ('numpy', 'scipy', 'soundfile', 'pystoi')},
              'limitations': 'Artifact/STOI check only; mock scores are not performance results. No model inference or defense evaluation.'}
    try:
        report.update(evaluate_validation(args.validation))
        fields = ['status', 'model_kind', 'input_kind', 'execution_id', 'seed', 'iterations',
                  'sample_rate', 'model_sampling_rate', 'samples', 'epsilon', 'identity_stoi', 'stoi',
                  'max_abs_delta', 'original_sha256', 'protected_sha256', 'generation_report_sha256']
        with (output / 'comparison.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerow(report)
    except Exception as error:
        report.update(status='FAIL', error_type=type(error).__name__, error=str(error))
    with (output / 'report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': report['status'], 'report': str(output / 'report.json')}))
    return 1 if report['status'] == 'FAIL' else 0


if __name__ == '__main__':
    raise SystemExit(main())
