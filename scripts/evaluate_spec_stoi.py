"""Score actual, validated SPEC WAV pairs on CPU; no transforms or alignment."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from tempfile import mkdtemp

from audio_metrics import load_audio, score_pair

ROOT = Path(__file__).resolve().parents[1]
METRICS_REVISION = '27ae06d1fe304320c671eeeb15338e8c23b36ac6'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def evaluate_pair(manifest_path, validation_path):
    manifest_path, validation_path = Path(manifest_path).resolve(), Path(validation_path).resolve()
    manifest, validation = read_json(manifest_path), read_json(validation_path)
    if (type(manifest.get('schema_version')) is not int or manifest['schema_version'] != 1
            or manifest.get('evaluation') != dict(reference_role='original',
                                                  target_role='protected', alignment='strict')):
        raise ValueError('Require manifest v1 with strict original -> protected roles')
    audio = manifest.get('audio')
    if not isinstance(audio, dict) or set(audio) != {'original', 'protected', 'synthetic'} or audio['synthetic'] is not None:
        raise ValueError('Require original/protected paths and synthetic=null')
    if any(not isinstance(audio[r], str) or not audio[r].strip() for r in ('original', 'protected')):
        raise ValueError('Missing original/protected path')
    original, protected = ((manifest_path.parent / audio[r]).resolve() for r in ('original', 'protected'))
    if original == protected or original.parent != protected.parent:
        raise ValueError('Require distinct WAVs from the same SPEC run directory')
    run = original.parent
    report_path = run / 'report.json'
    generation = read_json(report_path)
    if (generation.get('status') != 'PASS' or generation.get('mode') != 'SPEC'
            or type(generation.get('iterations')) is not int or generation['iterations'] < 1):
        raise ValueError('Not a successful actual SPEC generation report')
    if (validation.get('status') != 'PASS' or Path(validation['run_dir']).resolve() != run
            or validation.get('iterations') != generation['iterations']
            or validation.get('source_copy_identical') is not True
            or validation.get('saved_noise_reconstructs_wav') is not True):
        raise ValueError('Missing matching successful SPEC artifact validation')
    # Bind the current WAVs, report and noise to the completed torch-based check.
    expected = validation['files_sha256']
    for path in (original, protected, report_path, run / 'perturbation.pt'):
        if str(path) not in expected:
            raise ValueError(f'Artifact not covered by validation: {path.name}')
    watched = {Path(p): digest for p, digest in expected.items()}
    watched[manifest_path], watched[validation_path] = sha256(manifest_path), sha256(validation_path)
    for path, digest in watched.items():
        if sha256(path) != digest:
            raise ValueError(f'Artifact changed since validation: {path}')
    reference, rate = load_audio(original)
    candidate, candidate_rate = load_audio(protected)
    identity = score_pair(reference, reference, rate, rate)['stoi']
    if abs(identity - 1) > 1e-6:
        raise ValueError('Identity STOI sanity check failed')
    scores = score_pair(reference, candidate, rate, candidate_rate)
    if not 0 < scores['max_abs_delta'] <= float(generation['epsilon']) + 1e-7:
        raise ValueError('Actual WAV change is zero or exceeds generation epsilon')
    for path, digest in watched.items():
        if sha256(path) != digest:
            raise RuntimeError('An input changed during scoring')
    return dict(iterations=generation['iterations'], candidate_kind='actual_SPEC_protected',
                manifest_path=str(manifest_path), artifact_validation_path=str(validation_path),
                original_path=str(original), protected_path=str(protected),
                original_sha256=sha256(original), protected_sha256=sha256(protected),
                generation_report_sha256=sha256(report_path),
                sample_rate=rate, samples=len(reference), duration_seconds=len(reference) / rate,
                identity_stoi=identity, **scores, input_files_unchanged=True,
                transform_applied=None, declared_transform_not_executed=manifest.get('transform'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifests', nargs='+', required=True, type=Path)
    parser.add_argument('--validations', nargs='+', required=True, type=Path)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs')
    args = parser.parse_args()
    if len(args.manifests) != len(args.validations):
        parser.error('Provide one validation JSON per manifest, in the same order')
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix='spec_stoi_', dir=args.output_root))
    report = dict(status='FAIL', scope='actual_SPEC_pair_STOI_only', device='cpu',
                  created_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
                  executable=sys.executable, python=platform.python_version(), platform=platform.platform(),
                  versions={n: version(n) for n in ('numpy', 'scipy', 'soundfile', 'pystoi')},
                  metrics_source_revision=METRICS_REVISION, pairs=[],
                  limitations='STOI measures intelligibility; no voice-cloning defense evaluation.')
    try:
        for manifest, validation in zip(args.manifests, args.validations):
            report['pairs'].append(evaluate_pair(manifest, validation))
        iterations = [p['iterations'] for p in report['pairs']]
        if len(iterations) != len(set(iterations)):
            raise ValueError('Duplicate iteration counts cannot form this comparison')
        if len({p['original_sha256'] for p in report['pairs']}) != 1:
            raise ValueError('Iteration comparison requires the same original WAV bytes')
        if set(iterations) == {1, 10}:
            values = {p['iterations']: p['stoi'] for p in report['pairs']}
            report['stoi_10_minus_1'] = values[10] - values[1]
        with (output / 'comparison.csv').open('x', encoding='utf-8', newline='') as stream:
            fields = ['iterations', 'candidate_kind', 'sample_rate', 'samples', 'stoi',
                      'identity_stoi', 'max_abs_delta', 'original_sha256', 'protected_sha256']
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(report['pairs'])
        report['status'] = 'PASS'
    except Exception as error:
        report.update(status='FAIL', error_type=type(error).__name__, error=str(error))
    with (output / 'report.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(report_path=str(output / 'report.json'), **report), ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
