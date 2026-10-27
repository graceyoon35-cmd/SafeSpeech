"""Run a strict aligned audio-pair → Gaussian transform → STOI CPU experiment."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from tempfile import mkdtemp

from scipy.io import wavfile

from audio_metrics import load_audio, score_pair
from robustness_transforms import gaussian_noise

ROOT = Path(__file__).resolve().parents[1]
ROLES = {'original', 'protected', 'synthetic'}


def require_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f'{label} must contain exactly {sorted(keys)}')


def read_manifest(path):
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    require_keys(manifest, {'schema_version', 'experiment_id', 'audio', 'evaluation', 'transform'}, 'manifest')
    if type(manifest['schema_version']) is not int or manifest['schema_version'] != 1:
        raise ValueError('Unsupported schema_version')
    if not isinstance(manifest['experiment_id'], str) or not manifest['experiment_id'].strip():
        raise ValueError('experiment_id must be a nonempty string')
    require_keys(manifest['audio'], ROLES, 'audio')
    resolved = {}
    for role, value in manifest['audio'].items():
        if value is None:
            resolved[role] = None
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f'{role} must be a path string or null')
        else:
            resolved[role] = (path.parent / value).resolve()
            if not resolved[role].is_file():
                raise ValueError(f'Missing {role} file: {resolved[role]}')
    if resolved['original'] is None:
        raise ValueError('original audio is required')
    evaluation = manifest['evaluation']
    require_keys(evaluation, {'reference_role', 'target_role', 'alignment'}, 'evaluation')
    for key in ['reference_role', 'target_role']:
        role = evaluation[key]
        if not isinstance(role, str) or role not in ROLES or resolved[role] is None:
            raise ValueError(f'{key} must select an available audio role')
    if evaluation['alignment'] != 'strict':
        raise ValueError('Only strict alignment is supported')
    transform = manifest['transform']
    require_keys(transform, {'name', 'seed', 'std'}, 'transform')
    if transform['name'] != 'gaussian_noise':
        raise ValueError('Only gaussian_noise is supported')
    # Validate parameters even before loading the audio.
    gaussian_noise([0.0], seed=transform['seed'], std=transform['std'])
    return manifest, resolved


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run_experiment(manifest_path, output_root=ROOT / 'runs'):
    manifest, paths = read_manifest(manifest_path)
    before_hashes = {role: sha256(path) for role, path in paths.items() if path is not None}
    # All declared audio inputs must satisfy the common audio contract.
    audio = {role: load_audio(path) for role, path in paths.items() if path is not None}
    evaluation = manifest['evaluation']
    reference, reference_rate = audio[evaluation['reference_role']]
    target, rate = audio[evaluation['target_role']]
    before = score_pair(reference, target, reference_rate, rate)
    transform = manifest['transform']
    transformed = gaussian_noise(target, seed=transform['seed'], std=transform['std'])
    after = score_pair(reference, transformed, reference_rate, rate)
    after_hashes = {role: sha256(path) for role, path in paths.items() if path is not None}
    if before_hashes != after_hashes:
        raise RuntimeError('An input file changed during the experiment')
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix='experiment_', dir=output_root))
    # DOUBLE keeps scoring input samples unchanged when the artifact is reloaded.
    artifact = output / 'transformed.wav'
    wavfile.write(artifact, rate, transformed)
    report = {
        'schema_version': 1, 'status': 'PASS', 'scope': 'common_pipeline_smoke_only',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'manifest': manifest, 'manifest_path': str(Path(manifest_path).resolve()),
        'device': 'cpu', 'python': platform.python_version(), 'platform': platform.platform(),
        'versions': {name: version(name) for name in ['numpy', 'scipy', 'soundfile', 'pystoi']},
        'inputs': {role: {'path': str(paths[role]), 'sha256_before': before_hashes[role],
                          'sha256_after': after_hashes[role]} for role in before_hashes},
        'sample_rate': rate, 'samples': len(target), 'input_files_unchanged': True,
        'metrics': {'before': before, 'after': after, 'stoi_delta': after['stoi'] - before['stoi']},
        'artifact': {'path': str(artifact.resolve()), 'sha256': sha256(artifact)},
        'not_evaluated': ['SIM', 'MCD', 'WER', 'protection efficacy', 'model training'],
    }
    report_path = output / 'report.json'
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report_path, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs')
    args = parser.parse_args()
    try:
        path, report = run_experiment(args.manifest, args.output_root)
    except (ValueError, OSError, RuntimeError, Warning) as error:
        parser.exit(1, f'Experiment failed: {error}\n')
    print(json.dumps({'report': str(path), 'status': report['status'], 'metrics': report['metrics']}, indent=2))


if __name__ == '__main__':
    main()
