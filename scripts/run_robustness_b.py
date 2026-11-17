"""Transform a real aligned original/SPEC pair under identical conditions; no metric claims."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
from tempfile import mkdtemp

import numpy as np
from scipy.io import wavfile
import soundfile as sf

from robustness_b import transform_audio, validate_audio, validate_condition
from audio_metrics import load_audio

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run_pair(manifest_path, output_root):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or set(manifest) != {'schema_version', 'experiment_id', 'audio', 'conditions'}:
        raise ValueError('manifest requires schema_version, experiment_id, audio, conditions only')
    if type(manifest['schema_version']) is not int or manifest['schema_version'] != 1:
        raise ValueError('unsupported schema_version')
    if not isinstance(manifest['experiment_id'], str) or not manifest['experiment_id'].strip():
        raise ValueError('experiment_id must be nonempty')
    if not isinstance(manifest['audio'], dict) or set(manifest['audio']) != {'original', 'protected'}:
        raise ValueError('audio requires original and protected paths')
    paths = {}
    for role, value in manifest['audio'].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError('audio paths must be nonempty strings')
        paths[role] = (manifest_path.parent / value).resolve(strict=True)
    if paths['original'] == paths['protected']:
        raise ValueError('original and protected must be different files')
    conditions = manifest['conditions']
    if not isinstance(conditions, list) or not conditions:
        raise ValueError('conditions must be a nonempty list')
    for condition in conditions:
        validate_condition(condition)
    hashes = {role: sha256(path) for role, path in paths.items()}
    inputs = {}
    for role, path in paths.items():
        audio, rate = load_audio(path)
        inputs[role] = (validate_audio(audio, rate), rate)
    original, rate = inputs['original']
    protected, protected_rate = inputs['protected']
    if rate != protected_rate or original.shape != protected.shape:
        raise ValueError('pair must have matching rates/channels/length; no implicit alignment')
    if np.array_equal(original, protected):
        raise ValueError('identical waveforms are not a real SPEC pair')
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix='robustness_b_', dir=output_root))
    report = {
        'schema_version': 1, 'status': 'RUNNING', 'scope': 'paired_transforms_only',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'manifest': manifest,
        'manifest_path': str(manifest_path), 'python': platform.python_version(),
        'platform': platform.platform(), 'device': 'cpu',
        'versions': {name: version(name) for name in ['numpy', 'scipy', 'soundfile']},
        'inputs': {role: {'path': str(path), 'sha256_before': hashes[role]}
                   for role, path in paths.items()}, 'outputs': [],
        'not_evaluated': ['STOI', 'SIM', 'MCD', 'WER', 'protection efficacy', 'model synthesis'],
        'evaluation_note': 'Resampled outputs retain target Hz. Evaluator must explicitly align reference and candidate.',
    }
    report_path = output / 'report.json'
    try:
        for index, condition in enumerate(conditions):
            folder = output / f'{index:02d}_{condition["name"]}'
            folder.mkdir()
            for role, (audio, input_rate) in inputs.items():
                transformed, out_rate, metadata = transform_audio(audio, input_rate, condition)
                mp3 = metadata.pop('_encoded_mp3', None)
                artifact = folder / f'{role}.wav'
                wavfile.write(artifact, out_rate, transformed)
                saved, saved_rate = sf.read(artifact, dtype='float64')
                if saved_rate != out_rate or not np.array_equal(saved, transformed):
                    raise RuntimeError('saved WAV failed exact readback')
                entry = {'condition_index': index, 'role': role, 'path': str(artifact),
                         'sha256': sha256(artifact), **metadata}
                if mp3 is not None:
                    compressed = folder / f'{role}.mp3'
                    compressed.write_bytes(mp3)
                    entry['mp3'] = {'path': str(compressed), 'sha256': sha256(compressed)}
                report['outputs'].append(entry)
        for role, path in paths.items():
            after = sha256(path)
            report['inputs'][role]['sha256_after'] = after
            if after != hashes[role]:
                raise RuntimeError('source file changed during experiment')
        report['input_files_unchanged'] = True
        report['status'] = 'PASS'
    except Exception as error:
        report['status'] = 'FAIL'
        report['error'] = str(error)
        raise
    finally:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report_path, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs')
    args = parser.parse_args()
    path, report = run_pair(args.manifest, args.output_root)
    print(json.dumps({'status': report['status'], 'report': str(path), 'wav_count': len(report['outputs'])}, indent=2))


if __name__ == '__main__':
    main()
