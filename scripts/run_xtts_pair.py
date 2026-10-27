"""Compare clean and SPEC references using identical KO/EN prompts and settings."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
from tempfile import mkdtemp

from audio_metrics import load_audio
from run_experiment import require_keys, sha256, ROOT
from xtts_inference import XttsSynthesizer, validate_request


def read_pair(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    require_keys(data, {'schema_version', 'model_dir', 'device', 'consent_confirmed',
                        'cpml_accepted', 'references', 'seed', 'prompts'}, 'XTTS manifest')
    if type(data['schema_version']) is not int or data['schema_version'] != 1:
        raise ValueError('Unsupported schema_version')
    if data['consent_confirmed'] is not True or data['cpml_accepted'] is not True:
        raise ValueError('Reference permission and CPML acceptance must be confirmed')
    if data['device'] not in ('cpu', 'cuda:0'):
        raise ValueError('device must be cpu or cuda:0')
    if not isinstance(data['model_dir'], str) or not data['model_dir'].strip():
        raise ValueError('model_dir must be a nonempty path')
    if type(data['seed']) is not int or not 0 <= data['seed'] < 2**32:
        raise ValueError('Invalid seed')
    require_keys(data['references'], {'clean', 'protected'}, 'references')
    references = {}
    for role, value in data['references'].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f'{role} reference must be a nonempty path')
        references[role] = (path.parent / value).resolve()
    if references['clean'] == references['protected']:
        raise ValueError('Clean and protected references must be different files')
    if not isinstance(data['prompts'], list) or not data['prompts']:
        raise ValueError('At least one prompt is required')
    for index, prompt in enumerate(data['prompts']):
        require_keys(prompt, {'language', 'text'}, 'prompt')
        for ref in references.values():
            validate_request(ref, prompt['text'], prompt['language'],
                             path.parent / f'.xtts_validation_{index}.wav')
    clean, rate = load_audio(references['clean'])
    protected, protected_rate = load_audio(references['protected'])
    if clean.shape != protected.shape or rate != protected_rate:
        raise ValueError('Clean/SPEC reference pairs must have the same length and sample rate')
    return data, references, (path.parent / data['model_dir']).resolve()


def run_pair(path, output_root=ROOT / 'runs', *, backend=None):
    data, refs, model_dir = read_pair(path)
    before = {role: sha256(ref) for role, ref in refs.items()}
    synth = XttsSynthesizer(model_dir, data['device'], consent_confirmed=True,
                            cpml_accepted=True, backend=backend)
    Path(output_root).mkdir(parents=True, exist_ok=True)
    folder = Path(mkdtemp(prefix='xtts_pair_', dir=output_root))
    packages = {}
    for name in ['coqui-tts', 'torch', 'torchaudio', 'transformers']:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    report = {'status': 'RUNNING', 'backend': 'mock' if backend is not None else 'xtts-v2',
              'created_utc': datetime.now(timezone.utc).isoformat(), 'manifest': data,
              'python': platform.python_version(), 'platform': platform.platform(),
              'versions': packages, 'reference_hashes_before': before, 'outputs': [],
              'model_hashes': {p.name: sha256(p) for p in model_dir.glob('*') if p.is_file()},
              'limitations': 'No protection efficacy conclusion; synthetic audio is not aligned for STOI.'}
    try:
        for index, prompt in enumerate(data['prompts']):
            for role, ref in refs.items():
                result = synth.synthesize(ref, prompt['text'], prompt['language'],
                                          folder / f'{index}_{prompt["language"]}_{role}.wav', seed=data['seed'])
                report['outputs'].append({'reference_role': role, **result})
        report['reference_hashes_after'] = {role: sha256(ref) for role, ref in refs.items()}
        if before != report['reference_hashes_after']:
            raise RuntimeError('Input references changed during comparison')
        report['status'] = 'MOCK_PASS' if backend is not None else 'PASS'
    except Exception as error:
        report.update(status='FAIL', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['model_hashes_after'] = {p.name: sha256(p) for p in model_dir.glob('*') if p.is_file()}
        (folder / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    return folder / 'report.json', report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs')
    args = parser.parse_args()
    try:
        path, report = run_pair(args.manifest, args.output_root)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'XTTS experiment failed: {error}\n')
    print(json.dumps({'report': str(path), 'status': report['status']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
