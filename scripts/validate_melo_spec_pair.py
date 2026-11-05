"""CPU artifact contract for melo_kr/protect_melo.py; no model imports."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_metrics import load_audio

MELO_SOURCE_REVISION = 'b6db484367d38909b4ffd74e06fcf1166cd1e0b9'
PCM16_LSB = 1 / 32768
FLOAT32_TOLERANCE = 1e-7


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    if positive and value <= 0:
        raise ValueError(f'{label} must be positive')
    return value


def basename(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Generation input/output paths must be nonempty strings')
    return value.replace('\\', '/').rsplit('/', 1)[-1]


def validate(report_path, original_path, protected_path, *, expected_original_sha256,
             execution_id, input_kind, record_index=0):
    if input_kind not in ('mock_contract', 'melo_artifact'):
        raise ValueError('input_kind must be mock_contract or melo_artifact')
    if not isinstance(execution_id, str) or not execution_id.strip():
        raise ValueError('A nonempty execution_id is required')
    if (not isinstance(expected_original_sha256, str) or len(expected_original_sha256) != 64
            or any(c not in '0123456789abcdef' for c in expected_original_sha256)):
        raise ValueError('Expected original SHA-256 must be 64 lowercase hexadecimal characters')
    report_path, original, protected = map(lambda p: Path(p).resolve(),
                                         (report_path, original_path, protected_path))
    if len({report_path, original, protected}) != 3:
        raise ValueError('Generation report, original and protected paths must be distinct')
    before = {str(p): sha256(p) for p in (report_path, original, protected)}
    if before[str(original)] != expected_original_sha256:
        raise ValueError('Original hash differs from the pre-generation baseline')
    generation = json.loads(report_path.read_text(encoding='utf-8'))
    args, entries = generation['args'], generation['files']
    if not isinstance(args, dict) or not isinstance(entries, list):
        raise ValueError('Require melo_kr generation args and files list')
    if type(record_index) is not int or not 0 <= record_index < len(entries):
        raise ValueError('Invalid report record index')
    # Current producer flattens input paths to basenames: reject ambiguous runs.
    input_names = [basename(entry['input']) for entry in entries]
    output_names = [basename(entry['output']) for entry in entries]
    if len(set(input_names)) != len(entries) or len(set(output_names)) != len(entries):
        raise ValueError('Duplicate basenames make this generation report ambiguous')
    entry = entries[record_index]
    if input_names[record_index] != original.name or output_names[record_index] != protected.name:
        raise ValueError('Explicit WAVs do not match the selected generation record filenames')
    if args.get('mode') != 'SPEC':
        raise ValueError('This validator accepts SPEC mode only')
    if type(args.get('iters')) is not int or args['iters'] < 1:
        raise ValueError('Invalid SPEC iteration count')
    if type(args.get('seed')) is not int or not 0 <= args['seed'] < 2**32:
        raise ValueError('Require a resolved nonnegative seed in the generation report')
    if type(args.get('random_init')) is not bool:
        raise ValueError('Require explicit random_init in the generation report')
    if input_kind == 'melo_artifact' and args['random_init']:
        raise ValueError('Random-initialized runs cannot be declared pretrained Melo artifacts')
    epsilon = number(args['epsilon'], 'epsilon', positive=True) / 255
    if epsilon > 1:
        raise ValueError('epsilon must not exceed 255 amplitude units')
    model_rate = generation['model_sampling_rate']
    if type(model_rate) is not int or model_rate <= 0:
        raise ValueError('Invalid model sampling rate')
    reported_delta = number(entry['max_abs_delta'], 'reported max_abs_delta', positive=True)
    if reported_delta > epsilon + FLOAT32_TOLERANCE:
        raise ValueError('Pre-quantization reported delta exceeds epsilon')
    for key in ('recon_mel_l1_clean', 'recon_mel_l1_protected', 'snr_db'):
        number(entry[key], key)
    history = entry['history']
    if not isinstance(history, list) or not history:
        raise ValueError('Require nonempty loss history')
    for step in history:
        if type(step['iter']) is not int or not 0 <= step['iter'] < args['iters']:
            raise ValueError('Invalid history iteration')
        for key in ('loss_mel', 'loss_nr', 'loss_kl'):
            number(step[key], key)
    history_iterations = [step['iter'] for step in history]
    if (history_iterations != sorted(set(history_iterations))
            or history_iterations[0] != 0 or history_iterations[-1] != args['iters'] - 1):
        raise ValueError('Loss history must cover the first and final iterations in order')
    clean, rate = load_audio(original)
    candidate, candidate_rate = load_audio(protected)
    clean_info, protected_info = sf.info(original), sf.info(protected)
    if rate != candidate_rate or clean.shape != candidate.shape:
        raise ValueError('Original/protected sample rate or length mismatch; no alignment is performed')
    if entry['sample_rate'] != rate:
        raise ValueError('Generation record sample rate differs from WAV')
    if protected_info.subtype != 'PCM_16':
        raise ValueError('Expected PCM_16 output from protect_melo.py')
    maximum = float(np.max(np.abs(candidate - clean)))
    tolerance = PCM16_LSB + FLOAT32_TOLERANCE
    if not 0 < maximum <= epsilon + tolerance:
        raise ValueError('Saved PCM16 delta is zero or exceeds epsilon plus quantization tolerance')
    if abs(maximum - reported_delta) > tolerance:
        raise ValueError('Saved WAV delta does not match reported pre-quantization delta')
    after = {str(p): sha256(p) for p in (report_path, original, protected)}
    if before != after:
        raise RuntimeError('An input changed during artifact validation')

    def info(wave, metadata):
        return {'sample_rate': metadata.samplerate, 'channels': metadata.channels,
                'samples': len(wave), 'subtype': metadata.subtype,
                'peak': float(np.max(np.abs(wave))), 'rms': float(np.sqrt(np.mean(wave ** 2))),
                'nan_count': 0, 'inf_count': 0,
                'boundary_samples': int(np.count_nonzero((wave <= -1) | (wave >= 32767 / 32768)))}

    return {
        'schema_version': 1, 'status': 'MOCK_PASS' if input_kind == 'mock_contract' else 'PASS',
        'scope': 'mock_input_contract_only' if input_kind == 'mock_contract' else 'melo_artifact_integrity_only',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'execution_id': execution_id,
        'input_kind': input_kind, 'model_kind': 'MeloTTS-Korean', 'mode': 'SPEC',
        'model_sampling_rate': model_rate, 'seed': args['seed'], 'iterations': args['iters'],
        'random_init_declared': args['random_init'], 'model_weights_verified': False,
        'producer_schema_reference': MELO_SOURCE_REVISION, 'producer_revision_verified': False,
        'generation_report': str(report_path), 'record_index': record_index,
        'original_path': str(original), 'protected_path': str(protected),
        'original': info(clean, clean_info), 'protected': info(candidate, protected_info),
        'epsilon': epsilon, 'epsilon_units_255': args['epsilon'],
        'pcm16_tolerance': PCM16_LSB, 'float32_tolerance': FLOAT32_TOLERANCE,
        'max_abs_delta': maximum, 'reported_pre_quantization_delta': reported_delta,
        'expected_original_sha256': expected_original_sha256,
        'files_sha256': before, 'files_sha256_after': after, 'input_files_unchanged': True,
        'limitations': 'No saved noise reconstruction or model execution proof. Boundary samples do not prove prior clipping. SHA baseline must be recorded before generation.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation-report', type=Path, required=True)
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--protected', type=Path, required=True)
    parser.add_argument('--expected-original-sha256', required=True)
    parser.add_argument('--execution-id', required=True)
    parser.add_argument('--input-kind', choices=['mock_contract', 'melo_artifact'], required=True)
    parser.add_argument('--record-index', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = {p.resolve() for p in (args.generation_report, args.original, args.protected)}
    try:
        if args.output.resolve() in inputs or args.output.exists():
            raise FileExistsError('Validation output must be a new file distinct from inputs')
        result = validate(args.generation_report, args.original, args.protected,
                          expected_original_sha256=args.expected_original_sha256,
                          execution_id=args.execution_id, input_kind=args.input_kind,
                          record_index=args.record_index)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f'Melo validation failed: {error}\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
