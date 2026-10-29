"""Validate saved SPEC artifacts without modifying audio or generation reports."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audio_info(path):
    audio, rate = sf.read(path, dtype='float64')
    info = sf.info(path)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError(f'Expected nonempty finite mono WAV: {path}')
    stats = dict(sample_rate=rate, samples=len(audio), channels=info.channels,
                 duration_seconds=len(audio) / rate, subtype=info.subtype,
                 minimum=float(audio.min()), maximum=float(audio.max()),
                 peak=float(np.abs(audio).max()), rms=float(np.sqrt(np.mean(audio ** 2))),
                 nan_count=int(np.isnan(audio).sum()), inf_count=int(np.isinf(audio).sum()))
    if stats['peak'] > 1 or stats['rms'] < 1e-6 or len(audio) < rate:
        raise ValueError(f'WAV outside strict evaluation requirements: {path}')
    return audio, stats


def validate(run, expected_source_sha256=None):
    run = Path(run).resolve()
    root = run.parent.parent
    report_path = run / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    source = root / report['source']
    original = run / source.name
    protected = run / 'protected.wav'
    noise_path = run / 'perturbation.pt'
    paths = [source, original, protected, noise_path, report_path]
    before = {str(path): sha256(path) for path in paths}
    if expected_source_sha256 and before[str(source)] != expected_source_sha256:
        raise ValueError('Source changed since pre-execution snapshot')
    if before[str(source)] != before[str(original)]:
        raise ValueError('Copied input differs from source bytes')
    if report['status'] != 'PASS' or report['mode'] != 'SPEC':
        raise ValueError('Not a successful SPEC generation report')
    if type(report['iterations']) is not int or report['iterations'] < 1:
        raise ValueError('Invalid iteration count')
    epsilon = float(report['epsilon'])
    if not math.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError('Invalid epsilon')
    if not all(math.isfinite(float(value)) for value in report['losses'].values()):
        raise ValueError('Non-finite final loss')
    clean, clean_info = audio_info(original)
    candidate, protected_info = audio_info(protected)
    if clean.shape != candidate.shape or clean_info['sample_rate'] != protected_info['sample_rate']:
        raise ValueError('Sample rate or length mismatch')
    delta = float(np.max(np.abs(candidate - clean)))
    if not 0 < delta <= epsilon + 1e-7:
        raise ValueError(f'Saved WAV perturbation exceeds epsilon or is zero: {delta}')
    noise = torch.load(noise_path, map_location='cpu', weights_only=True)
    if tuple(noise.shape) != (1, 1, len(clean)) or not torch.isfinite(noise).all():
        raise ValueError('Invalid saved perturbation tensor')
    if not 0 < float(noise.abs().max()) <= epsilon + 1e-7:
        raise ValueError('Invalid perturbation amplitude')
    reconstructed = np.clip(clean.astype(np.float32) + noise[0, 0].numpy(), -1, 1)
    if not np.array_equal(reconstructed.astype(np.float64), candidate):
        raise ValueError('Protected WAV does not equal clip(original + saved noise)')
    after = {str(path): sha256(path) for path in paths}
    if before != after:
        raise ValueError('Input artifacts changed during validation')
    return dict(status='PASS', run_dir=str(run), iterations=report['iterations'],
                original=clean_info, protected=protected_info, max_abs_delta=delta,
                epsilon=epsilon, files_sha256=before, source_copy_identical=True,
                saved_noise_reconstructs_wav=True, files_unchanged_during_validation=True,
                source_matches_pre_execution_snapshot=bool(expected_source_sha256),
                scope='Artifact integrity and execution only; no STOI or cloning defense evaluation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--expected-source-sha256')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = validate(args.run_dir, args.expected_source_sha256)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
