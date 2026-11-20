"""Offline SPEC WAV-in/WAV-out command; uses the pinned Korean Melo backend."""
import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from run_korean_melo_spec import check_assets, load_backend, save, sha256
from validate_melo_spec_pair import validate
from evaluate_melo_spec_stoi import evaluate_validation


def project_pcm16(clean, optimized, epsilon):
    """Constrain saved integer samples, avoiding outward PCM16 rounding."""
    radius = math.floor((epsilon/255)*32768)
    a = np.rint(np.asarray(clean, dtype=np.float64)*32768).astype(np.int32)
    b = np.rint(np.asarray(optimized, dtype=np.float64)*32768).astype(np.int32)
    projected = np.clip(b, np.maximum(a-radius, -32768), np.minimum(a+radius, 32767))
    return projected.astype(np.int16), int(np.count_nonzero(projected != b))


def protect(source, output_dir, assets, *, iterations=10, epsilon=8., seed=52, device='cuda', deterministic=False):
    source = Path(source).resolve()
    output_dir = Path(output_dir).resolve()
    if type(iterations) is not int or not 1 <= iterations <= 200:
        raise ValueError('iterations must be 1..200')
    if not math.isfinite(epsilon) or not 0 < epsilon <= 255:
        raise ValueError('epsilon must be in (0,255], measured in /255 units')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be uint32')
    info = sf.info(source)
    wave, sr = sf.read(source, dtype='float32')
    if wave.ndim != 1 or sr != 44100 or not 1 <= len(wave)/sr <= 10:
        raise ValueError('Require mono 44100 Hz WAV of 1..10 seconds; caller must resample explicitly')
    if info.format != 'WAV' or info.subtype != 'PCM_16' or not np.isfinite(wave).all() or not np.any(wave):
        raise ValueError('Require non-silent finite PCM16 WAV')
    if np.max(np.abs(wave)) > 1:
        raise ValueError('Input exceeds normalized amplitude range')
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    assets, config, hashes = check_assets(assets)
    before = sha256(source)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {'status':'FAIL', 'input':str(source), 'input_sha256':before,
              'iterations':iterations, 'epsilon_units_255':epsilon, 'seed':seed,
              'device':device, 'deterministic':deterministic, 'assets_sha256':hashes}
    try:
        generate = load_backend(assets, config, output_dir, device)
        raw_output = output_dir/'optimized.wav'
        output = output_dir/'protected.wav'
        settings = SimpleNamespace(epsilon=epsilon, step=None, iters=iterations, seed=seed,
                                   mode='SPEC', beta=10., alpha=.05, log_every=1, random_init=False)
        stats = generate(source, raw_output, {'model_speaker':'KR'}, settings)
        save(output_dir/'generation_unprojected.json', {'args':vars(settings), 'model_sampling_rate':44100, 'files':[stats.copy()]})
        optimized, _ = sf.read(raw_output, dtype='float64')
        codes, adjusted = project_pcm16(wave, optimized, epsilon)
        with output.open('xb') as stream:
            sf.write(stream, codes, sr, format='WAV', subtype='PCM_16')
        stats['output'] = str(output)
        actual, _ = sf.read(output, dtype='float64')
        linf = float(np.max(np.abs(actual-wave.astype(np.float64))))
        delta = actual-wave.astype(np.float64)
        noise_power = float(np.sum(delta**2))
        snr_db = float(10*np.log10(np.sum(wave.astype(np.float64)**2)/noise_power)) if noise_power else None
        stats.update(max_abs_delta=linf, snr_db=snr_db)
        if linf > epsilon/255:
            raise ValueError('Saved WAV violates strict epsilon bound')
        save(output_dir/'generation.json', {'args':vars(settings), 'model_sampling_rate':44100, 'files':[stats]})
        valid = validate(output_dir/'generation.json', source, output,
                         expected_original_sha256=before, execution_id=output_dir.name, input_kind='melo_artifact')
        save(output_dir/'validation.json', valid)
        score = evaluate_validation(output_dir/'validation.json')
        save(output_dir/'stoi.json', score)
        if sha256(source) != before:
            raise RuntimeError('Input changed during protection')
        report.update(status='PASS', output=str(output), output_sha256=sha256(output),
                      sample_rate=sr, samples=len(wave), subtype='PCM_16', stoi=score['stoi'],
                      epsilon_normalized=epsilon/255, strict_linf=linf,
                      snr_db=snr_db, pcm16_adjusted_samples=adjusted,
                      unprojected_output=str(raw_output), unprojected_sha256=sha256(raw_output))
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        save(output_dir/'result.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=10)
    p.add_argument('--epsilon', type=float, default=8.)
    p.add_argument('--seed', type=int, default=52)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--deterministic', action='store_true', help='Require deterministic PyTorch algorithms; fail if unavailable')
    args = p.parse_args()
    try:
        result = protect(args.input, args.output_dir, args.assets, iterations=args.iterations,
                         epsilon=args.epsilon, seed=args.seed, device=args.device, deterministic=args.deterministic)
    except Exception as error:
        p.exit(1, f'{type(error).__name__}: {error}\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
