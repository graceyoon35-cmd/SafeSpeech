"""Shared strict mono audio loading and aligned STOI scoring (CPU)."""
import warnings
import numpy as np
import soundfile as sf
from pystoi import stoi


def load_audio(path):
    audio, rate = sf.read(path, dtype='float64')
    if audio.ndim != 1:
        raise ValueError('Expected mono audio; downmix explicitly before evaluation')
    if len(audio) < rate:
        raise ValueError('Use at least one second of speech for this smoke check')
    if not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1:
        raise ValueError('Expected finite audio in [-1, 1]')
    if np.sqrt(np.mean(audio ** 2)) < 1e-6:
        raise ValueError('Silent input is not a valid STOI fixture')
    return audio, rate


def score_pair(reference, candidate, reference_rate, candidate_rate):
    if reference_rate != candidate_rate or reference.shape != candidate.shape:
        raise ValueError('Pairs must have identical sample rate and length; no implicit alignment')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        value = float(stoi(reference, candidate, reference_rate, extended=False))
    if not np.isfinite(value):
        raise ValueError('STOI returned a non-finite value')
    return {'stoi': value, 'max_abs_delta': float(np.max(np.abs(reference - candidate)))}
