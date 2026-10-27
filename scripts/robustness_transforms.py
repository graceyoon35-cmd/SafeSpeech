"""Small deterministic CPU transforms for shared experiment plumbing."""
import numpy as np


def gaussian_noise(audio, *, seed, std):
    """Return a new float64 waveform with additive noise, clipped to [-1, 1]."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    if isinstance(std, bool) or not isinstance(std, (int, float)) or not np.isfinite(std) or std < 0:
        raise ValueError('std must be a finite non-negative number')
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all() or np.max(np.abs(audio)) > 1:
        raise ValueError('Expected a finite nonempty mono waveform in [-1, 1]')
    return np.clip(audio + np.random.default_rng(seed).normal(0, std, audio.shape), -1, 1)
