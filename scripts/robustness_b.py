"""Validated CPU transforms; preserve inputs and expose actual output rates."""
from io import BytesIO
import math
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
import soundfile as sf

from robustness_transforms import gaussian_noise


def validate_audio(audio, rate):
    if type(rate) is not int or not 8000 <= rate <= 192000:
        raise ValueError('sample rate must be an integer between 8000 and 192000')
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise ValueError('audio must be finite, nonempty and mono')
    if np.max(np.abs(audio)) > 1:
        raise ValueError('audio must be within [-1, 1]')
    return audio


def validate_condition(condition):
    if not isinstance(condition, dict):
        raise ValueError('condition must be an object')
    name = condition.get('name')
    if name == 'gaussian_noise' and set(condition) == {'name', 'seed', 'std'}:
        gaussian_noise([0.0], seed=condition['seed'], std=condition['std'])
    elif name == 'resample' and set(condition) == {'name', 'target_hz'}:
        validate_audio([0.0], condition['target_hz'])
    elif name == 'mp3' and set(condition) == {'name', 'bitrate_kbps'}:
        bitrate = condition['bitrate_kbps']
        if type(bitrate) is not int or bitrate not in (32, 64, 96, 128, 192, 256, 320):
            raise ValueError('unsupported MP3 bitrate')
    else:
        raise ValueError('unknown transform or unexpected/missing parameters')


def transform_audio(audio, rate, condition):
    audio = validate_audio(audio, rate)
    validate_condition(condition)
    name = condition['name']
    metadata = {'parameters': dict(condition), 'clipped_samples': 0}
    if name == 'gaussian_noise':
        result = gaussian_noise(audio, seed=condition['seed'], std=condition['std'])
        raw = audio + np.random.default_rng(condition['seed']).normal(0, condition['std'], audio.shape)
        metadata['clipped_samples'] = int(np.count_nonzero(np.abs(raw) > 1))
        out_rate = rate
        metadata['implementation'] = 'numpy.default_rng; same seed for each paired input'
    elif name == 'resample':
        out_rate = condition['target_hz']
        divisor = math.gcd(rate, out_rate)
        raw = resample_poly(audio, out_rate // divisor, rate // divisor)
        metadata['clipped_samples'] = int(np.count_nonzero(np.abs(raw) > 1))
        result = np.clip(raw, -1, 1)
        metadata['implementation'] = 'scipy.signal.resample_poly; default Kaiser window'
        metadata['alignment'] = 'native target rate; no implicit conversion for evaluation'
    else:
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg is None:
            raise RuntimeError('ffmpeg is required for MP3 conversion')
        def execute(arguments):
            process = subprocess.run([ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', *arguments],
                                     capture_output=True, timeout=120)
            if process.returncode:
                raise RuntimeError(process.stderr.decode('utf-8', errors='replace'))
            return process.stdout
        with TemporaryDirectory(prefix='safespeech_mp3_') as temp:
            source = Path(temp) / 'input.wav'
            encoded = Path(temp) / 'encoded.mp3'
            wavfile.write(source, rate, audio)
            execute(['-i', str(source), '-map_metadata', '-1', '-c:a', 'libmp3lame',
                     '-b:a', str(condition['bitrate_kbps']) + 'k', '-ar', str(rate), str(encoded)])
            encoded_bytes = encoded.read_bytes()
            decoded = execute(['-i', str(encoded), '-map_metadata', '-1', '-c:a', 'pcm_f64le',
                               '-f', 'wav', 'pipe:1'])
            raw, out_rate = sf.read(BytesIO(decoded), dtype='float64')
        if raw.ndim != 1 or out_rate != rate or raw.shape != audio.shape:
            raise ValueError('MP3 round trip changed rate/channels/length; no automatic truncation')
        metadata['clipped_samples'] = int(np.count_nonzero(np.abs(raw) > 1))
        result = np.clip(raw, -1, 1)
        metadata['implementation'] = 'ffmpeg libmp3lame CBR -> pcm_f64le WAV'
        metadata['ffmpeg_version'] = subprocess.run(
            [ffmpeg, '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
        metadata['encoded_bytes'] = len(encoded_bytes)
        # Keep the compressed artifact so the decoder input can be independently checked.
        metadata['_encoded_mp3'] = encoded_bytes
    result = validate_audio(result, int(out_rate)).copy()
    metadata.update(input_rate=rate, output_rate=int(out_rate), input_samples=len(audio),
                    output_samples=len(result), input_seconds=len(audio) / rate,
                    output_seconds=len(result) / out_rate)
    return result, int(out_rate), metadata
