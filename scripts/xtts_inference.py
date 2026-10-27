"""Isolated XTTS adapter. Heavy dependencies are imported only for real inference."""
from pathlib import Path
import hashlib

import numpy as np
import soundfile as sf

from audio_metrics import load_audio

CONDITIONING = {'max_ref_length': 30, 'gpt_cond_len': 6, 'gpt_cond_chunk_len': 6,
                'sound_norm_refs': False, 'librosa_trim_db': None, 'load_sr': 22050}
DECODING = {'do_sample': False, 'num_beams': 1, 'speed': 1.0,
            'temperature': 0.75, 'length_penalty': 1.0, 'repetition_penalty': 10.0,
            'top_k': 50, 'top_p': 0.85, 'enable_text_splitting': False}

class XttsError(RuntimeError):
    pass


class CudaMemoryError(XttsError):
    pass


def validate_request(reference_wav, text, language, output_path):
    if language not in ('ko', 'en'):
        raise ValueError('language must be exactly ko or en')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('text must be a nonempty string')
    reference = Path(reference_wav).resolve()
    output = Path(output_path).resolve()
    if not reference.is_file():
        raise FileNotFoundError(f'Reference WAV missing: {reference}')
    if reference.suffix.lower() != '.wav' or output.suffix.lower() != '.wav':
        raise ValueError('Reference and output must use .wav paths')
    load_audio(reference)
    if output == reference or output.exists():
        raise FileExistsError(f'Output must be a new file: {output}')
    return reference, output


class LocalXttsBackend:
    """Official local-checkpoint API; no automatic downloads or license acceptance."""
    def __init__(self, model_dir, device):
        if device not in ('cpu', 'cuda:0'):
            raise ValueError('device must be cpu or cuda:0')
        if model_dir is None:
            raise ValueError('model_dir is required for real inference')
        folder = Path(model_dir).resolve()
        for name in ('config.json', 'model.pth', 'vocab.json'):
            if not (folder / name).is_file():
                raise FileNotFoundError(f'XTTS checkpoint file missing: {folder / name}')
        try:
            import torch
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import Xtts
        except ImportError as error:
            raise XttsError('Install requirements-xtts.txt in a separate environment') from error
        self.torch, self.device = torch, device
        if device == 'cuda:0' and not torch.cuda.is_available():
            raise XttsError('CUDA unavailable; run on the Windows WSL GPU environment')
        if device == 'cuda:0':
            torch.cuda.reset_peak_memory_stats(device)
        try:
            config = XttsConfig()
            config.load_json(str(folder / 'config.json'))
            self.model = Xtts.init_from_config(config)
            self.model.load_checkpoint(config, checkpoint_dir=str(folder), eval=True)
            self.model.to(device)
            self.rate = config.audio.output_sample_rate
            self.load_memory = self.memory()
        except torch.cuda.OutOfMemoryError as error:
            raise CudaMemoryError('CUDA OOM loading XTTS; release GPU memory and retry') from error

    def memory(self):
        if self.device == 'cpu':
            return None
        cuda = self.torch.cuda
        cuda.synchronize(self.device)
        return {'allocated_bytes': cuda.memory_allocated(self.device),
                'reserved_bytes': cuda.memory_reserved(self.device),
                'peak_allocated_bytes': cuda.max_memory_allocated(self.device),
                'total_bytes': cuda.get_device_properties(self.device).total_memory}

    def synthesize(self, reference_wav, text, language, seed):
        torch = self.torch
        try:
            if self.device == 'cuda:0':
                torch.cuda.reset_peak_memory_stats(self.device)
            with torch.random.fork_rng(devices=[0] if self.device == 'cuda:0' else []):
                torch.manual_seed(seed)
                with torch.inference_mode():
                    latent, embedding = self.model.get_conditioning_latents(
                        audio_path=[str(reference_wav)], **CONDITIONING)
                    result = self.model.inference(text, language, latent, embedding, **DECODING)
            return result['wav'], self.rate, {
                'load_memory': self.load_memory, 'inference_memory': self.memory(),
                'decoding': DECODING, 'conditioning': CONDITIONING, 'device': self.device,
            }
        except torch.cuda.OutOfMemoryError as error:
            raise CudaMemoryError('CUDA OOM during synthesis; shorten text/reference or release GPU memory') from error


class XttsSynthesizer:
    def __init__(self, model_dir=None, device='cuda:0', *, consent_confirmed=False,
                 cpml_accepted=False, backend=None):
        if consent_confirmed is not True or cpml_accepted is not True:
            raise ValueError('Confirm reference permission and review/accept CPML before inference')
        self.model_dir, self.device, self.backend = model_dir, device, backend

    def synthesize(self, reference_wav, text, language, output_path, *, seed=42):
        reference, output = validate_request(reference_wav, text, language, output_path)
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError('seed must be an integer in [0, 2**32)')
        before = hashlib.sha256(reference.read_bytes()).hexdigest()
        if self.backend is None:
            self.backend = LocalXttsBackend(self.model_dir, self.device)
        wave, rate, runtime = self.backend.synthesize(reference, text, language, seed)
        wave = np.asarray(wave, dtype=np.float64)
        if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all() or type(rate) is not int or rate <= 0:
            raise XttsError('XTTS returned invalid audio')
        if hashlib.sha256(reference.read_bytes()).hexdigest() != before:
            raise XttsError('Reference file changed during inference')
        output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation prevents overwriting existing files, including references.
        with output.open('xb') as stream:
            sf.write(stream, wave, rate, format='WAV', subtype='FLOAT')
        return {'status': 'PASS', 'reference_wav': str(reference), 'reference_sha256': before,
                'text': text, 'language': language, 'seed': seed, 'output_path': str(output),
                'output_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
                'samples': len(wave), 'sample_rate': rate, 'runtime': runtime}


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-wav', type=Path, required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--language', choices=['ko', 'en'], required=True)
    parser.add_argument('--output-path', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--consent-confirmed', action='store_true')
    parser.add_argument('--cpml-accepted', action='store_true')
    args = parser.parse_args()
    try:
        synth = XttsSynthesizer(args.model_dir, args.device, consent_confirmed=args.consent_confirmed,
                                cpml_accepted=args.cpml_accepted)
        report = synth.synthesize(args.reference_wav, args.text, args.language, args.output_path, seed=args.seed)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f'XTTS synthesis failed: {error}\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
