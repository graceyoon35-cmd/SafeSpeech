"""CPU-only audio-pair contract and STOI smoke check; no model downloads."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

import soundfile as sf
from audio_metrics import load_audio, score_pair
from robustness_transforms import gaussian_noise

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIO = ROOT / 'data/LibriTTS/5339/5339_14134_000092_000003.wav'



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=DEFAULT_AUDIO)
    parser.add_argument('--candidate', type=Path, help='Optional aligned mono WAV; otherwise seeded noise control')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs')
    args = parser.parse_args()
    reference, rate = load_audio(args.reference)
    identity = score_pair(reference, reference, rate, rate)
    if abs(identity['stoi'] - 1) > 1e-6:
        raise ValueError('Identity STOI sanity check failed')
    if args.candidate:
        candidate, candidate_rate = load_audio(args.candidate)
        kind = 'user_supplied_pair'
    else:
        candidate = gaussian_noise(reference, seed=args.seed, std=0.005)
        candidate_rate = rate
        kind = 'seeded_noise_control_not_SPEC'
    pair = score_pair(reference, candidate, rate, candidate_rate)
    args.output_root.mkdir(parents=True, exist_ok=True)
    from tempfile import mkdtemp
    output = Path(mkdtemp(prefix='evaluation_smoke_', dir=args.output_root))
    sf.write(output / 'reference.wav', reference, rate, subtype='FLOAT')
    sf.write(output / 'candidate.wav', candidate, rate, subtype='FLOAT')
    report = {
        'status': 'PASS', 'scope': 'audio_pair_STOI_smoke_only',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'platform': platform.platform(), 'device': 'cpu',
        'versions': {name: version(name) for name in ['numpy', 'scipy', 'soundfile', 'pystoi']},
        'reference': str(args.reference.resolve()),
        'candidate': str(args.candidate.resolve()) if args.candidate else None,
        'sample_rate': rate, 'samples': len(reference), 'seed': args.seed,
        'candidate_kind': kind, 'identity': identity, 'pair': pair,
        'SIM': {'status': 'NOT_RUN', 'reason': 'Requires separate SpeechBrain/ECAPA model validation'},
        'limitations': 'No synthesis, protection efficacy, robustness or training validation. pystoi STOI is not the torch_stoi training loss.',
    }
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': str(output / 'report.json'), **report}, indent=2))


if __name__ == '__main__':
    main()
