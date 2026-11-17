"""Download and verify the existing runner's pinned public Melo assets.

Does not accept data licenses, generate speech, or train a voice model.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

from run_korean_melo_spec import OFFICIAL, MODEL, WEIGHT_SHA, CONFIG_SHA, OVERLAY, check_assets


def download(url, target, expected):
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Existing asset has unexpected hash: {target.name}')
        return
    partial = target.with_suffix(target.suffix + '.partial')
    with urllib.request.urlopen(url, timeout=120) as source, partial.open('xb') as out:
        while block := source.read(1024 * 1024):
            out.write(block)
    if hashlib.sha256(partial.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Download hash mismatch: {target.name}')
    partial.rename(target)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assets = args.output.resolve()
    assets.mkdir(parents=True, exist_ok=True)
    source = assets / 'MeloTTS'
    if not source.exists():
        subprocess.run(['git', 'clone', '--no-checkout', 'https://github.com/myshell-ai/MeloTTS.git', str(source)], check=True)
        subprocess.run(['git', '-C', str(source), 'checkout', '--detach', OFFICIAL], check=True)
    (assets / 'model').mkdir(exist_ok=True)
    for name, digest in [('config.json', CONFIG_SHA), ('checkpoint.pth', WEIGHT_SHA)]:
        print(f'Verifying/downloading {name}', flush=True)
        download(f'https://huggingface.co/myshell-ai/MeloTTS-Korean/resolve/{MODEL}/{name}', assets / 'model' / name, digest)
    config = json.loads((assets / 'model/config.json').read_text())
    effective = copy.deepcopy(config)
    for dotted, value in OVERLAY.items():
        section, field = dotted.split('.')
        if field in effective[section]:
            raise ValueError('Overlay may only fill absent fields')
        effective[section][field] = value
    output = assets / 'effective_loss_config_verified.json'
    payload = {'config': effective, 'missing_field_overlay': OVERLAY}
    if output.exists():
        if json.loads(output.read_text()) != payload:
            raise ValueError('Existing effective config differs')
    else:
        with output.open('x') as stream:
            json.dump(payload, stream, indent=2)
    _, _, hashes = check_assets(assets)
    print(json.dumps({'status': 'PASS', 'hashes': hashes}, indent=2))


if __name__ == '__main__':
    main()
