"""Pinned Melo SPEC runner and private Korean clean/protected handoff contract."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf
from audio_metrics import load_audio
from validate_melo_spec_pair import sha256, validate
from evaluate_melo_spec_stoi import evaluate_validation

ROOT = Path(__file__).resolve().parents[1]
TEAM = 'b6db484367d38909b4ffd74e06fcf1166cd1e0b9'
OFFICIAL = '209145371cff8fc3bd60d7be902ea69cbdb7965a'
MODEL = '0207e5adfc90129a51b6b03d89be6d84360ed323'
WEIGHT_SHA = '48e3ff3fd0b5348e095f0468e60ae727507564100f58142ef3a922ead6e0a4d0'
CONFIG_SHA = '74543376976dfadde45ba34336fa79c7e95509f43a7c2e701b22c0f71fd7695c'
OVERLAY = {'data.win_length': 2048, 'data.n_mel_channels': 128,
           'data.mel_fmin': 0.0, 'data.mel_fmax': None, 'train.c_mel': 45}


def save(path, data):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', '-C', str(cwd), *args])


def check_manifest(path, *, allow_nonhuman=False):
    """Checks declarations, not independent consent or transcript recognition."""
    path = Path(path).resolve()
    before = sha256(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('schema_version') != 1 or not data.get('utterances'):
        raise ValueError('Require schema_version=1 and nonempty utterances')
    ids, sources, result = set(), set(), []
    for row in data['utterances']:
        row = copy.deepcopy(row)
        uid = row['utterance_id']
        if not isinstance(uid, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', uid) or uid in ids:
            raise ValueError('Unsafe or duplicate utterance_id')
        ids.add(uid)
        if not isinstance(row.get('speaker_id'), str) or not re.fullmatch(r'[A-Za-z0-9_-]+', row['speaker_id']):
            raise ValueError('Use a safe pseudonymous speaker_id')
        for key in ('speaker_id', 'source', 'permission_reference', 'transcript_reference'):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f'{uid}: missing {key}')
        if row.get('language') != 'ko' or row.get('split') not in ('train', 'validation', 'test'):
            raise ValueError('Require Korean language and explicit split')
        if row.get('model_speaker') != 'KR':
            raise ValueError('Model speaker must be KR; human speaker_id is a separate identity')
        text = row.get('text')
        if not isinstance(text, str) or not re.search('[가-힣]', text) or any(c in text for c in '|\r\n'):
            raise ValueError('Require Korean transcript without metadata separators')
        if row.get('transcript_verified') is not True:
            raise ValueError('Require reviewed transcript declaration')
        kind = row.get('input_kind')
        if kind not in ('human_recording', 'model_synthesized', 'mock'):
            raise ValueError('Explicit input_kind required')
        if kind != 'human_recording' and not allow_nonhuman:
            raise ValueError('Nonhuman input requires explicit --allow-nonhuman')
        if row.get('research_use_permitted') is not True:
            raise ValueError('Research use permission declaration required')
        if kind == 'human_recording' and row.get('permission_scope') != 'spec_and_finetuning':
            raise ValueError('Permission must cover SPEC and clean/protected finetuning')
        audio_path = (path.parent / row['original_path']).resolve()
        if audio_path in sources:
            raise ValueError('Same source cannot appear in multiple utterances/splits')
        sources.add(audio_path)
        digest = sha256(audio_path)
        if digest != row.get('original_sha256'):
            raise ValueError('Original hash mismatch')
        wave, sr = load_audio(audio_path)
        if type(row.get('sample_rate')) is not int or row['sample_rate'] != sr or sr not in (16000, 22050, 24000, 44100, 48000):
            raise ValueError('Unsupported or mismatched sample rate')
        if not 1 <= len(wave) / sr <= 10:
            raise ValueError('Initial experiment supports 1-10 second mono inputs')
        row.update(original_path=str(audio_path), samples=len(wave))
        result.append(row)
    if len({r['original_sha256'] for r in result}) != len(result):
        raise ValueError('Duplicate recording content across IDs/splits')
    if sha256(path) != before:
        raise RuntimeError('Manifest changed during validation')
    return result


def validate_handoff(rows):
    """Recheck exact pair identity, split and saved WAV bindings before export."""
    seen = set()
    identity = {}
    for r in rows:
        key = (r['utterance_id'], r['iterations'])
        if key in seen:
            raise ValueError('Duplicate handoff condition')
        seen.add(key)
        fixed = tuple(r[k] for k in ('text', 'speaker_id', 'split', 'clean_sha256'))
        if r['split'] not in ('train', 'validation', 'test'):
            raise ValueError('Invalid split')
        if r['utterance_id'] in identity and identity[r['utterance_id']] != fixed:
            raise ValueError('Clean/protected split or transcript identity mismatch')
        identity[r['utterance_id']] = fixed
        for role in ('clean', 'protected'):
            if r[role]['split'] != r['split'] or r[role]['text'] != r['text'] or r[role]['speaker_id'] != r['speaker_id']:
                raise ValueError('Pair metadata mismatch')
            if sha256(r[f'{role}_path']) != r[f'{role}_sha256']:
                raise ValueError('Handoff file hash mismatch')
        a, sr = load_audio(r['clean_path'])
        b, br = load_audio(r['protected_path'])
        if sr != br or a.shape != b.shape or sf.info(r['protected_path']).subtype != 'PCM_16':
            raise ValueError('Handoff audio contract mismatch')
        delta = float(np.max(np.abs(a-b)))
        if not 0 < delta <= r['epsilon'] + 1/32768 + 1e-7:
            raise ValueError('Handoff epsilon violation')
    return True


def export_finetune_inputs(folder, rows):
    """Private split-aware inputs for the team's prepare_finetune.py, no training config."""
    validate_handoff(rows)
    if any(r['input_kind'] != 'human_recording' for r in rows):
        raise ValueError('Nonhuman data cannot be exported as finetuning inputs')
    groups = {}
    for row in rows:
        for condition, role in [('clean', 'clean'), (f"SPEC_{row['iterations']}", 'protected')]:
            bucket = groups.setdefault((condition, row['split'], row['speaker_id']), {})
            bucket[row['utterance_id']] = (row, role)
    for (condition, split, speaker), records in groups.items():
        target = Path(folder)/'finetune_inputs'/condition/split/speaker
        (target/'wavs').mkdir(parents=True, exist_ok=False)
        with (target/'transcripts.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['file', 'text'])
            for uid, (row, role) in sorted(records.items()):
                dst = target/'wavs'/f'{uid}.wav'
                with dst.open('xb') as out, Path(row[f'{role}_path']).open('rb') as inp:
                    shutil.copyfileobj(inp, out)
                if sha256(dst) != row[f'{role}_sha256']:
                    raise ValueError('Finetuning export hash mismatch')
                writer.writerow([dst.name, row['text']])
    save(Path(folder)/'finetune_inputs'/'README.json', {
        'status': 'INPUTS_FOR_PREPARE_FINETUNE_NOT_TRAINING_CONFIG',
        'sample_rate_policy': 'Original rates preserved; team prepare_finetune resamples to 44100.',
        'split_policy': 'Run prepare_finetune separately for each condition/split; never randomly re-split.',
        'speaker_policy': 'Pass real pseudonymous speaker_id; group multi-speaker inputs separately.',
        'config': 'Team must create train/validation/test config. prepare_finetune does not generate config.json.'})


def check_assets(assets):
    assets = Path(assets).resolve()
    melo = assets / 'MeloTTS'
    if git('rev-parse', 'HEAD', cwd=melo).decode().strip() != OFFICIAL:
        raise ValueError('Official source revision mismatch')
    if git('status', '--porcelain', '--untracked-files=all', cwd=melo).strip():
        raise ValueError('Official Melo source checkout must be clean')
    checks = {'model/checkpoint.pth': WEIGHT_SHA, 'model/config.json': CONFIG_SHA}
    for name, expected in checks.items():
        if sha256(assets/name) != expected:
            raise ValueError(f'Pinned model asset mismatch: {name}')
    raw = json.loads((assets/'model/config.json').read_text())
    effective = json.loads((assets/'effective_loss_config_verified.json').read_text())
    wanted = copy.deepcopy(raw)
    for key, value in OVERLAY.items():
        section, field = key.split('.')
        if field in wanted[section]:
            raise ValueError('Overlay must only fill absent training fields')
        wanted[section][field] = value
    if effective['config'] != wanted or effective['missing_field_overlay'] != OVERLAY:
        raise ValueError('Effective config differs from approved missing-field overlay')
    if wanted['data']['sampling_rate'] != 44100 or wanted['data']['spk2id'].get('KR') != 0:
        raise ValueError('Korean model rate/speaker mismatch')
    checks['effective_loss_config_verified.json'] = sha256(assets/'effective_loss_config_verified.json')
    return assets, wanted, checks


def load_backend(assets, config, folder, device):
    """Export exact pinned team files into a NEW run, never mutate other branches."""
    team = folder/'team'/'melo_kr'
    team.mkdir(parents=True)
    team_hashes = {}
    for name in ('common.py', 'protect_melo.py'):
        content = git('show', f'{TEAM}:melo_kr/{name}')
        with (team/name).open('xb') as f:
            f.write(content)
        team_hashes[name] = sha256(team/name)
    sys.path[:0] = [str(team), str(assets/'MeloTTS'), str(ROOT)]
    import torch
    import common as C
    import protect_melo as P
    if Path(C.__file__).resolve() != team/'common.py' or Path(P.__file__).resolve() != team/'protect_melo.py':
        raise RuntimeError('Unexpected imported team module')
    torch.set_num_threads(4)
    torch.manual_seed(52)
    hps = C.to_ns(config)
    model = C.build_melo(hps, device)
    state = torch.load(assets/'model/checkpoint.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True)
    del state
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    save(folder/'backend.json', {'python': sys.version, 'torch': torch.__version__,
         'device': device, 'strict_state_dict': True, 'team_source_sha256': team_hashes})

    def generate(original, output, row, settings):
        audio, sr = sf.read(original, dtype='float32')
        torch.manual_seed(settings.seed)
        wave = torch.from_numpy(audio.copy()).view(1, 1, -1).to(device)
        sid = torch.tensor([config['data']['spk2id'][row['model_speaker']]], device=device)
        checks = []
        actual = torch.autograd.grad

        def inspect(*args, **kwargs):
            loss = args[0]
            if not torch.isfinite(loss).all():
                raise ValueError('Nonfinite loss')
            returned = actual(*args, **kwargs)
            gradient = returned[0]
            if gradient is None or not torch.isfinite(gradient).all() or not torch.count_nonzero(gradient):
                raise ValueError('Missing/nonfinite/zero gradient')
            checks.append({'iteration': len(checks)+1, 'loss': float(loss.detach()),
                           'finite': True, 'nonzero': int(torch.count_nonzero(gradient)),
                           'max_abs': float(gradient.abs().max()), 'l2': float(gradient.norm())})
            return returned

        started = time.perf_counter()
        try:
            with patch.object(torch.autograd, 'grad', side_effect=inspect):
                protected, stats = P.perturb(model, hps, wave, sr, sid, settings, None)
            if len(checks) != settings.iters:
                raise ValueError('Gradient history incomplete')
            for item in stats['history']:
                if not all(math.isfinite(item[k]) for k in ('loss_mel', 'loss_nr', 'loss_kl')):
                    raise ValueError('Nonfinite loss component')
            with output.open('xb') as f:
                sf.write(f, protected.reshape(-1).cpu().numpy(), sr, format='WAV', subtype='PCM_16')
            stats.update(input=str(original), output=str(output), sample_rate=sr,
                         seconds=time.perf_counter()-started, audio_seconds=len(audio)/sr)
            return stats
        finally:
            save(output.parent/'gradients.json', checks)
    return generate


def execute(manifest, assets, *, seed=52, epsilon=8., iterations=(1, 10), device='cpu', allow_nonhuman=False):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Invalid seed')
    if not math.isfinite(epsilon) or not 0 < epsilon <= 255:
        raise ValueError('Invalid epsilon in /255 units')
    if not iterations or len(set(iterations)) != len(iterations) or any(type(i) is not int or not 1 <= i <= 200 for i in iterations):
        raise ValueError('Iterations must be unique integers in 1..200')
    rows = check_manifest(manifest, allow_nonhuman=allow_nonhuman)
    manifest_hash = sha256(manifest)
    assets, config, hashes = check_assets(assets)
    runs = ROOT/'runs'
    runs.mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='korean_melo_spec_', dir=runs))
    started = time.perf_counter()
    report = {'status': 'FAIL', 'human_recording_experiment': all(r['input_kind']=='human_recording' for r in rows),
              'team_revision': TEAM, 'official_revision': OFFICIAL, 'model_revision': MODEL,
              'assets_sha256': hashes, 'seed': seed, 'epsilon_units_255': epsilon,
              'iterations': list(iterations), 'device': device, 'input_manifest_sha256': manifest_hash,
              'runner_sha256': sha256(Path(__file__)), 'project_head': git('rev-parse', 'HEAD').decode().strip(),
              'limitations': 'STOI is intelligibility, not cloning defense. Transcript/permission are declarations.'}
    try:
        save(folder/'input_manifest.json', {'schema_version': 1, 'utterances': rows})
        save(folder/'effective_config.json', config)
        generate = load_backend(assets, config, folder, device)
        handoff = []
        for row in rows:
            item = folder/row['utterance_id']
            (item/'clean').mkdir(parents=True)
            original = item/'clean'/f"{row['utterance_id']}.wav"
            with original.open('xb') as target, Path(row['original_path']).open('rb') as source:
                shutil.copyfileobj(source, target)
            if sha256(original) != row['original_sha256']:
                raise ValueError('Source changed before copy')
            for count in iterations:
                case = item/f'spec_{count}'
                case.mkdir()
                output = case/original.name
                settings = SimpleNamespace(epsilon=epsilon, step=None, iters=count, seed=seed,
                    mode='SPEC', beta=10., alpha=.05, log_every=1, random_init=False)
                stats = generate(original, output, row, settings)
                generation = case/'report.json'
                save(generation, {'args': vars(settings), 'model_sampling_rate': 44100, 'files': [stats]})
                validation = validate(generation, original, output,
                    expected_original_sha256=row['original_sha256'], execution_id=f'{folder.name}_{row["utterance_id"]}_{count}',
                    input_kind='melo_artifact')
                save(case/'validation.json', validation)
                score = evaluate_validation(case/'validation.json')
                score['source_input_kind'] = row['input_kind']
                save(case/'stoi.json', score)
                identity = {k: row[k] for k in ('split', 'text', 'speaker_id')}
                handoff.append({**row, 'clean_path': str(original), 'clean_sha256': sha256(original),
                    'protected_path': str(output), 'protected_sha256': sha256(output),
                    'clean': identity.copy(), 'protected': identity.copy(), 'model_speaker_id': 0,
                    'epsilon': epsilon/255, 'iterations': count, 'seed': seed, 'stoi': score['stoi'],
                    'validation_path': str(case/'validation.json')})
        validate_handoff(handoff)
        if sha256(manifest) != manifest_hash or any(sha256(r['original_path']) != r['original_sha256'] for r in rows):
            raise RuntimeError('Source manifest/audio changed during run')
        for name, expected in hashes.items():
            if sha256(assets/name) != expected:
                raise RuntimeError('Model/config changed during run')
        check_assets(assets)
        for name, expected in json.loads((folder/'backend.json').read_text())['team_source_sha256'].items():
            if sha256(folder/'team'/'melo_kr'/name) != expected:
                raise RuntimeError('Team code changed during run')
        eligible = report['human_recording_experiment']
        if eligible:
            export_finetune_inputs(folder, handoff)
        save(folder/'handoff_manifest.json', {'schema_version': 1,
             'status': 'READY_FOR_MODEL_REVIEW' if eligible else 'NONHUMAN_NOT_FOR_FINETUNING',
             'path_base': 'absolute; regenerate paths and verify hashes when transferring',
             'provenance': {'team': TEAM, 'official': OFFICIAL, 'model': MODEL, 'assets_sha256': hashes},
             'utterances': handoff})
        report.update(status='PASS' if eligible else 'NONHUMAN_EXECUTION_PASS',
                      handoff_count=len(handoff), input_files_unchanged=True, assets_unchanged=True)
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['total_seconds'] = time.perf_counter()-started
        save(folder/'run.json', report)
        print(json.dumps({'run': str(folder), 'status': report['status']}, ensure_ascii=False), flush=True)
    return folder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--assets', type=Path)
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--allow-nonhuman', action='store_true')
    p.add_argument('--seed', type=int, default=52)
    p.add_argument('--epsilon', type=float, default=8.)
    p.add_argument('--iterations', type=int, nargs='+', default=[1, 10])
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    try:
        if args.validate_only:
            rows = check_manifest(args.manifest, allow_nonhuman=args.allow_nonhuman)
            print(json.dumps({'status': 'INPUT_CONTRACT_PASS', 'count': len(rows),
                              'kinds': sorted({r['input_kind'] for r in rows}),
                              'scope': 'declarations and file integrity, not consent/transcript authenticity'}))
        else:
            if args.assets is None:
                p.error('--assets is required for execution')
            execute(args.manifest, args.assets, seed=args.seed, epsilon=args.epsilon,
                    iterations=tuple(args.iterations), device=args.device, allow_nonhuman=args.allow_nonhuman)
    except Exception as error:
        p.exit(1, f'{type(error).__name__}: {error}\n')


if __name__ == '__main__':
    main()
