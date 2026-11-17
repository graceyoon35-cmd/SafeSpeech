"""Build fixed-split, byte-preserved paired Melo datasets from a validated SPEC run."""
import argparse
import copy
import csv
import json
from pathlib import Path
import shutil
import tempfile

from melo_training_core import training_config
from run_korean_melo_spec import ROOT, check_manifest, save, sha256, validate_handoff


def prepare(spec_run, assets):
    spec_run = Path(spec_run).resolve()
    report = json.loads((spec_run/'run.json').read_text())
    if report.get('status') not in ('PASS', 'NONHUMAN_EXECUTION_PASS'):
        raise ValueError('Completed SPEC run required')
    licensed=report.get('validation_policy')=='licensed_noncommercial_kss_research_v1'
    inputs = check_manifest(spec_run/'input_manifest.json', allow_nonhuman=True,allow_licensed_research=licensed)
    by_id = {r['utterance_id']: r for r in inputs}
    handoff = json.loads((spec_run/'handoff_manifest.json').read_text())['utterances']
    validate_handoff(handoff)
    if {r['iterations'] for r in handoff} != {1, 10}:
        raise ValueError('Require both SPEC_1 and SPEC_10')
    if len(handoff) != 2*len(inputs):
        raise ValueError('Missing paired utterances')
    if {r['split'] for r in inputs} != {'train', 'validation', 'test'}:
        raise ValueError('Require fixed train/validation/test splits')
    if len({r['speaker_id'] for r in inputs}) != 1:
        raise ValueError('Initial paired trainer supports one target speaker')
    speaker = inputs[0]['speaker_id']
    folder = Path(tempfile.mkdtemp(prefix='paired_melo_experiment_', dir=ROOT/'runs'))
    cfg = training_config(assets)
    cfg['data']['spk2id'] = {speaker: 0}
    cfg['train']['fp16_run'] = False
    cfg['experiment'] = {'scope': 'synthetic_training_smoke' if all(r['input_kind']!='human_recording' for r in inputs) else 'human_recording',
        'source_spec_run': str(spec_run), 'input_kind': inputs[0]['input_kind'],
        'permission_declarations_validated': True, 'precision': 'bf16',
        'generator_initialization': str(Path(assets).resolve()/'model/checkpoint.pth'),
        'discriminator_initialization': 'seeded_random', 'duration_discriminator': True,
        'limitations': 'Bounded trainer using official Melo modules/losses; not official train.py execution.'}
    if licensed:
        cfg['experiment'].update(scope='kss_noncommercial_research',research_policy_validated=True,
            permission_declarations_validated=False,manual_listening_verified=False,
            separate_speaker_consent='not_verified',external_publication='not_approved_by_this_record')
    records = []
    for condition in ('clean', 'SPEC_1', 'SPEC_10'):
        target = folder/condition
        target.mkdir()
        metadata, cleaned, dataset = {}, {}, []
        for uid, row in sorted(by_id.items()):
            choice = next(r for r in handoff if r['utterance_id']==uid and r['iterations']==(10 if condition=='SPEC_10' else 1))
            role = 'clean' if condition=='clean' else 'protected'
            source = Path(choice[f'{role}_path'])
            out = target/row['split']
            out.mkdir(exist_ok=True)
            wav = out/f'{uid}.wav'
            shutil.copyfile(source, wav)
            if sha256(wav) != choice[f'{role}_sha256']:
                raise ValueError('Copy hash mismatch')
            feature_source = Path(row['original_path']).parent/row['features_path']
            feature_target = out/f'{uid}.features.pt'
            shutil.copyfile(feature_source, feature_target)
            import torch
            feature = torch.load(feature_target, map_location='cpu', weights_only=True)
            if feature['text'] != row['text']:
                raise ValueError('Feature transcript mismatch')
            metadata.setdefault(row['split'], []).append(f"{wav}|{speaker}|KR|{row['text']}\n")
            cleaned.setdefault(row['split'], []).append('|'.join([str(wav), speaker, 'KR', feature['normalized'],
                ' '.join(feature['phones']), ' '.join(map(str, feature['tones'])),
                ' '.join(map(str, feature['word2ph']))])+'\n')
            dataset.append({'utterance_id': uid, 'split': row['split'], 'text': row['text'],
                'wav': str(wav), 'wav_sha256': sha256(wav), 'features': str(feature_target),
                'features_sha256': sha256(feature_target), 'input_kind': row['input_kind'],
                'source_qc':row.get('automatic_qc'), 'transcript_verified':row.get('transcript_verified')})
        for split in metadata:
            (target/split/'metadata.list').write_text(''.join(metadata[split]), encoding='utf-8')
            (target/split/'metadata.list.cleaned').write_text(''.join(cleaned[split]), encoding='utf-8')
        current = copy.deepcopy(cfg)
        current['data']['training_files'] = str(target/'train/metadata.list.cleaned')
        current['data']['validation_files'] = str(target/'validation/metadata.list.cleaned')
        current['experiment'].update(condition=condition, dataset_manifest=str(target/'dataset.json'))
        save(target/'config.json', current)
        save(target/'dataset.json', {'utterances': dataset})
        records.append({'condition': condition, 'config': str(target/'config.json')})
    save(folder/'experiment.json', {'status': 'INPUTS_PREPARED_NOT_TRAINED', 'conditions': records,
         'split_counts': {s: sum(r['split']==s for r in inputs) for s in ('train','validation','test')},
         'source_spec_run': str(spec_run), 'scope': cfg['experiment']['scope']})
    return folder


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec-run', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    args = p.parse_args()
    print(prepare(args.spec_run, args.assets))
