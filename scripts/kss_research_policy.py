"""Scoped public-license research validation, separate from listening/consent."""
import json
from pathlib import Path
from validate_melo_spec_pair import sha256

POLICY='licensed_noncommercial_kss_research_v1'
LICENSE='https://creativecommons.org/licenses/by-nc-sa/4.0/'
SOURCE='https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset'
CARD_SHA='bb69e3376bb806367367024f57d4513fb36eb93c8d6f0f19f80ada79ad4b7c82'


def validate_policy(payload):
    policy=payload.get('research_policy',{})
    if policy.get('name')!=POLICY or policy.get('basis')!='public_dataset_license':
        raise ValueError('Explicit KSS public-license research policy required')
    if policy.get('license_url')!=LICENSE or policy.get('original_source_url')!=SOURCE:
        raise ValueError('Incorrect KSS license/source')
    if policy.get('noncommercial_research') is not True or policy.get('operations')!=['internal_spec','internal_tts_finetuning','internal_evaluation_synthesis']:
        raise ValueError('Policy only covers internal noncommercial TTS research')
    if policy.get('separate_speaker_consent')!='not_verified' or policy.get('external_publication')!='not_approved_by_this_record':
        raise ValueError('Do not fabricate consent or external publication approval')
    evidence=policy['license_evidence']
    if evidence['sha256']!=CARD_SHA or sha256(evidence['path'])!=CARD_SHA:
        raise ValueError('Pinned public license evidence mismatch')
    source=Path(policy['original_manifest'])
    if sha256(source)!=policy['original_manifest_sha256']:
        raise ValueError('Original KSS manifest changed')
    original=json.loads(source.read_text(encoding='utf-8'))['utterances']
    by_id={r['utterance_id']:r for r in original}
    rows=payload['utterances']
    if len(rows)!=len(original) or {r['utterance_id'] for r in rows}!=set(by_id):
        raise ValueError('Research run must preserve the complete fixed KSS bundle')
    for row in rows:
        previous=by_id[row['utterance_id']]
        for key in ('text','split','original_sha256','input_kind','speaker_id','transcript_verified','research_use_permitted','permission_reference'):
            if row.get(key)!=previous.get(key):
                raise ValueError(f'Original KSS identity/declaration changed: {key}')
        if row['input_kind']!='human_recording' or row['speaker_id']!='kss_f1' or 'KSS' not in row['source']:
            raise ValueError('Scoped policy is only for the KSS dataset')
    return policy
