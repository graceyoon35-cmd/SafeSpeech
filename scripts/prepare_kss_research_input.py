"""Reuse KSS auto validation/features without inventing listening or consent."""
import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf

from kss_research_policy import POLICY,LICENSE,SOURCE,CARD_SHA
from run_korean_melo_spec import ROOT,check_manifest,save,sha256


def prepare(prepared,bundle,asr):
    prepared,bundle,asr=map(lambda p:Path(p).resolve(),(prepared,bundle,asr))
    original=bundle/'spec_input.json'
    original_hash=sha256(original)
    source=json.loads(prepared.read_text(encoding='utf-8'))
    with asr.open(encoding='utf-8-sig',newline='') as stream:
        asr_rows={r['utterance_id']:r for r in csv.DictReader(stream)}
    features=json.loads((prepared.parent/'clean/dataset.json').read_text())['utterances']
    feature_hashes={r['utterance_id']:r['features_sha256'] for r in features}
    if len(source['utterances'])!=200 or set(asr_rows)!={r['utterance_id'] for r in source['utterances']}:
        raise ValueError('Expected matching complete KSS 200/ASR bundle')
    folder=Path(tempfile.mkdtemp(prefix='kss_licensed_research_input_',dir=ROOT/'runs'))
    evidence=ROOT/'runs/training_20261118/license_evidence/kss_mirror_card.md'
    policy={'name':POLICY,'basis':'public_dataset_license','license_url':LICENSE,'original_source_url':SOURCE,
        'noncommercial_research':True,'operations':['internal_spec','internal_tts_finetuning','internal_evaluation_synthesis'],
        'separate_speaker_consent':'not_verified','external_publication':'not_approved_by_this_record',
        'manual_listening_is_training_gate':False,'manual_listening_status':'unreviewed',
        'original_manifest':str(original),'original_manifest_sha256':original_hash,
        'license_evidence':{'path':str(evidence),'sha256':CARD_SHA},
        'mirror_revision':'48fdfd7ab1dbc1a62e4e8a8b9f4c360259d51d3c',
        'assessment':'Published noncommercial TTS purpose plus CC BY-NC-SA reproduction/adaptation basis; scoped research interpretation, not individual consent.',
        'user_scope':'2026-11-18 user instructed noncommercial university KSS research; automatic QC is sufficient to begin, listening remains quality review.'}
    rows=[];quality=[]
    for item in source['utterances']:
        row=copy.deepcopy(item);uid=row['utterance_id'];record=asr_rows[uid]
        if record['sha256']!=row['original_sha256'] or record['reference']!=row['text']:
            raise ValueError('ASR identity mismatch')
        if sha256(row['original_path'])!=row['original_sha256'] or sha256(row['features_path'])!=feature_hashes[uid]:
            raise ValueError('KSS WAV/feature integrity mismatch')
        wave,rate=sf.read(row['original_path'],dtype='float64')
        if rate!=44100 or wave.ndim!=1 or not np.isfinite(wave).all():raise ValueError('Invalid KSS waveform')
        rms=float(np.sqrt(np.mean(wave**2)));peak=float(np.abs(wave).max());quiet=float(np.mean(np.abs(wave)<.001))
        flags=[]
        if float(record['normalized_cer'])>0:flags.append('ASR_TRANSCRIPT_DIFFERENCE')
        if peak>=.999:flags.append('NEAR_CLIPPING')
        if rms<.001:flags.append('LOW_RMS')
        if quiet>.7:flags.append('HIGH_LOW_AMPLITUDE_FRACTION')
        row.update(features_sha256=feature_hashes[uid],manual_listening_status='unreviewed',
            automatic_qc={'wav_integrity':'PASS','asr_cer':float(record['normalized_cer']),
                'asr_exact_match':float(record['normalized_cer'])==0,'review_recommended':bool(flags),'flags':flags})
        rows.append(row)
        quality.append({'utterance_id':uid,'split':row['split'],'asr_cer':float(record['normalized_cer']),
            'rms':rms,'peak':peak,'low_amplitude_fraction':quiet,'flags':';'.join(flags),
            'training_included':True,'manual_listening_status':'unreviewed'})
    save(folder/'input.json',{'schema_version':1,'research_policy':policy,'utterances':rows})
    check_manifest(folder/'input.json',allow_licensed_research=True)
    for name,selected in [('automatic_quality.csv',quality),('review_recommended.csv',[r for r in quality if r['flags']])]:
        with (folder/name).open('x',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(quality[0]));writer.writeheader();writer.writerows(selected)
    with (folder/'excluded_utterances.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        csv.writer(stream).writerow(['utterance_id','split','reason'])
    save(folder/'input_audit.json',{'status':'AUTO_VALIDATED_FOR_SCOPED_RESEARCH','utterances':200,
        'split_counts':dict(Counter(r['split'] for r in rows)),'excluded_count':0,
        'asr_difference_count':sum(r['asr_cer']>0 for r in quality),'review_recommended_count':sum(bool(r['flags']) for r in quality),
        'manual_listening_verified_count':sum(r['transcript_verified'] is True for r in rows),
        'legacy_research_use_permitted_true_count':sum(r['research_use_permitted'] is True for r in rows),
        'original_manifest_unchanged':sha256(original)==original_hash,'research_policy':policy,
        'inclusion_rule':'All structurally valid 200 utterances retained. ASR differences are review recommendations, not automatic label-error findings.'})
    return folder/'input.json'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-input',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--asr',type=Path,required=True);args=p.parse_args()
    print(prepare(args.prepared_input,args.bundle,args.asr))
