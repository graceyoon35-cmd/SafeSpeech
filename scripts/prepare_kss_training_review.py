"""Prepare fixed KSS metadata/features/configs; keep all approvals unchanged.

Protected WAVs are explicitly pending. Configs carry a false execution gate;
this command never generates SPEC or trains a human voice.
"""
import argparse
import copy
import json
from pathlib import Path
import tempfile

import torch

from melo_training_core import KoreanFrontend, training_config
from run_korean_melo_spec import ROOT, save, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--bert',type=Path,required=True)
    args=p.parse_args()
    bundle=args.bundle.resolve()
    source=bundle/'spec_input.json'
    original_hash=sha256(source)
    manifest=json.loads(source.read_text())
    rows=manifest['utterances']
    summary=json.loads((bundle/'manifest.json').read_text())
    if len(rows)!=200 or len({r['utterance_id'] for r in rows})!=200:
        raise ValueError('Expected KSS 200 bundle')
    torch.set_num_threads(4)
    cfg=training_config(args.assets)
    cfg['train']['fp16_run']=False
    frontend=KoreanFrontend(args.bert,cfg)
    folder=Path(tempfile.mkdtemp(prefix='kss_training_review_',dir=ROOT/'runs'))
    features=folder/'features';features.mkdir()
    records=[]
    for i,row in enumerate(rows):
        original=(bundle/row['original_path']).resolve()
        if sha256(original)!=row['original_sha256']:
            raise ValueError('KSS WAV hash mismatch')
        feature=frontend.features(row['text'])
        path=features/(row['utterance_id']+'.features.pt')
        torch.save(feature,path)
        current=copy.deepcopy(row)
        current.update(original_path=str(original),features_path=str(path))
        records.append(current)
        if (i+1)%25==0:
            print(f'Prepared real Korean features {i+1}/200; approvals unchanged',flush=True)
    save(folder/'input_unreviewed.json',{'schema_version':1,'utterances':records})
    for condition in ('clean','SPEC_1','SPEC_10'):
        target=folder/condition;target.mkdir()
        dataset=[]
        for split in ('train','validation','test'):
            sub=target/split;sub.mkdir()
            raw=[];cleaned=[]
            for row in records:
                if row['split']!=split:
                    continue
                wav=Path(row['original_path']) if condition=='clean' else sub/(row['utterance_id']+'.wav')
                feature=torch.load(row['features_path'],map_location='cpu',weights_only=True)
                raw.append(f"{wav}|{row['speaker_id']}|KR|{row['text']}\n")
                cleaned.append('|'.join([str(wav),row['speaker_id'],'KR',feature['normalized'],
                    ' '.join(feature['phones']),' '.join(map(str,feature['tones'])),
                    ' '.join(map(str,feature['word2ph']))])+'\n')
                dataset.append({'utterance_id':row['utterance_id'],'split':split,'text':row['text'],
                    'wav':str(wav),'wav_sha256':row['original_sha256'] if condition=='clean' else None,
                    'features':row['features_path'],'features_sha256':sha256(row['features_path']),
                    'input_kind':'human_recording','protected_wav_pending':condition!='clean'})
            (sub/'metadata.list').write_text(''.join(raw),encoding='utf-8')
            (sub/'metadata.list.cleaned').write_text(''.join(cleaned),encoding='utf-8')
        current=copy.deepcopy(cfg)
        current['data']['spk2id']={'kss_f1':0}
        current['data']['training_files']=str(target/'train/metadata.list.cleaned')
        current['data']['validation_files']=str(target/'validation/metadata.list.cleaned')
        current['experiment']={'condition':condition,'scope':'KSS_REVIEW_PREPARATION_ONLY',
            'permission_declarations_validated':False,'dataset_manifest':str(target/'dataset.json'),
            'source_input_manifest':str(folder/'input_unreviewed.json'),
            'generator_initialization':str(args.assets.resolve()/'model/checkpoint.pth'),
            'precision':'bf16','execution_status':'BLOCKED_UNREVIEWED_AND_USE_UNCONFIRMED',
            'protected_audio_generated':False,'note':'Do not toggle this gate manually; run authorized workflow after documented review.'}
        save(target/'config.json',current)
        save(target/'dataset.json',{'utterances':dataset})
    if sha256(source)!=original_hash:
        raise ValueError('Source manifest changed')
    for old,new in zip(rows,records):
        for key in ('transcript_verified','research_use_permitted','permission_reference','permission_scope'):
            if old[key]!=new[key]:
                raise ValueError('Approval declaration changed')
    save(folder/'review_status.json',{'status':'METADATA_FEATURES_CONFIG_PREPARED_EXECUTION_BLOCKED',
        'source_manifest_sha256':original_hash,'source_manifest_unchanged':True,
        'usage_status':summary['dataset']['usage_status'],
        'transcript_verified_count':sum(r['transcript_verified'] is True for r in records),
        'research_use_permitted_count':sum(r['research_use_permitted'] is True for r in records),
        'wav_count':len(rows),'split_counts':summary['counts'],
        'protected_wavs_generated':0,'training_steps':0,
        'limitations':'Text frontend preprocessing only. No approval, listening, human SPEC or human finetuning.'})
    print(json.dumps({'directory':str(folder),'status':'EXECUTION_BLOCKED'}),flush=True)


if __name__=='__main__':
    main()
