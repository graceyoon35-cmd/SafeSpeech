"""Combine explicit human listening records and documented use scope, fail closed."""
import argparse
import copy
import csv
import json
from pathlib import Path
import tempfile

from run_korean_melo_spec import ROOT,check_manifest,save,sha256


def build(prepared_input,reviews,rights_record):
    prepared_input=Path(prepared_input).resolve();reviews=Path(reviews).resolve();rights_record=Path(rights_record).resolve()
    rights=json.loads(rights_record.read_text(encoding='utf-8-sig'))
    required=('noncommercial_research','spec_permitted','clean_protected_finetuning_permitted','internal_evaluation_synthesis_permitted')
    if rights.get('schema_version')!=1 or rights.get('dataset')!='KSS' or any(rights.get(k) is not True for k in required):
        raise ValueError('Actual KSS use-scope confirmation required; template defaults are not approval')
    if any(not isinstance(rights.get(k),str) or not rights[k].strip() for k in ('confirmed_by','confirmed_at','evidence_reference')):
        raise ValueError('Use-scope decision requires a reviewer, date and evidence reference')
    with reviews.open(encoding='utf-8-sig',newline='') as stream:
        decisions=list(csv.DictReader(stream))
    ids=[r['utterance_id'] for r in decisions]
    if len(ids)!=len(set(ids)):
        raise ValueError('Duplicate listening review IDs')
    by_id={r['utterance_id']:r for r in decisions}
    source=json.loads(prepared_input.read_text(encoding='utf-8-sig'))
    dataset=json.loads((prepared_input.parent/'clean/dataset.json').read_text())['utterances']
    feature_hashes={r['utterance_id']:r['features_sha256'] for r in dataset}
    selected=[]
    for original in source['utterances']:
        review=by_id.get(original['utterance_id'])
        if not review or review.get('heard_and_matches(예/아니오)')!='예':
            continue
        if review.get('review_status')!='MATCH_CONFIRMED_BY_REVIEWER' or any(not review.get(k) for k in ('checked_by','date')):
            raise ValueError('Listening match requires an explicit reviewer record')
        if review.get('audio_sha256')!=original['original_sha256'] or review.get('transcript')!=original['text']:
            raise ValueError('Reviewed audio/transcript identity mismatch')
        if sha256(original['features_path'])!=feature_hashes[original['utterance_id']]:
            raise ValueError('Prepared Korean features changed')
        row=copy.deepcopy(original)
        row.update(transcript_verified=True,
            transcript_reference=f"Manual listening: {reviews}; {review['checked_by']}; {review['date']}",
            research_use_permitted=True,permission_scope='spec_and_finetuning',
            permission_reference=f"Documented scope: {rights_record}; {rights['evidence_reference']}",
            features_sha256=feature_hashes[row['utterance_id']])
        selected.append(row)
    if not selected or {r['split'] for r in selected}!={'train','validation','test'}:
        raise ValueError('Need explicitly reviewed matching utterances in all three fixed splits')
    folder=Path(tempfile.mkdtemp(prefix='kss_authorized_input_',dir=ROOT/'runs'))
    save(folder/'input.json',{'schema_version':1,'utterances':selected})
    check_manifest(folder/'input.json')
    save(folder/'evidence.json',{'prepared_input_sha256':sha256(prepared_input),
        'review_csv_sha256':sha256(reviews),'rights_record_sha256':sha256(rights_record),
        'selected_count':len(selected),'originals_modified':False,
        'assessment':'Declaration/evidence binding only, not independent legal or listening authentication.'})
    return folder/'input.json'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared-input',type=Path,required=True)
    p.add_argument('--reviews',type=Path,required=True)
    p.add_argument('--rights-record',type=Path,required=True)
    args=p.parse_args()
    print(build(args.prepared_input,args.reviews,args.rights_record))
