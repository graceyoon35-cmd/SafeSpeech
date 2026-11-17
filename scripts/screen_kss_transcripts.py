"""ASR-assisted listening-review priority; never approve a KSS transcript."""
import argparse
import csv
import json
from pathlib import Path
import tempfile

import jiwer
import torch
import whisper

from evaluate_paired_melo_smoke import normalize
from run_korean_melo_spec import ROOT, save, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    args=p.parse_args()
    manifest=args.bundle/'spec_input.json';before=sha256(manifest)
    rows=json.loads(manifest.read_text())['utterances']
    folder=Path(tempfile.mkdtemp(prefix='kss_asr_screening_',dir=ROOT/'runs'))
    torch.set_num_threads(4)
    model=whisper.load_model('small',device='cuda',download_root='/home/safespeech/models/melo-evaluation-20261118/whisper')
    results=[]
    for i,row in enumerate(rows):
        path=(args.bundle/row['original_path']).resolve()
        if sha256(path)!=row['original_sha256']:
            raise ValueError('KSS original hash mismatch')
        output=model.transcribe(str(path),language='ko',temperature=0,fp16=True,condition_on_previous_text=False)
        ref=normalize(row['text']);hyp=normalize(output['text'])
        if not ref:
            raise ValueError('Empty original transcript')
        score=jiwer.cer(ref,hyp)
        results.append({'utterance_id':row['utterance_id'],'split':row['split'],
            'wav':str(path),'sha256':row['original_sha256'],'reference':row['text'],
            'asr_hypothesis':output['text'].strip(),'normalized_cer':score,
            'transcript_verified':row['transcript_verified'],
            'research_use_permitted':row['research_use_permitted'],
            'review_status':'UNREVIEWED_ASR_SCREENING_ONLY'})
        if sha256(path)!=row['original_sha256']:
            raise ValueError('KSS source changed during screening')
        if (i+1)%20==0:
            print(f'ASR screened {i+1}/{len(rows)}; all review flags preserved',flush=True)
    ranked=sorted(results,key=lambda r:r['normalized_cer'],reverse=True)
    with (folder/'listening_priority.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['priority']+list(ranked[0]));writer.writeheader()
        writer.writerows({'priority':i+1,**row} for i,row in enumerate(ranked))
    if sha256(manifest)!=before:
        raise ValueError('KSS manifest changed')
    save(folder/'summary.json',{'status':'ASR_SCREENING_COMPLETED_NOT_LISTENING_REVIEW',
        'count':len(results),'whisper_model':'small','language':'ko',
        'source_manifest_unchanged':True,'source_manifest_sha256':before,
        'mean_normalized_cer':sum(r['normalized_cer'] for r in results)/len(results),
        'exact_normalized_matches':sum(r['normalized_cer']==0 for r in results),
        'transcript_verified_count':sum(r['transcript_verified'] is True for r in results),
        'limitations':'Whisper errors do not prove bad labels; exact ASR matches do not constitute listening review. No permission flags changed.'})
    print(json.dumps({'directory':str(folder),'status':'ASR_SCREENING_ONLY'}),flush=True)


if __name__=='__main__':
    main()
