"""Preserve primary CER and audit all outputs with one fixed short-clip decoder."""
import argparse
import csv
import json
from pathlib import Path

import jiwer
import numpy as np
import torch
import whisper

from evaluate_paired_melo_smoke import normalize
from run_korean_melo_spec import save,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);args=p.parse_args()
    with (args.results/'kss_per_file.csv').open(encoding='utf-8-sig',newline='') as stream:
        original=list(csv.DictReader(stream))
    torch.set_num_threads(4)
    model=whisper.load_model('small',device='cuda',download_root='/home/safespeech/models/melo-evaluation-20261118/whisper')
    audited=[]
    for i,row in enumerate(original):
        if sha256(row['output'])!=row['output_sha256']:raise ValueError('Evaluation WAV changed')
        result=model.transcribe(row['output'],language='ko',temperature=0,fp16=True,
            condition_on_previous_text=False,sample_len=64)
        ref=normalize(row['text']);hyp=normalize(result['text']);edits=jiwer.process_characters(ref,hyp)
        audited.append({'condition':row['condition'],'sentence_id':row['sentence_id'],'primary_cer':float(row['cer']),
            'audited_cer':edits.cer,'audited_transcript':result['text'].strip(),
            'substitutions':edits.substitutions,'deletions':edits.deletions,'insertions':edits.insertions,
            'reference_characters':len(ref),'output_sha256':row['output_sha256']})
        if (i+1)%30==0:print(f'Decoder sensitivity audited {i+1}/{len(original)}',flush=True)
    with (args.results/'kss_cer_decoder_audit_per_file.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(audited[0]));w.writeheader();w.writerows(audited)
    summary=[]
    for condition in ('pretrained','clean','SPEC_10','SPEC_1'):
        rows=[r for r in audited if r['condition']==condition]
        primary=np.array([r['primary_cer'] for r in rows]);alternate=np.array([r['audited_cer'] for r in rows])
        summary.append({'condition':condition,'n':len(rows),'primary_mean_cer':float(primary.mean()),
            'primary_median_cer':float(np.median(primary)),'primary_cer_above_100pct_count':int(np.sum(primary>1)),
            'primary_max_cer':float(primary.max()),'short_clip_decoder_mean_cer':float(alternate.mean()),
            'short_clip_decoder_median_cer':float(np.median(alternate)),
            'short_clip_decoder_corpus_cer':sum(r['substitutions']+r['deletions']+r['insertions'] for r in rows)/sum(r['reference_characters'] for r in rows)})
    with (args.results/'kss_cer_decoder_audit_summary.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    save(args.results/'kss_cer_decoder_audit.json',{'status':'ALL_120_OUTPUTS_AUDITED','primary_results_modified':False,
        'asr_model':'Whisper small','alternative_settings':{'sample_len':64,'condition_on_previous_text':False,'language':'ko','temperature':0},
        'scope':'Uniform post-hoc short-clip decoder sensitivity after observing insertion outliers; not a replacement for primary scores.',
        'summary':summary,'notes':['CER=(substitutions+deletions+insertions)/reference characters and can exceed 100%.',
            'Large insertion counts on poor speech can reflect ASR hallucinations; do not report CER as attack success rate.',
            'No samples removed; same settings applied to every condition.']})
    print(json.dumps({'status':'ALL_120_OUTPUTS_AUDITED','summary':summary}),flush=True)


if __name__=='__main__':main()
