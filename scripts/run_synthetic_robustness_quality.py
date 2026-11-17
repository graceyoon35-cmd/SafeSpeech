"""Paired channel transforms and STOI on held-out synthetic speech, no defense claim."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
from pystoi import stoi

from run_korean_melo_spec import ROOT,save,sha256,validate_handoff
from run_robustness_batch_b import run_batch
from run_robustness_sweep_b import default_conditions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec-run',type=Path,required=True)
    args=p.parse_args()
    rows=json.loads((args.spec_run/'handoff_manifest.json').read_text())['utterances']
    validate_handoff(rows)
    selected=[r for r in rows if r['split']=='test']
    if not selected or any(r['input_kind']!='model_synthesized' for r in selected):
        raise ValueError('Held-out synthesized inputs only')
    folder=Path(tempfile.mkdtemp(prefix='synthetic_robustness_quality_',dir=ROOT/'runs'))
    manifests=[];iterations={}
    for i,row in enumerate(selected):
        uid=f"synthetic_test_{row['utterance_id']}_spec{row['iterations']}"
        path=folder/f'pair_{i:03d}.json'
        save(path,{'schema_version':1,'experiment_id':uid,
            'audio':{'original':row['clean_path'],'protected':row['protected_path']},'conditions':default_conditions()})
        manifests.append(path);iterations[uid]=row['iterations']
    summary_path,batch=run_batch(manifests,folder,repeats=2)
    metrics=[]
    for pair in batch['pairs']:
        pair_summary=summary_path.parent/pair['summary']
        with (pair_summary.parent/'artifacts.csv').open(encoding='utf-8-sig',newline='') as stream:
            artifacts=list(csv.DictReader(stream))
        paired=defaultdict(dict)
        for record in artifacts:
            paired[(int(record['repeat']),int(record['condition_index']))][record['role']]=record
        for (repeat,index),roles in sorted(paired.items()):
            waves={};rates={}
            for role,record in roles.items():
                path=pair_summary.parent/record['wav_path']
                if sha256(path)!=record['wav_sha256']:
                    raise ValueError('Transformed audio hash mismatch')
                waves[role],rates[role]=sf.read(path,dtype='float64')
            if rates['original']!=rates['protected'] or waves['original'].shape!=waves['protected'].shape:
                raise ValueError('No implicit rate/length alignment allowed')
            score=float(stoi(waves['original'],waves['protected'],rates['original'],extended=False))
            if not np.isfinite(score):raise ValueError('Nonfinite STOI')
            metrics.append({'experiment_id':pair['experiment_id'],'iterations':iterations[pair['experiment_id']],
                'repeat':repeat,'condition_index':index,'parameters':json.dumps(batch['conditions'][index],sort_keys=True),
                'output_sample_rate':rates['original'],'paired_channel_stoi':score,
                'scope':'Synthetic speech quality after paired transforms; no retraining or defense efficacy'})
    with (folder/'per_condition_repeat.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    save(folder/'quality_summary.json',{'status':'PAIRED_SYNTHETIC_QUALITY_PASS','batch_summary':str(summary_path),
        'rows':metrics,'conditions':batch['conditions'],'wav_count':batch['wav_count'],'mp3_count':batch['mp3_count'],
        'distinct_original_utterances':len({r['utterance_id'] for r in selected}),
        'repeats_are_independent_samples':False,
        'limitations':'Same-seed paired transforms; STOI quality plumbing only, not trained attack robustness.'})
    print(json.dumps({'directory':str(folder),'status':'PAIRED_SYNTHETIC_QUALITY_PASS'}))


if __name__=='__main__':main()
