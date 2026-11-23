"""Independent CER arithmetic and exact baseline/candidate protocol checks."""
import csv
import json
import math
from pathlib import Path
import re
from prepare_demo import HERE,OLD,save,sha

D=HERE/'defense_attack_140'
summary=json.loads((D/'training_summary.json').read_text())
base=json.loads((OLD/'runs/paired_melo_experiment_ong85l9i/clean/config.json').read_text())
data=json.loads((OLD/'runs/paired_melo_experiment_ong85l9i/clean/dataset.json').read_text())['utterances']
old_summary=json.loads((OLD/'runs/paired_melo_training_rhp4a8t2/summary.json').read_text())
old_clean=next(r for r in old_summary['training'] if r['condition']=='clean')
old_report=json.loads(Path(old_clean['report']).read_text())
protocol=[]
for r in summary['training']:
    if not r['condition'].startswith('SPEC_eps'):continue
    cfg=json.loads((Path(r['run'])/'config.json').read_text())
    report=json.loads(Path(r['report']).read_text())
    candidate=json.loads(Path(cfg['experiment']['dataset_manifest']).read_text())['utterances']
    assert cfg['train']==base['train'] and cfg['model']==base['model']
    assert {k:v for k,v in cfg['data'].items() if k!='training_files'}=={k:v for k,v in base['data'].items() if k!='training_files'}
    assert report['trainer_sha256']==old_report['trainer_sha256']
    assert report['completed_steps']==old_report['completed_steps']==1000
    assert report['initial_checkpoint_sha256']==old_report['initial_checkpoint_sha256']
    assert [x['utterance_id'] for x in report['losses']]==[x['utterance_id'] for x in old_report['losses']]
    assert sha(r['checkpoint'])==report['checkpoint_sha256']
    assert len(candidate)==len(data)==200
    for a,b in zip(data,candidate):
        assert all(a[k]==b[k] for k in ('utterance_id','split','text','features','features_sha256','input_kind'))
        if a['split']!='train':assert a['wav']==b['wav'] and a['wav_sha256']==b['wav_sha256']
        assert sha(b['wav'])==b['wav_sha256']
    protocol.append(dict(condition=r['condition'],same_train_model_and_data_parameters=True,same_trainer_sha256=True,
        same_train_order=True,same_initial_weights=True,same_1000_steps=True,held_out_data_unchanged=True,checkpoint_sha256=sha(r['checkpoint'])))

comp=json.loads((HERE/'comparison.json').read_text());evaluation=Path(comp['evaluation'])
evaluated=json.loads(evaluation.read_text())
with (evaluation.parent/'per_file.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
def norm(t):return re.sub(r'[^0-9A-Za-z가-힣]','',t)
def distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(previous[j]+1,current[-1]+1,previous[j-1]+int(x!=y)))
        previous=current
    return previous[-1]
groups={}
for r in rows:
    ref=norm(r['text']);hyp=norm(r['asr_transcript']);edits=distance(ref,hyp)
    assert len(ref)==int(r['reference_characters'])
    assert edits==sum(int(r[k]) for k in ('substitutions','deletions','insertions'))
    assert abs(edits/len(ref)-float(r['cer']))<1e-12
    groups.setdefault(r['condition'],[]).append((edits,len(ref)))
for m in evaluated['measurements']:
    g=groups[m['condition']]
    assert len(g)==m['test_count']==30
    assert abs(math.fsum(e/n for e,n in g)/len(g)-m['mean_cer'])<1e-12
    assert abs(sum(e for e,n in g)/sum(n for e,n in g)-m['corpus_cer'])<1e-12
assets=json.loads((HERE/'evaluation_assets_before.json').read_text())['sha256']
assert all(sha(p)==h for p,h in assets.items())
preserved=json.loads((D/'preservation.json').read_text())
assert not preserved['changed']
assert all(sha(p)==h for p,h in preserved['files'].items())
save(HERE/'final_verification.json',dict(status='PASS',protocol=protocol,independent_CER_checks=len(rows),
    mean_and_corpus_CER_verified=True,CER_not_capped_or_redecoded=True,
    evaluation_model_files_unchanged=len(assets),historical_source_files_unchanged=len(preserved['files']),
    scope='Checks read inputs/weights and numeric reports; does not establish human naturalness or universal cloning defense.'))
print(json.dumps(dict(status='PASS',new_attack_models=len(protocol),independent_CER_checks=len(rows),source_hash_changes=0)))
