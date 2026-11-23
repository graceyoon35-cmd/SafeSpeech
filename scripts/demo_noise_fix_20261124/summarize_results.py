"""Join actual waveform quality with newly trained attack results; never infer missing metrics."""
import csv
import json
from pathlib import Path
import numpy as np
from prepare_demo import HERE,OLD,save,sha

D=HERE/'defense_attack_140'
evals=list((D/'runs').glob('melo_smoke_evaluation_*/evaluation.json'))
if len(evals)!=1:raise RuntimeError('Expected one complete, isolated evaluation')
evaluation=json.loads(evals[0].read_text())
assert evaluation['status']=='EVALUATION_PASS' and evaluation['test_utterances']==30
with (evals[0].parent/'per_file.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
by={}
for r in rows:by.setdefault(r['condition'],{})[r['sentence_id']]=r
assert all(len(v)==30 for v in by.values())
baseline=by['clean'];pairs=[]
rng=np.random.default_rng(20261124)
for c in ('SPEC_eps2_iter10','SPEC_eps4_iter10','SPEC_10'):
    assert set(by[c])==set(baseline)
    record=dict(condition=c,n=30,reference='archived clean checkpoint, same training seed/steps and fresh same-code evaluation',
        scope='Single speaker and single training seed. Bootstrap only across prompts; not independence across speakers/seeds/attacks.')
    for metric in ('sim','cer'):
        delta=np.array([float(by[c][uid][metric])-float(baseline[uid][metric]) for uid in sorted(baseline)])
        means=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
        record[metric+'_difference_mean']=float(delta.mean())
        record[metric+'_paired_prompt_bootstrap_ci95']=np.quantile(means,[.025,.975]).tolist()
    record['direction_observed']=record['sim_difference_mean']<0 and record['cer_difference_mean']>0
    record['both_prompt_intervals_exclude_zero_in_defense_direction']=record['sim_paired_prompt_bootstrap_ci95'][1]<0 and record['cer_paired_prompt_bootstrap_ci95'][0]>0
    record['cer_over_100_percent_count']=sum(float(r['cer'])>1 for r in by[c].values())
    record['max_cer']=max(float(r['cer']) for r in by[c].values())
    pairs.append(record)

with (HERE/'demo_quality.csv').open(encoding='utf-8-sig') as f:demo=list(csv.DictReader(f))
demo_by={r['condition']:r for r in demo}
mapping={'SPEC_eps2_iter10':'02_candidate_eps2_iter10','SPEC_eps4_iter10':'03_candidate_eps4_iter10','SPEC_10':'01_existing_profile_eps8_iter10'}
quality={}
for c in ('SPEC_eps2_iter10','SPEC_eps4_iter10'):
    q=json.loads((D/c/'quality.json').read_text())
    quality[c]={k:q[k] for k in ('n','mean_snr_db','mean_stoi','clipping')}
# Historical eps8 model used the unprojected old training dataset, not today's strict demo WAV.
ids={r['utterance_id'] for r in json.loads((D/'SPEC_eps2_iter10/quality.json').read_text())['per_file']}
with (HERE.parent/'audio_quality_diagnosis_20261123/historical_files.csv').open(encoding='utf-8-sig') as f:
    old=[r for r in csv.DictReader(f) if r['condition']=='SPEC_10_historical' and r['utterance_id'] in ids]
assert len(old)==140
quality['SPEC_10']=dict(n=140,mean_snr_db=float(np.mean([float(r['snr_db']) for r in old])),
    mean_stoi=float(np.mean([float(r['stoi']) for r in old])),clipping=sum(int(r['rail_samples']) for r in old),
    source='Archived eps8 unprojected inputs actually used to train the archived baseline. Distinct from today strict demo output.')

comparison=[]
for row in evaluation['measurements']:
    r=dict(row)
    if r['condition'] in mapping:
        q=demo_by[mapping[r['condition']]]
        r.update(demo_snr_db=float(q['snr_db']),demo_stoi=float(q['stoi']),demo_clipping=int(q['clipping']),
            demo_path=q['path'],training_input_quality=quality[r['condition']],
            checkpoint_kind='newly_trained_on_this_candidate' if r['condition']!='SPEC_10' else 'archived_eps8_only')
    comparison.append(r)
save(HERE/'comparison.json',dict(status='LIMITED_ATTACK_COMPARISON_COMPLETE',evaluation=str(evals[0]),
    evaluation_sha256=sha(evals[0]),rows=comparison,paired_differences=pairs,
    defense_scope='Same Melo architecture surrogate and attack, 140 KSS training utterances, 1000 steps, train seed52, 30 held-out prompts. Clean/eps8 archived references; eps2/eps4 fresh attack training. Not universal anti-cloning evidence.',
    human_listening_verified=False,cer_is_not_defense_success_rate=True,output_quality_is_not_defense_metric=True))
flat=[]
for r in comparison:
    flat.append({k:r.get(k) for k in ('condition','demo_snr_db','demo_stoi','demo_clipping','mean_sim','mean_cer','corpus_cer','test_count','checkpoint_kind')})
with (HERE/'comparison.csv').open('x',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
save(HERE/'defense_completion.json',dict(status='NEW_CANDIDATE_ATTACKS_AND_EVALUATION_COMPLETE',
    old_checkpoints_used_only_as_their_own_baselines=True,fresh_candidates=['SPEC_eps2_iter10','SPEC_eps4_iter10'],
    models_trained_per_new_candidate=1,steps=1000,train_utterances=140,test_prompts=30,
    evaluation_settings='Unchanged original evaluator settings, ECAPA fixed five clean references; Whisper small Korean, temperature0, fp16; original text normalization and seed2026+i.',
    warning='High CER may include repeated ASR insertions; never interpret CER as blocking probability. Human naturalness, other speakers/attackers and seed repeats remain unverified.'))
print(json.dumps(dict(rows=flat,paired_differences=pairs),ensure_ascii=False,indent=2))
