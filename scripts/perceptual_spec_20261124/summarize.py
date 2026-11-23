"""Separate low-noise ranking, observed defense, and perceptual approval."""
import csv
import json
import math
from pathlib import Path
import re
import numpy as np
from experiment_io import ROOT,OUT,save,sha,table

def loadcsv(path):
    with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))

def interval(delta,rng):
    a=np.asarray(delta,dtype=np.float64)
    boot=a[rng.integers(len(a),size=(10000,len(a)))].mean(axis=1)
    return dict(mean=float(a.mean()),ci95=np.quantile(boot,[.025,.975]).tolist())

def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(prev[j]+1,cur[-1]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]

def main():
    quality=json.loads((OUT/'quality/summary.json').read_text())
    perquality=loadcsv(OUT/'quality/per_file.csv')
    evals=list((OUT/'attacks_followup/runs').glob('melo_smoke_evaluation_*/evaluation.json'))
    assert len(evals)==1
    result=json.loads(evals[0].read_text());assert result['status']=='EVALUATION_PASS'
    rows=loadcsv(evals[0].parent/'per_file.csv');groups={}
    normalize=lambda t:re.sub(r'[^0-9A-Za-z가-힣]','',t)
    for r in rows:
        distance=edit_distance(normalize(r['text']),normalize(r['asr_transcript']))
        assert distance==sum(int(r[k]) for k in ('substitutions','deletions','insertions'))
        assert abs(distance/len(normalize(r['text']))-float(r['cer']))<1e-12
        groups.setdefault(r['condition'],{})[r['sentence_id']]=r
    contrasts=[];rng=np.random.default_rng(20261124)
    for candidate in ('eps1_iter5','mask_eps2_iter10_s2','mask_eps2_iter10_s05'):
        record=dict(candidate=candidate)
        for reference in ('clean','SPEC_eps2_iter10'):
            assert set(groups[candidate])==set(groups[reference]) and len(groups[candidate])==30
            record[reference]={metric:interval([float(groups[candidate][uid][metric])-float(groups[reference][uid][metric])
                for uid in sorted(groups[candidate])],rng) for metric in ('sim','cer')}
        against_clean=record['clean']
        record['limited_defense_observed']=against_clean['sim']['ci95'][1]<0 and against_clean['cer']['ci95'][0]>0
        record['point_estimates_weaker_than_baseline']=record['SPEC_eps2_iter10']['sim']['mean']>0 and record['SPEC_eps2_iter10']['cer']['mean']<0
        record['non_inferiority_to_baseline_established']=False
        contrasts.append(record)
    measurement={r['condition']:r for r in result['measurements']}
    demo={r['profile']:r for r in perquality if r['utterance_id']=='kss_hf_00078'}
    joint=[]
    for label in ('SPEC_eps2_iter10','eps1_iter5','mask_eps2_iter10_s2','mask_eps2_iter10_s05'):
        q=demo['eps2_iter10' if label=='SPEC_eps2_iter10' else label]
        m=measurement[label]
        joint.append(dict(profile=label,demo_snr_db=float(q['snr_db']),demo_stoi=float(q['stoi']),
            demo_quiet_residual_dbfs=float(q['quiet20_delta_dbfs']),mean_sim=m['mean_sim'],mean_cer=m['mean_cer'],
            corpus_cer=m['corpus_cer'],test_prompts=30,model_kind='archived_baseline_only' if label=='SPEC_eps2_iter10' else 'fresh_attack_training'))
    minnoise=min(quality,key=lambda r:r['delta_dbfs'])['profile']
    defended=[r['candidate'] for r in contrasts if r['limited_defense_observed']]
    eligible=[q for q in quality if q['profile'] in defended]
    choice=min(eligible,key=lambda q:q['quiet20_delta_dbfs'])['profile'] if eligible else 'SPEC_eps2_iter10'
    save(OUT/'results.json',dict(status='LIMITED_EXPERIMENT_COMPLETE_HUMAN_LISTENING_PENDING',
        minimum_measured_noise_candidate=minnoise,minimum_noise_defense_tested=minnoise in groups,
        newly_observed_defense_candidates=defended,combined_provisional_choice=choice,
        combined_success_confirmed=False,human_listening_verified=False,imperceptibility_established=False,
        listening_comparisons=str(OUT/'listening_comparisons/manifest.json'),
        selection_rule='Both paired prompt-bootstrap intervals vs Clean must exclude zero in the defense direction. This is not equivalence/non-inferiority to the archived eps2 baseline. Lowest quiet residual among qualifying new candidates is only a provisional audition choice.',
        joint=joint,contrasts=contrasts,quality_pilot=quality,independent_cer_checks=len(rows),
        reference='https://proceedings.mlr.press/v97/qin19a.html',
        pilot_split_overlap=dict(train_ids=['kss_hf_00078','kss_hf_03746'],test_audio_used_for_quality=['kss_hf_04258'],
            test_audio_used_for_attack_training=False,fully_unseen_model_selection_validation=False),
        adaptive_followup=True,independent_followup_validation=False,
        scope='3-file quality pilot; three new 140-utterance, 1000-step, seed52 Melo attacks. Third attack adaptively selected after first two outcomes. All models jointly re-evaluated on the same reused 30 prompts. Single speaker/seed/architecture; no calibrated masking or population audibility claim.',
        evaluator=str(evals[0]),source_files_changed=json.loads((OUT/'attacks/preservation.json').read_text())['changed']+json.loads((OUT/'attacks_followup/preservation.json').read_text())['changed']))
    table(OUT/'joint_comparison.csv',joint)
    verify_listening(perquality)
    print(json.dumps(dict(minimum_noise=minnoise,new_defense=defended,provisional=choice,joint=joint,contrasts=contrasts),ensure_ascii=False,indent=2))

def verify_listening(rows):
    import soundfile as sf
    from scipy.io import wavfile
    checked=0
    for row in rows:
        if row['profile']=='archived_eps2_iter10':continue
        p=Path(row['path']);original=p.parent.parent/'original.wav'
        x,sr=sf.read(original,dtype='int16');y,yr=sf.read(p,dtype='int16')
        ab,ar=sf.read(p.parent/'AB_original_then_protected.wav',dtype='int16')
        assert sr==yr==ar==44100 and np.array_equal(ab[:len(x)],x) and np.array_equal(ab[len(x)+26460:],y)
        assert not np.count_nonzero(ab[len(x):len(x)+26460])
        independent_rate,codes=wavfile.read(p)
        assert independent_rate==sr and np.array_equal(codes,y)
        delta,_=sf.read(p.parent/'residual_original_amplitude_FLOAT.wav')
        assert np.array_equal(delta,(y.astype(np.float64)-x.astype(np.float64))/32768)
        checked+=1
    save(OUT/'listening_verification.json',dict(status='PASS',candidate_pairs=checked,source_segments_exact=True,
        original_gain_retained=True,independent_decoder_samples_exact=True,residuals_exact=True,human_listening='PENDING'))

if __name__=='__main__':main()
