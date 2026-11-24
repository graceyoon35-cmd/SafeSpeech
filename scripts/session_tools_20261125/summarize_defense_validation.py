"""Summarize only completed fresh attacks; no training, generation or GPU use."""
import csv
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from run_defense_validation import BASE,ROOT,LEGACY,TARGET,LABELS,sha,save,table


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def load(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def interval(values,rng):
    values=np.asarray(values,dtype=float)
    b=values[rng.integers(len(values),size=(10000,len(values)))].mean(axis=1)
    return dict(mean=float(values.mean()),ci95=np.quantile(b,[.025,.975]).tolist())


def edit(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(previous[j]+1,current[j-1]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]


def main():
    assert read(TARGET/'complete.json')['status']=='TWO_FRESH_ATTACKS_AND_EVALUATION_COMPLETE'
    assert not read(TARGET/'preservation.json')['changed']
    before=read(TARGET/'source_hashes_before.json')
    changed=[p for p,h in before.items() if sha(p)!=h];assert not changed
    evaluations=list((TARGET/'runs').glob('melo_smoke_evaluation_*/evaluation.json'));assert len(evaluations)==1
    result=read(evaluations[0]);assert result['status']=='EVALUATION_PASS'
    assert result['test_utterances']==30 and result['base_evaluation_seed']==2026
    measurements={r['condition']:r for r in result['measurements']}
    trained=read(TARGET/'training_summary.json')
    configs={r['condition']:read(Path(r['run'])/'config.json') for r in trained['training']}
    clean_cfg=configs['clean']
    for cfg in configs.values():
        assert cfg['train']==clean_cfg['train'] and cfg['model']==clean_cfg['model']
        a=dict(cfg['data']);b=dict(clean_cfg['data']);a.pop('training_files');b.pop('training_files');assert a==b
        assert cfg['experiment']['generator_initialization']==clean_cfg['experiment']['generator_initialization']
        assert cfg['experiment']['precision']==clean_cfg['experiment']['precision']=='bf16'
    rows=load(evaluations[0].parent/'per_file.csv');by={}
    import re
    normalize=lambda x:re.sub(r'[^0-9A-Za-z가-힣]','',x)
    for row in rows:
        assert sha(row['output'])==row['output_sha256'] and sha(row['reference'])==row['checkpoint_sha256']
        edits=sum(int(row[k]) for k in ('substitutions','deletions','insertions'))
        assert edit(normalize(row['text']),normalize(row['asr_transcript']))==edits
        assert abs(edits/len(normalize(row['text']))-float(row['cer']))<1e-12
        by.setdefault(row['condition'],{})[row['sentence_id']]=row
    assert len(by)==5 and all(len(a)==30 for a in by.values()) and len(rows)==150
    tests=set(by['clean'])
    for group in by.values():
        assert set(group)==tests
        for uid in tests:
            assert group[uid]['text']==by['clean'][uid]['text']
            assert group[uid]['evaluation_seed']==by['clean'][uid]['evaluation_seed']
    for label in LABELS:
        complete=read(TARGET/label/'complete.json');assert complete['status']=='FRESH_1000_STEP_ATTACK_PASS'
        training=read(complete['record']['report'])
        assert training['completed_steps']==1000 and training['start_step']==0
        assert {r['checkpoint_sha256'] for r in by[label].values()}=={training['checkpoint_sha256']}
        assert complete['record']['checkpoint'] not in {r['reference'] for r in by['clean'].values()} | {r['reference'] for r in by['SPEC_eps2_iter10'].values()}
    assert len({read(read(TARGET/label/'complete.json')['record']['report'])['checkpoint_sha256'] for label in LABELS})==2
    quality={r['condition']:r for r in read(BASE/'analysis/summary.json')}
    recheck=load(TARGET/'quality_recheck.csv')
    for label in LABELS:
        q=[r for r in recheck if r['condition']==label]
        assert len(q)==140 and all(r['strict_pcm16_pass']=='True' and r['clipping']=='0' for r in q)
        for key in ('snr_db','stoi'):
            assert abs(float(np.mean([float(r[key]) for r in q]))-quality[label][key])<1e-10
    baseline=measurements['SPEC_eps2_iter10'];clean=measurements['clean'];bq=quality['eps2_iter10']
    comparison=[]
    for label in ('clean','SPEC_eps2_iter10',*LABELS):
        q=quality['eps2_iter10' if label=='SPEC_eps2_iter10' else label];m=measurements[label]
        comparison.append(dict(condition=label,quality_n=140,test_n=30,snr_db=q['snr_db'],stoi=q['stoi'],
            sim=m['mean_sim'],mean_cer=m['mean_cer'],corpus_cer=m['corpus_cer'],silence_noise_rms=q['silence_noise_rms'],
            delta_snr_vs_eps2=q['snr_db']-bq['snr_db'] if q['snr_db'] is not None else None,
            delta_stoi_vs_eps2=q['stoi']-bq['stoi'],delta_sim_vs_eps2=m['mean_sim']-baseline['mean_sim'],
            delta_mean_cer_vs_eps2=m['mean_cer']-baseline['mean_cer'],
            delta_sim_vs_clean=m['mean_sim']-clean['mean_sim'],delta_mean_cer_vs_clean=m['mean_cer']-clean['mean_cer'],
            silence_rms_change_db_vs_eps2=20*math.log10(q['silence_noise_rms']/bq['silence_noise_rms']) if q['silence_noise_rms'] else None,
            baseline_clean_sim_reduction_retained=(clean['mean_sim']-m['mean_sim'])/(clean['mean_sim']-baseline['mean_sim']),
            fresh_attack_trained=label in LABELS,all_models_directly_evaluated_now=True))
    rng=np.random.default_rng(20261124);contrasts=[]
    for label in LABELS:
        c=dict(candidate=label)
        for ref in ('clean','SPEC_eps2_iter10'):
            c[ref]={metric:interval([float(by[label][uid][metric])-float(by[ref][uid][metric]) for uid in sorted(tests)],rng)
                for metric in ('sim','cer')}
        c['limited_defense_observed_vs_clean']=c['clean']['sim']['ci95'][1]<0 and c['clean']['cer']['ci95'][0]>0
        c['non_inferiority_to_eps2_established']=False
        contrasts.append(c)
    eligible=[c['candidate'] for c in contrasts if c['limited_defense_observed_vs_clean']]
    # Defense-first ranking among quality-improved candidates; no arbitrary equivalence threshold.
    choice=min(eligible,key=lambda label:measurements[label]['mean_sim']) if eligible else None
    analysis=dict(status='COMPLETED_MEASUREMENTS',comparison=comparison,paired_prompt_bootstrap=contrasts,
        provisional_balance_candidate=choice,selection_rule='Only candidates with paired SIM decrease and CER increase vs Clean (both 95% prompt-bootstrap intervals exclude zero); prioritize stronger observed SIM reduction among them. All candidates already have objectively reduced quiet residual. Not a non-inferiority test or human quality approval.',
        minimum_measured_quiet_rms_candidate=min(LABELS,key=lambda label:quality[label]['silence_noise_rms']),
        human_listening_verified=False,imperceptibility_established=False,baseline_defense_equivalence_established=False,
        limits='Single KSS speaker, seed52, bounded Melo trainer, fixed reused30 prompts. Candidate selected with these results, not independently validated. Bootstrap covers prompts only. No universal cloning-defense guarantee.',
        evaluation=str(evaluations[0]),new_attack_models=[read(TARGET/label/'complete.json')['record'] for label in LABELS],
        independent_cer_checks=len(rows),source_hash_changes=changed)
    save(TARGET/'results.json',analysis);table(TARGET/'comparison.csv',comparison)
    lines=['SafeSpeech 구간별 저잡음 후보: 신규 공격 학습·평가 결과',
        '완료: 두 후보 각각 새 공격 모델 1,000-step. 기존 Clean/eps2 모델도 직접 재평가.',
        '조건: KSS train140 / validation30 / test30, seed52, batch1, bf16, 동일 공식 초기 가중치.',
        '품질은 동일 train140개 평균, SIM/CER는 같은 test30문장 평균. CER corpus도 별도 기록.',
        '140개씩 동일 발화와 해시 재확인, 두 후보 모두 strict PCM16 epsilon2/255 통과, clipping0.',
        '', '조건 | SNR dB | STOI | SIM | 평균 CER | ΔSNR/ΔSTOI/ΔSIM/ΔCER (eps2 대비)']
    for r in comparison:
        num=lambda k:('해당없음' if r[k] is None else f'{r[k]:.6f}')
        lines.append(' | '.join([r['condition'],num('snr_db'),num('stoi'),num('sim'),num('mean_cer'),
            '/'.join(num(k) for k in ('delta_snr_vs_eps2','delta_stoi_vs_eps2','delta_sim_vs_eps2','delta_mean_cer_vs_eps2'))]))
    lines+=['','무음 RMS 최저 후보: '+analysis['minimum_measured_quiet_rms_candidate'],
        '제한된 방어 관찰이 있는 잠정 균형 후보: '+str(choice),
        '선정은 Clean 대비 SIM 감소/CER 증가의 paired 95% 구간과 관측 SIM을 우선했다. 실사용 승인 기준이 아니다.',
        '신규 후보의 실제 학습 모델로 측정했으며 기존 모델 지표를 신규 결과로 재사용하지 않았다.',
        'eps2와 동등한 방어/불가청/정량적 사람 청취 검증을 입증하지 않았다. 단일 화자·seed·공격·재사용30문장 한계.',
        '새 가중치, optimizer/RNG 상태, 로그, 합성 WAV 및 문장별SIM/CER는 이 폴더 아래 보존.',
        '기존 데이터/결과/체크포인트/백업에 삭제·덮어쓰기 없음. Git Commit/Push와 앱팀 브랜치 수정 없음.',
        '', 'paired bootstrap 상세:']
    for c in contrasts:lines.append(json.dumps(c,ensure_ascii=False))
    lines+=['','평가 결과: '+str(evaluations[0]),'발화별 품질 재검사: '+str(TARGET/'quality_recheck.csv'),
        '학습 및 평가 설정: '+str(TARGET/'training_summary.json'),
        '기존 비교 청취 파일: '+str(BASE/'listening'),'사람의 실제 청취 검수는 별도로 필요하다.']
    with (TARGET/'REPORT_KO.txt').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    winroot=subprocess.check_output(['wslpath','-w',str(ROOT)],text=True).strip()
    git=lambda *a:subprocess.check_output(['/mnt/c/Program Files/Git/cmd/git.exe','-C',winroot,*a],text=True).strip()
    branch=git('branch','--show-current');head=git('rev-parse','HEAD');status=git('status','--porcelain')
    assert branch=='research/perceptual-spec-20261124' and head=='954a06cde13065074328bc9476eaf8238845d1ea' and not status
    save(TARGET/'final_verification.json',dict(status='PASS',old_hash_changes=changed,fresh_models=2,steps_each=1000,
        independent_cer_checks=len(rows),matched_evaluation_prompts=len(tests),branch=branch,head=head,git_status=status,
        archived_and_new_training_config_parameters_match=True,
        no_commit_push=True,human_listening_verified=False))
    print(json.dumps(analysis,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
