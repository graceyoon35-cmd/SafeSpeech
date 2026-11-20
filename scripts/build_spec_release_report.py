"""Verify the November 20 release evidence and build measured tables and figures."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_kss_final import digest, distance, load, normalize, table, write
from run_korean_melo_spec import ROOT


def read_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def verify_evaluation(folder, baseline):
    metadata = load(folder/'evaluation.json')
    assert metadata['status'] == 'EVALUATION_PASS'
    rows = read_rows(folder/'per_file.csv')
    original = {r['sentence_id']: r for r in read_rows(baseline/'per_file.csv') if r['condition']=='pretrained'}
    outliers = []
    for r in rows:
        b = original[r['sentence_id']]
        assert r['text']==b['text'] and r['evaluation_seed']==b['evaluation_seed']
        assert digest(r['output'])==r['output_sha256']
        a,h = normalize(r['text']),normalize(r['asr_transcript'])
        edits = distance(a,h)
        assert abs(edits/len(a)-float(r['cer']))<1e-12
        assert edits==sum(int(r[k]) for k in ('substitutions','deletions','insertions'))
        assert len(a)==int(r['reference_characters'])
    for m in metadata['measurements']:
        selected = [r for r in rows if r['condition']==m['condition']]
        assert len(selected)==30
        for metric in ('sim','cer'):
            assert abs(np.mean([float(r[metric]) for r in selected])-m['mean_'+metric])<1e-12
        corpus = sum(distance(normalize(r['text']),normalize(r['asr_transcript'])) for r in selected)/sum(len(normalize(r['text'])) for r in selected)
        assert abs(corpus-m['corpus_cer'])<1e-12
        largest = sorted(selected,key=lambda r:float(r['cer']),reverse=True)[:3]
        outliers.append({'condition':m['condition'],'cer_over_100_percent':sum(float(r['cer'])>1 for r in selected),
                        'insertions':sum(int(r['insertions']) for r in selected),
                        'largest':[{'sentence_id':r['sentence_id'],'cer':float(r['cer']),
                           'insertions':int(r['insertions']),'reference_characters':int(r['reference_characters']),
                           'transcript_preview':r['asr_transcript'][:180]} for r in largest]})
    return {'status':'PASS','n':len(rows),'aggregates_and_independent_edit_distance':'PASS','outliers':outliers}


def build(validation):
    old = ROOT/'runs/final_validation_20261119'
    out = validation/'presentation'
    out.mkdir(exist_ok=False)
    strict_location = load(validation/'strict_experiment/evaluation_location.json')
    strict_folder = Path(strict_location['directory'])
    baseline = ROOT/'runs/melo_smoke_evaluation_qyh7qbgy'
    checks = {}
    for label,folder in [('seed53',ROOT/'runs/melo_smoke_evaluation_5am0uct6'),('strict',strict_folder)]:
        checks[label] = verify_evaluation(folder,baseline)
        sim = load(validation/(label+'_sim_recomputed/summary.json'))
        assert sim['status']=='PASS'
        checks[label]['sim_reencoding']=sim
    provenance=load(validation/'strict_experiment/input_provenance.json')
    a,b=provenance['source_config'],provenance['actual_config']
    assert a['train']==b['train'] and a['model']==b['model']
    assert a['train']['seed']==52 and len(provenance['files'])==200
    for r in provenance['files']:
        assert digest(r['old_wav'])==r['old_sha256'] and digest(r['new_wav'])==r['new_sha256']
    training=load(validation/'strict_experiment/summary.json')
    for r in training['training']:
        report=load(r['report'])
        assert report['completed_steps']==1000 and digest(r['checkpoint'])==report['checkpoint_sha256']
        assert report['initial_checkpoint_sha256']==training['initial_checkpoint_sha256']
    audit=load(old/'audit_strict/audit.json')
    changed=[p for p,h in audit['hashes'].items() if digest(p)!=h]
    assert not changed
    checks['preservation']={'status':'PASS','files':len(audit['hashes']),'changed':changed}
    write(validation/'final_verification.json',checks)
    results=read_rows(old/'presentation/final_results.csv')
    for r in results:
        r['input_version']='historical'
    measured=load(strict_folder/'evaluation.json')
    m=next(r for r in measured['measurements'] if r['condition']=='SPEC_10_strict')
    q=load(old/'strict_corrected_inputs/summary.json')
    results.append({'training_seed':52,'condition':'SPEC_10_strict','sim':m['mean_sim'],
        'macro_cer_percent':100*m['mean_cer'],'corpus_cer_percent':100*m['corpus_cer'],
        'short_clip_macro_cer_percent':'not evaluated','input_stoi':q['mean_stoi'],
        'input_snr_db':q['mean_snr_db'],'metric_scope':'Corrected input n=200; synthesis n=30; one new training run',
        'checkpoint':next(r['checkpoint'] for r in training['training'] if r['condition']=='SPEC_10_strict'),
        'input_version':'strict integer projection'})
    table(out/'final_results.csv',results)
    quality=[]
    for r in audit['quality_summary']:
        quality.append({'condition':r['condition'],'n':r['n'],'stoi':r['mean_stoi'],'snr_db':r['mean_snr_db'],
                        'max_linf':r['max_linf'],'strict_epsilon_failures':r['strict_epsilon_exceedances']})
    quality.append({'condition':'SPEC_10_strict','n':q['n'],'stoi':q['mean_stoi'],'snr_db':q['mean_snr_db'],
                    'max_linf':q['max_linf'],'strict_epsilon_failures':q['strict_epsilon_failures']})
    table(out/'input_quality.csv',quality)
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    labels=[f"{r['condition']}\nseed {r['training_seed']}" for r in results]
    colors=['#487f9c','#78954a','#b87945','#487f9c','#b87945','#8963a8']
    for ax,key,title in [(axes[0,0],'sim','Cloned synthesis: mean ECAPA cosine SIM'),
                         (axes[0,1],'macro_cer_percent','Cloned synthesis: macro CER (%)')]:
        values=[float(r[key]) for r in results]
        ax.bar(range(len(results)),values,color=colors)
        ax.set_xticks(range(len(results)),labels,rotation=20,ha='right',fontsize=8)
        ax.set_title(title)
        ax.grid(axis='y',alpha=.2)
        for i,value in enumerate(values):
            ax.annotate(f'{value:.3f}' if key=='sim' else f'{value:.1f}',(i,value),ha='center',va='bottom',fontsize=8)
        ax.margins(y=.15)
    names=[r['condition'] for r in quality]
    axes[1,0].bar(names,[r['stoi'] for r in quality],color=['#78954a','#b87945','#8963a8'])
    axes[1,0].set(title='Protected inputs: STOI (200 WAV pairs each)',ylim=(0,1.08))
    axes[1,1].bar(names,[r['snr_db'] for r in quality],color=['#78954a','#b87945','#8963a8'])
    axes[1,1].set(title='Protected inputs: SNR (dB)')
    fig.suptitle('KSS single speaker | 1,000 steps | strict epsilon correction evaluated separately')
    for ext in ('png','svg'):
        fig.savefig(out/('spec_release_results.'+ext),dpi=160)
    plt.close(fig)
    same=m['mean_sim']<float(results[0]['sim']) and m['mean_cer']>float(results[0]['macro_cer_percent'])/100
    smoke=load(validation/'wav_smoke_checks.json')
    assert smoke['status']=='PASS'
    tests=(validation/'tests.log').read_text()
    assert 'Ran 99 tests' in tests and '\nOK\n' in tests
    assert 'No broken requirements found' in (validation/'pip_check.log').read_text()
    md=['# SPEC 연구 개발 최종 보고서 — 2026-11-20','',
        '생성·학습·방어 성능 평가 및 앱팀 오프라인 인계 준비를 완료했다. 11월 19일 보고서의 보정본 SIM·CER 미측정 항목은 아래 별도 실험으로 갱신한다. 기존 실험·체크포인트·백업은 보존했다.','',
        '## 검증된 성과','',
        '- KSS 200개, 분할 140/30/30, 단일 화자, 동일 초기 생성기, batch 1, BF16, segment 8192, AdamW 0.0001, 1,000-step 제한 실험이다.',
        '- seed53 기존 평가 90건과 보정본 평가 90건의 WAV 해시·문장·합성 seed·독립 편집거리·macro 및 corpus CER를 검산했다. 두 평가의 SIM도 ECAPA 재인코딩으로 검산했다.',
        f"- 기존 증거 {len(audit['hashes'])}개 파일의 SHA256이 유지됐다. 신규 보정본만 별도 학습했고 Clean은 기존 seed52 체크포인트를 재사용했다.",'',
        '## 최종 복제 합성 평가','',
        'SIM은 고정된 clean 학습 음성 5개에 대한 ECAPA cosine 평균이다. CER는 30개 테스트 발화별 비율의 평균(macro), corpus CER는 문자 수 가중 전체 비율이다. 낮은 SIM·높은 CER는 본 복제 실험에서 방어 방향이지만 차단 성공률은 아니다.','',
        '| 학습 seed | 입력 조건 | SIM | macro CER | corpus CER |','|---:|---|---:|---:|---:|']
    for r in results:
        md.append(f"| {r['training_seed']} | {r['condition']} | {float(r['sim']):.7f} | {float(r['macro_cer_percent']):.4f}% | {float(r['corpus_cer_percent']):.4f}% |")
    md += ['', '## ε 보정 전후의 차이','',
        '기존 SPEC_10은 PCM16 저장 반올림으로 200개 모두 엄격한 ε=8/255를 최대 1 LSB 이내 초과했다. ε+1/32768를 허용하던 기존 검증에서는 통과했다. 원본 파일은 수정하지 않았다. 정수 투영 보정본은 원본 PCM16 대비 floor(ε×32768)=1028 code 이내로 제한한다.','',
        '| 보호 입력 | 개수 | 평균 STOI | 평균 SNR (dB) | 최대 진폭 차이 | 엄격한 ε 초과 |','|---|---:|---:|---:|---:|---:|']
    for r in quality:
        md.append(f"| {r['condition']} | {r['n']} | {r['stoi']:.7f} | {r['snr_db']:.7f} | {r['max_linf']:.15f} | {r['strict_epsilon_failures']} |")
    md += ['', '보정본은 섭동을 새로 최적화한 결과가 아니라 기존 SPEC_10의 저장 단계 정수 투영 사본이다. seed52·분할·텍스트 특징·초기 가중치·모델·학습 설정을 유지하고 WAV 경로·해시, manifest/list 경로, 조건 이름만 바꾸었다. 같은 30문장과 합성 seed 2026~2055, Whisper small 기본 한국어 decoder를 사용했다.','',
        f"보정 후 Clean 대비 방어 방향(낮은 SIM·높은 CER) 동시 충족: **{'관측됨' if same else '관측되지 않음'}**. 보정본 SIM {m['mean_sim']:.7f}, macro CER {100*m['mean_cer']:.4f}%. 한 번의 추가 학습이므로 기존 대비 차이를 저장 보정만의 인과 효과로 해석할 수 없다. CUDA 학습 변동이 함께 포함된다.",'',
        '## ASR 민감도와 GPU 재현 범위','',
        'CER 100% 초과는 반복 전사의 삽입 오류를 포함한 유효한 편집거리 비율이다. 주 결과에서 잘라내거나 상한 처리하지 않았다. 기존 seed52 SPEC_10의 짧은 발화 decoder 민감도 결과는 174.0339% → 115.3416%다. 이는 사후 별도 분석이며 주 결과를 대체하지 않는다. seed53 및 보정본에는 이 추가 decoder 분석을 수행하지 않았다. 발화별 삽입 수·최대 CER 전사 미리보기는 final_verification.json에 보존했다.','',
        '기존 동일 seed GPU 보호 두 실행은 76,205샘플이 달랐고 출력 간 최대 차이는 0.050201416015625였다. 각 출력은 입력 대비 엄격한 ε를 만족했다. 강제 결정론은 reflection_pad1d_backward_out_cuda의 결정론 구현 부재로 실패했다. 해당 연산을 유일 원인으로 단정하지 않는다. 실패 로그를 보존했고 추가 장시간 해결을 시도하지 않았다. 같은 seed는 바이트 동일성 보장이 아니다. 재현 범위는 이 고정 환경·동일 화자·짧은 제한 학습에서 방어 지표 방향을 확인한 수준이다.','',
        '## 테스트와 앱팀 전달','',
        '- 2026-11-20 전체 자동 테스트 99개 PASS, pip check PASS.',
        f"- 실제 WAV→보호 WAV GPU 실행 PASS: {smoke['result']['samples']:,}샘플, mono 44,100Hz PCM16, 최대 차이 {smoke['result']['strict_linf']}, STOI {smoke['result']['stoi']:.7f}, SNR {smoke['result']['snr_db']:.6f}dB.",
        '- 최종 저장 WAV를 독립 재계산했고 입력 해시 보존, 기존 출력 폴더 재사용 거부(exit 1), 기존 결과 해시 보존을 확인했다.',
        '- 실행·의존성·입출력·예제 경로: [앱팀 README](README_SPEC_HANDOFF_20261120_KO.md). UI·프론트엔드는 앱팀 범위다.','',
        '## 한계','',
        '- 단일 화자, 두 원래 학습 seed, 보정본 학습 seed 하나, 단일 대리·복제 모델, 1,000-step 제한이다. 장기 수렴·다화자·다른 공격 모델로 일반화하지 않는다.',
        '- 사람 청취 검수 미완료, 사전학습 코퍼스 비중복 미확인이다. STOI/SNR은 보호 입력, SIM/CER는 복제 합성이므로 같은 품질 지표로 섞지 않는다.',
        '- SPEC_10의 보호 입력 STOI 약 0.902와 SNR 약 11.79dB는 품질 비용을 보인다. 무손실·불가청·복제 불가능으로 발표하지 않는다.','',
        '## 미해결 사항','',
        '- GPU 바이트 동일성 및 강제 결정론 실행 실패.',
        '- 독립 섭동 seed 반복, SPEC_1 추가 학습 seed, 보정본 두 번째 학습 seed, 보정본 decoder 민감도는 미실행.',
        '- CPU 실행 성능 및 새 기기의 전체 신규 설치는 미검증. 앱 UI·비동기 실행·배포 연결은 앱팀 담당.',
        '- 별도 화자 동의·음성/가중치 외부 공개 승인을 확보한 것으로 기록하지 않는다. 기존 KSS 내부 비상업적 연구 범위를 유지한다.','',
        '## 발표 그래프','', '![최종 SPEC 지표](figures/spec_release_results_20261120.svg)','',
        'CSV·PNG·SVG 및 재검산 색인은 runs/final_validation_20261120/presentation/에 있다. Git·백업 인계 상태는 SPEC_RELEASE_STATUS_20261120_KO.md에 별도로 기록한다.','']
    with (out/'SPEC_FINAL_REPORT_20261120_KO.md').open('x',encoding='utf-8') as stream:
        stream.write('\n'.join(md))
    write(out/'report.json',{'status':'MEASURED_AND_VERIFIED','results':results,'input_quality':quality,
          'strict_defense_direction_observed':same,'strict_evaluation':str(strict_folder),
          'strict_checkpoint':results[-1]['checkpoint']})
    print(json.dumps({'output':str(out),'strict_sim':m['mean_sim'],'strict_cer':m['mean_cer']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation-root',type=Path,required=True)
    build(p.parse_args().validation_root)
