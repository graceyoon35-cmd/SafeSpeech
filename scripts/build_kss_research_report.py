"""Actual KSS results, separately from synthetic benchmarks, with artifact checks."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from run_korean_melo_spec import save,sha256,validate_handoff,check_manifest
from evaluation_statistics import statistics


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def csv_write(path,rows):
    with Path(path).open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();study=read(args.study/'study.json')
    if study['status']!='KSS_RESEARCH_TRAINING_AND_EVALUATION_PASS':raise ValueError('Study incomplete')
    inputs=check_manifest(study['manifest'],allow_licensed_research=True)
    trained=read(study['training_summary']);evaluation=read(study['evaluation'])
    if trained['scope']!='kss_noncommercial_research' or evaluation['scope']!=trained['scope']:
        raise ValueError('Mixed synthetic/KSS scope')
    if trained['steps_per_condition']!=1000 or evaluation['test_utterances']!=30:
        raise ValueError('Unexpected study budget/test count')
    args.output.mkdir(parents=True,exist_ok=False)
    manifest_dir=Path(study['manifest']).parent
    for filename in ('automatic_quality.csv','review_recommended.csv','excluded_utterances.csv','input_audit.json'):
        shutil.copyfile(manifest_dir/filename,args.output/filename)
    spec=read(Path(study['spec_run'])/'run.json')
    handoff=read(Path(study['spec_run'])/'handoff_manifest.json')['utterances']
    validate_handoff(handoff)
    if len(handoff)!=400 or spec['handoff_count']!=400:raise ValueError('Expected 200 x 2 SPEC artifacts')
    spec_rows=[]
    for row in handoff:
        case=Path(row['validation_path']).parent
        stats=read(case/'report.json')['files'][0];gradients=read(case/'gradients.json')
        if len(gradients)!=row['iterations'] or not all(g['finite'] and g['nonzero']>0 for g in gradients):
            raise ValueError('Missing valid gradients')
        spec_rows.append({'utterance_id':row['utterance_id'],'split':row['split'],'iterations':row['iterations'],
            'stoi':row['stoi'],'epsilon':row['epsilon'],'generation_seconds':stats['seconds'],
            'max_abs_delta':stats['max_abs_delta'],'manual_listening_verified':row['transcript_verified']})
    csv_write(args.output/'kss_spec_per_utterance.csv',spec_rows)
    training=[];history={};initializations=set();datasets=[]
    for item in trained['training']:
        report=read(item['report']);cfg=read(Path(item['run'])/'config.json')
        if report['status']!='TRAINING_SMOKE_PASS' or report['completed_steps']!=1000:
            raise ValueError('Incomplete training')
        if sha256(item['checkpoint'])!=report['checkpoint_sha256']:raise ValueError('Changed checkpoint')
        initializations.add(report['initial_checkpoint_sha256'])
        data=read(cfg['experiment']['dataset_manifest'])['utterances']
        datasets.append([(r['utterance_id'],r['split'],r['text'],r['features_sha256']) for r in data])
        if Counter(r['split'] for r in data)!=Counter(train=140,validation=30,test=30):raise ValueError('Split changed')
        if cfg['train']['seed']!=52 or cfg['experiment']['permission_declarations_validated'] is not False:
            raise ValueError('Unexpected seed or fabricated legacy approval')
        losses=report['losses']
        if [r['step'] for r in losses]!=list(range(1,1001)):raise ValueError('Step history missing')
        times=np.array([r['step_seconds_excluding_checkpoint'] for r in losses])
        condition=item['condition'];history[condition]=losses
        training.append({'condition':condition,'steps':1000,'seed':52,'train_utterances':140,
            'mean_step_seconds':float(times.mean()),'median_step_seconds':float(np.median(times)),
            'timed_training_seconds':report['seconds'],'checkpoint_save_seconds':sum(c['save_seconds'] for c in report['checkpoints']),
            'peak_gpu_allocated_gib':report['peak_allocated_bytes']/1024**3,
            'generator_loss_first50_mean':float(np.mean([r['generator'] for r in losses[:50]])),
            'generator_loss_last50_mean':float(np.mean([r['generator'] for r in losses[-50:]])),
            'checkpoint':item['checkpoint'],'checkpoint_sha256':report['checkpoint_sha256']})
    if len(initializations)!=1 or any(d!=datasets[0] for d in datasets):raise ValueError('Unequal paired initialization/data')
    csv_write(args.output/'kss_training.csv',training)
    for name in ('per_file.csv','summary.csv','source_quality_breakdown.csv','evaluation.json'):
        shutil.copyfile(Path(study['evaluation']).parent/name,args.output/('kss_'+name))
    with (args.output/'kss_per_file.csv').open(encoding='utf-8-sig',newline='') as stream:
        outputs=list(csv.DictReader(stream))
    if len(outputs)!=120:raise ValueError('Expected 4 models x 30 test sentences')
    for row in outputs:
        if sha256(row['output'])!=row['output_sha256'] or sha256(row['reference'])!=row['checkpoint_sha256']:
            raise ValueError('Changed evaluation artifact')
    clean={r['sentence_id']:r for r in outputs if r['condition']=='clean'}
    differences=[]
    for condition in ('SPEC_10','SPEC_1'):
        chosen=[r for r in outputs if r['condition']==condition]
        for metric in ('sim','cer'):
            delta=[float(r[metric])-float(clean[r['sentence_id']][metric]) for r in chosen]
            differences.append({'condition_minus_clean':condition+'-clean','metric':metric,**statistics(delta),
                'scope':'Paired prompt bootstrap only, one speaker and one training seed'})
    csv_write(args.output/'kss_paired_differences.csv',differences)
    conditions=['pretrained','clean','SPEC_10','SPEC_1'];scores={r['condition']:r for r in evaluation['measurements']}
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,metric,scale,title in [(axes[0],'sim',1.,'ECAPA cosine SIM'),(axes[1],'cer',100.,'Mean Korean CER (%)')]:
        mean=np.array([scores[c]['mean_'+metric]*scale for c in conditions])
        low=np.array([scores[c][metric+'_ci95_low']*scale for c in conditions]);high=np.array([scores[c][metric+'_ci95_high']*scale for c in conditions])
        ax.bar(conditions,mean,yerr=np.vstack([np.maximum(0,mean-low),np.maximum(0,high-mean)]),capsize=5,
            color=['#718096','#3179ab','#c95446','#ef9e42'])
        ax.set(title=title);ax.tick_params(axis='x',rotation=20)
    axes[0].set_ylim(min(0.,min(scores[c]['sim_ci95_low'] for c in conditions)-.05),1)
    fig.suptitle('KSS real recordings: 140 train / 30 validation / 30 test, 1,000 steps\nOne speaker/seed; 95% prompt bootstrap; transcripts not listening-reviewed',fontsize=11)
    fig.tight_layout();fig.savefig(args.output/'kss_sim_cer.png',dpi=180);fig.savefig(args.output/'kss_sim_cer.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.5))
    for c,losses in history.items():
        values=np.array([r['generator'] for r in losses]);axes[0].plot(np.arange(50,1001),np.convolve(values,np.ones(50)/50,mode='valid'),label=c)
    axes[0].set(title='Generator loss: 50-step moving mean',xlabel='Step',ylabel='Loss');axes[0].legend()
    axes[1].bar([r['condition'] for r in training],[r['mean_step_seconds'] for r in training]);axes[1].set(title='Actual step time',ylabel='Seconds (save excluded)')
    axes[2].bar([r['condition'] for r in training],[r['peak_gpu_allocated_gib'] for r in training]);axes[2].set(title='PyTorch peak allocated',ylabel='GiB',ylim=(0,8))
    fig.suptitle('KSS matched training settings: batch 1, BF16, seed 52, 1,000 steps')
    fig.tight_layout();fig.savefig(args.output/'kss_training.png',dpi=180);fig.savefig(args.output/'kss_training.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    values=[[r['stoi'] for r in spec_rows if r['iterations']==n] for n in (1,10)]
    ax.boxplot(values,labels=['SPEC_1','SPEC_10'],showmeans=True)
    ax.set(title='KSS 200 input pairs: STOI distribution',ylabel='STOI',ylim=(0,1.02))
    fig.tight_layout();fig.savefig(args.output/'kss_spec_stoi.png',dpi=180);fig.savefig(args.output/'kss_spec_stoi.svg');plt.close(fig)
    # Preserve synthetic results as an explicitly separate comparison artifact.
    root=Path(__file__).resolve().parents[1]
    synthetic=root/'runs/benchmark_20261118/presentation/evaluation_summary.csv'
    shutil.copyfile(synthetic,args.output/'PRIOR_SYNTHETIC_100STEP_summary.csv')
    summary={'status':'KSS_REAL_RECORDING_RESEARCH_RESULTS_VERIFIED','scope':'kss_noncommercial_research',
        'study':study,'dataset_counts':dict(Counter(r['split'] for r in inputs)),
        'excluded_count':0,'review_recommended_count':read(manifest_dir/'input_audit.json')['review_recommended_count'],
        'manual_listening_verified_count':sum(r['transcript_verified'] is True for r in inputs),
        'separate_speaker_consent':'not_verified','external_publication':'not_approved_by_this_record',
        'pretraining_corpus_overlap_with_KSS':'not_verified; test held out from this finetuning only',
        'source_and_license':read(manifest_dir/'input_audit.json')['research_policy'],
        'training':training,'evaluation':evaluation['measurements'],'paired_prompt_differences':differences,
        'spec':{'pairs':400,'total_generation_run_seconds':spec['total_seconds'],
            'stoi_mean_spec1':float(np.mean(values[0])),'stoi_mean_spec10':float(np.mean(values[1]))},
        'limitations':['One Korean speaker and one training seed','Unreviewed corpus transcripts; ASR strata supplied',
            'Melo surrogate and evaluated TTS share architecture','Random seeded discriminators',
            'KSS disjointness from pretrained model corpus not verified',
            'No transformed-data attack retraining or other attack model','No population-wide efficacy claim',
            'Synthetic100 and KSS1000 differ in dataset, steps and test texts; do not pool them']}
    save(args.output/'kss_results.json',summary)
    lines=['# KSS 실제 녹음 연구 실험 — 2026-11-18',
        '', '200개 유지: train 140 / validation 30 / test 30. 제외 0개. ASR 불일치 58개는 검수 권장 목록으로 보존.',
        '전수 청취 미실시 상태에서 공개 라이선스의 비상업 TTS 연구 범위로 수행했다. 별도 화자 동의나 외부 배포 승인을 확보한 것으로 기록하지 않았다.',
        '', '## 동일 조건 실제 학습', '', '| 조건 | step | 초/step | peak GPU GiB | loss 처음50 평균 | loss 마지막50 평균 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in training:
        lines.append(f"| {r['condition']} | {r['steps']} | {r['mean_step_seconds']:.4f} | {r['peak_gpu_allocated_gib']:.3f} | {r['generator_loss_first50_mean']:.4f} | {r['generator_loss_last50_mean']:.4f} |")
    lines+=['','초기 생성기·대본·특징·140개 train 목록·seed 52·batch 1·BF16·LR 0.0001·1,000-step을 동일하게 고정했다.',
        '단일 GPU 제한 학습 루프이며 공식 Melo 모델과 손실을 사용한다. 판별기는 같은 seed의 무작위 초기화다.',
        '', '## 동일한 test 30문장 평가', '', '| 조건 | 평균 SIM | 문장 평균 CER | 문자 가중 CER |', '|---|---:|---:|---:|']
    for c in conditions:
        r=scores[c];lines.append(f"| {c} | {r['mean_sim']:.4f} | {r['mean_cer']*100:.2f}% | {r['corpus_cer']*100:.2f}% |")
    lines+=['', 'SIM은 같은 clean train 기준 녹음 5개에 대한 ECAPA cosine 평균이다. CER는 Whisper small 한국어 전사의 공백/문장부호를 제외한 값이다.',
        '문장별 CSV, 평균/분산/문장 bootstrap 95% 구간, 원본 ASR 품질 strata, clean 대비 paired 차이를 별도 저장했다.',
        '구간은 문장 내 변동성만 표현한다. 단일 화자·단일 학습 seed의 결과이며 전체 화자/모델에 일반화할 수 없다.',
        'test 30개는 이번 미세조정의 train/validation에서 분리된 문장이다. 사전학습 모델의 학습 코퍼스와 KSS의 비중복은 확인하지 못했다.',
        '', '## SPEC 입력 검증', '',f"400개 보호 WAV와 2,200회 gradient 이력을 검사했다. 평균 STOI: 1회 {summary['spec']['stoi_mean_spec1']:.4f}, 10회 {summary['spec']['stoi_mean_spec10']:.4f}.",
        'STOI는 입력 명료도 지표이며 복제 방어 성공 자체가 아니다. 미청취 데이터의 주관적 품질을 검증한 것으로 표시하지 않는다.',
        '', '## 근거 및 제한', '', 'Kyubyong Park, KSS Dataset: Korean Single Speaker Speech Dataset, 2018.',
        '원 배포처: https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset',
        '라이선스: CC BY-NC-SA 4.0, https://creativecommons.org/licenses/by-nc-sa/4.0/',
        '실제 미러: Bingsu/KSS_Dataset, revision 48fdfd7ab1dbc1a62e4e8a8b9f4c360259d51d3c.',
        '외부 음성/가중치 배포·상업적 사용은 수행하지 않았다. 학습량/seed/공격 모델 확장, 채널 변환 후 재학습 방어 실험은 미실시다.',
        '기존 합성 100-step 결과는 PRIOR_SYNTHETIC_100STEP_summary.csv로 따로 보존했다. 데이터·학습량·평가문장이 달라 결과를 합산하거나 동등 조건으로 비교하지 않는다.',
        '', '실행 경로/체크포인트/해시/상세 기록은 kss_results.json과 kss_training.csv를 참조한다.']
    (args.output/'KSS_RESULTS_KO.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'directory':str(args.output.resolve()),'status':summary['status']}),flush=True)


if __name__=='__main__':main()
