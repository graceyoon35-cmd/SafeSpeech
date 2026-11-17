# 100-step 실측·복구·평가 및 발표 자료 — 2026-11-18 KST

작업 공간 `C:/Users/<USER>/SafeSpeech-desktop-integration-20261118`, 브랜치 `integration/desktop-20261118`을 유지했다.
기존 3-step 산출물은 보존했고 저장 당시의 생성기·optimizer/판별기 상태·설정/보고서 해시와 재비교했다.
커밋·푸시·다른 팀원 브랜치 병합은 하지 않았다.

## 결과 범위

이번에는 **공개 MeloTTS-Korean 모델의 합성 train 2발화**로 세 조건을 각각 100-step 학습했다.
실제 가중치 업데이트·저장·복구·합성·SIM/CER 측정이다. 3-step 스모크와 별도 폴더에 기록했다.
그러나 **실제 KSS 화자의 방어 성능을 검증한 결과는 아니다.** 단일 합성 화자·학습 seed 하나이며,
채널 변환 후 재학습 공격에 대한 방어도 측정하지 않았다.

환경은 Ubuntu 22.04 / Python 3.10.12 / RTX 5060 Ti 8GB /
torch·torchaudio 2.7.1+cu128이다. batch 1, BF16, segment 8192, seed 52, AdamW LR 0.0001을 동일하게 사용했다.
모델 가중치·대본·split·BERT 특징·학습 step·평가 문장과 seed를 조건 간 고정했다.
공식 Melo 구성요소와 손실을 사용하는 로컬 제한 학습 루프이며 공식 train.py 전체 실행과는 구분한다.

## 1. 50 → 100-step 학습과 시간

| 조건 | 완료 step | 전체 평균 초/step | 워밍업 제외 평균¹ | peak allocated | 학습 세션 시간² |
|---|---:|---:|---:|---:|---:|
| clean | 100 | 0.5793 | 0.5345 | 2.052GiB | 104.28초 |
| SPEC_1 | 100 | 0.5685 | 0.5297 | 2.052GiB | 108.10초 |
| SPEC_10 | 100 | 0.6102 | 0.5729 | 2.052GiB | 115.39초 |

¹ 각 프로세스의 첫 5 step, 즉 global 1~5와 51~55를 제외한 90개 step의 평균.
² 0→50과 50→100 세션의 보고서 시간을 합산. 모델 구성/복구/저장 포함, 사전 입력 검증은 제외한다.
OS 프로세스 전체 wall-clock 타이머라고 표시하지 않는다. 각 step은 CUDA 동기화 전후 타이머로 측정했으며 저장 시간은 제외했다.
loss의 처음/마지막 10-step 평균, 모든 step의 개별 loss·RMS·시간, checkpoint 경로는 CSV/JSON에 있다.
무음 crop과 음성 crop이 섞여 있어 단일 loss 값의 감소를 수렴으로 판단하지 않는다.

실행 summary: `runs/melo_timing_100_4n35l8ff/summary.json`.
모든 조건에서 50과 100-step checkpoint를 새 디렉터리에 보존했다.
`checkpoints/step_000050/` 또는 `step_000100/`에는 생성기, 판별기/optimizer 상태 및 해시 manifest가 있다.

## 2. 중단·재개 확인

- clean은 step 50 저장 완료 후 의도적으로 KeyboardInterrupt를 발생시켰다.
- 새 프로세스에서 생성기·파형 판별기·길이 판별기와 세 optimizer를 복구했다.
- 각 optimizer의 실제 상태 step이 `[50]`인 것을 검사한 뒤 global step 51~100을 진행했다.
- CPU/CUDA/Python/NumPy RNG, 데이터 순번, MAS noise schedule, 전체 loss history도 복원한다.
- config·dataset manifest·binary 파일 해시를 검사하며 불일치나 불완전 저장을 거부한다.
- 별도 WSL 저장 시험에서 같은 clean step50 checkpoint를 다시 복구했을 때 step51/52의
  기록된 손실/RMS가 본 실행과 모두 일치했다 (최대 차이 0.0).
- 이 확인은 두 재개 분기 사이의 일치다. 무중단 실행과의 비교나 다른 GPU/OS에서의 bitwise 재현을 검증한 것은 아니다.
- 완료 manifest를 마지막에 기록한다. 중간 저장 중 죽은 미완성 디렉터리는 복구 대상으로 받지 않는다.
  전원 장애 자체를 실제로 시험한 것은 아니며, 마지막 완전 checkpoint 이후 step은 유실될 수 있다.

`train_melo_bounded.py --resume <완료 checkpoint 디렉터리> --steps <목표 global step>`으로 새 결과 폴더에서 재개한다.
기존 checkpoint를 덮어쓰지 않는다. 3-step 당시의 구형 checkpoint 형식은 자동 복구 대상으로 지원하지 않으며 원본 그대로 보존한다.

## 3. 저장 병목과 본 학습 예상 시간

약 1.2GB checkpoint 저장+해시 검사에서 Windows 마운트 경로는 비교 표본 17.26초,
WSL 내부 디스크는 2.37초였다. native 비교는 **별도 clean 50→52-step 분기에서 저장 1회**로 측정했다.
100-step 평가 checkpoint를 이 분기의 52-step 결과로 바꾸지 않았다.
저장 성능 약 7.3배 차이를 전체 학습 속도 향상 배율로 해석하지 않는다.

본 학습은 새 checkpoint를 `/home/safespeech/experiments/SafeSpeech-spec-perturbation`에 두는 옵션과
500-step 저장 간격을 준비했다. 기존 Windows 결과와 소스는 이동하지 않았다.

추정식: `조건별 실제 평균 step 시간 × 목표 step + ceil(step/500) × 측정 native 저장 시간`.

| 조건별 목표 step | clean + SPEC_10 | 세 조건 전체 | 세 조건 일정용 50% 여유 포함 |
|---|---:|---:|---:|
| 1,000 | 약 20분 | 약 30분 | 약 44분 |
| 5,000 | 약 100분 | 약 148분 | 약 222분 |
| 10,000 | 약 200분 | 약 295분 | 약 443분 |

이는 **합성 2발화의 실측을 이용한 조건부 추정**이다. 실제 KSS 학습 시간을 측정했다고 주장하지 않는다.
사용 확인·청취·SPEC·전처리·시작 전 검증·초기화·합성/평가 시간은 제외한다.
50% 여유는 일정 가정이며 통계적 신뢰구간이 아니다.
실제 KSS pilot의 길이·메모리·속도·손실로 보정해야 하고, 위 step 수에서 수렴한다는 보장도 없다.
140 train·batch1이면 1,000/5,000/10,000 step은 각각 약 7.1/35.7/71.4 epoch다.

## 4. 12문장 평가 및 변동성

학습/validation과 겹치지 않는 12문장을 고정했고, 모든 조건에서 같은 특징·seed로 합성했다.
사전학습 baseline과 세 100-step 모델을 평가해 **48개 실제 평가 WAV**를 생성했다.
SIM은 동일 clean train 기준 2발화에 대한 ECAPA cosine 평균이다.
CER는 Whisper small의 한국어 전사에서 공백·문장부호를 제외해 계산했다.

| 조건 | 평균 SIM | 문장별 CER 평균 | 전체 문자 가중 CER |
|---|---:|---:|---:|
| 사전학습 | 0.7877 | 4.72% | 4.43% |
| clean 100-step | 0.7867 | 5.21% | 4.93% |
| SPEC_1 100-step | 0.7258 | 8.07% | 7.88% |
| SPEC_10 100-step | 0.3916 | 42.06% | 41.87% |

문장별 결과·ASR 전사·편집 오류 수·기준 문자 수·WAV/모델 해시는 `evaluation_per_file.csv`에 있다.
평균·표본 분산·seed 52의 5,000회 bootstrap 95% 구간은 `evaluation_summary.csv`에 있다.
구간은 12문장 내 변동성만 표현하며 학습 seed/화자 간 불확실성은 포함하지 않는다.
동일한 작은 train set의 단일 실험에서 관측한 차이이며 KSS나 사람 전체의 방어 효과로 일반화하면 안 된다.

평가 prompt: `runs/melo_eval_prompts_12_44lpp1o2/evaluation_prompts.json`.
평가 원본: `runs/melo_smoke_evaluation_yktb2105/`.

## 5. SPEC·강건성 자료

SPEC 자체의 epsilon·WAV·gradient·STOI 검증 결과는 기존 실제 한국어 합성 4발화 결과를 재사용했다.
별도로 합성 test 입력 1개에 SPEC 1/10회, Gaussian 9조건·MP3 3조건·Resampling 3조건,
각각 2회 변환을 실행했다. WAV 120개와 MP3 24개를 검증했다.
동일 채널을 통과한 clean/protected 사이의 STOI를 계산해 그래프를 만들었다.

이것은 **채널 변환 후 품질 검증**이다. 같은 seed의 두 반복은 독립 음성 표본이 아니며,
변환된 데이터로 공격 모델을 재학습한 방어 강건성 실험은 아직 미실행이다.
수학 신호 기반의 이전 강건성 결과와도 분리했다.
원본: `runs/synthetic_robustness_quality_of_e33b4/quality_summary.json`.

## 6. 로컬 KSS 검수 도구와 승인 후 실행 경로

Windows에서 프로젝트의 **`Review-KSS.cmd`를 더블클릭**한다. Windows Python의 표준 Tk/WAV 기능만 사용한다.
검수자 ID(가명 가능)를 입력하고 F5로 재생한 뒤, F8 일치/F9 불일치/F10 보류를 선택한다.
해당 발화 재생 전이나 재생 길이 경과 전에는 일치/불일치 저장을 거부한다.
버튼 클릭은 사용자의 실제 청취 판정 선언이며 프로그램이 청취 사실을 독립 검증하는 것은 아니다.

기본 기록: `runs/kss_listening_review_200/transcript_check_reviewed.csv`, 변경 이력 `review_events.jsonl`.
원본 WAV·대본·manifest나 사용 허가 플래그는 수정하지 않는다. 다시 실행하면 기록을 이어서 읽는다.
동시에 여러 창에서 같은 검수 폴더를 편집하는 방식은 지원하지 않는다.
우선순위는 기존 ASR 대조표를 사용하며 ASR 일치를 청취 완료로 변환하지 않는다.

원 배포처·CC 조건·미확인 범위는 `KSS_LICENSE_SCOPE_20261118_KO.md`에 유지했다.
`docs/templates/kss_research_use_record.template.json`은 실제 팀 판단 기록용 양식이며 모든 허용값 기본값은 false다.
일치 검수 CSV와 실제 사용 범위 기록이 모두 있어야 `build_authorized_kss_manifest.py`가 별도 입력을 생성한다.
원래 split을 유지한 train/validation/test 검수 표본이 필요하고, 미검수·불일치 발화는 제외한다.
이 코드는 기록의 형식/해시 연결을 검사할 뿐 동의나 법적 승인을 자체 생성하지 않는다.

승인 입력의 후속 자동 실행은 `run_authorized_korean_training.py`로 연결했다.
준비된 KSS BERT 특징을 해시 검증 후 재사용하고, SPEC → 동일 조건 학습 → 고정 test 평가까지 실행한다.
현재는 검수 0/200·사용 확인 플래그 0/200이므로 KSS 실제 SPEC/학습/평가는 실행하지 않았다.
KSS protected config의 예정 WAV는 여전히 미생성 상태다.

## 7. 검증과 발표 파일

- WSL 전체 테스트 **84개 PASS**. checkpoint 손상/설정 변경/미완성 저장·RNG 복원,
  수동 검수 전 승인 거부·원본 보존·통계 구간 검사 포함.
- Windows 실제 Tk UI 구성 self-test PASS (청취·판정 자동 수행 안 함).
  처음 Windows에서 과학 패키지까지 포함한 혼합 테스트를 실행한 시도는 NumPy 미설치로 실패했다.
  Windows 과학 환경을 추가 설치하지 않고 GUI는 Windows, 전체 backend 테스트는 WSL에서 검증했다.
- 기존 3-step 생성기·학습 상태·설정·보고서의 해시 재비교 PASS.
- 그래프는 실제 CSV/JSON에서 생성했으며 PNG·SVG와 데이터 CSV를 함께 보존했다.
- `runs/benchmark_20261118/presentation/`:
  `training_100step.*`, `evaluation_12prompts.*`, `robustness_quality.*`,
  `training_timing.csv`, `training_time_estimates.csv`, `spec_technical_results.csv`, `final_report.json`.
- 10장 PPT 초안: `SafeSpeech_synthetic_benchmark_20261118.pptx`.
  ZIP/재읽기/shape 경계 검사를 통과했다. PowerPoint 자체의 렌더링 검사는 하지 않았다.
  원본 그래프는 별도 이미지로 확인했다. 슬라이드는 KSS 미실행과 합성 실험의 한계를 명시한다.
- 커밋하지 않은 최신 구현은 같은 발표 폴더의 `source_overlay.zip`과
  `source_overlay_manifest.json`으로 별도 보존한다. Git 기준 commit에 적용할 소스 묶음이며
  오디오·checkpoint·계정 파일은 넣지 않는다. 각 소스 파일 SHA-256과 ZIP 재읽기를 검사한다.

## 8. 남은 문제와 현실적 일정

현재 핵심 장애물은 GPU가 아니라 **실제 KSS 사용 범위 기록과 발화별 청취 검수**다.
추가로 KSS 실제 pilot, 충분한 학습량/여러 seed, 변환 후 공격 모델 재학습은 미완료다.
전체 다국어 Melo API/공식 전체 trainer 검증도 이번 전용 학습 경로와 구분한다.

1. **11/18:** 합성 100-step·복구·12문장 평가 완료, 검수 도구/자료 전달.
2. **11/19:** 사용 범위 기록과 우선 검수 확보. 조건이 충족된 표본으로만 실제 KSS 100-step pilot.
3. **11/20:** pilot이 통과하면 clean/SPEC_1/SPEC_10을 동일 1,000-step으로 실행하고 고정 test 평가.
   loss/품질을 보고 필요 시 5,000-step으로 확대한다. 시간 추정은 pilot 결과로 다시 계산한다.
4. **11/21:** 실제 결과·체크포인트·CSV 고정, 시간이 되면 seed 반복 또는 강건성 공격 재학습 추가.
5. **11/22~23:** 재검증·출처·제한·PPT 최종 편집. **11/24 마감**.

사용/검수 확인이 늦어지면 11/21까지 실제 KSS 방어 결과를 확보한다고 보장할 수 없다.
그 경우 합성 실험의 검증 가능한 결과와 미완료 범위를 구분해 발표한다.
