# WSL2 Melo 실제 학습 및 평가 기록 — 2026-11-18

현재 작업 위치는 `C:/Users/<USER>/SafeSpeech-desktop-integration-20261118`, 브랜치는
`integration/desktop-20261118`다. 이전 수학 신호 검증에 이어 **실제 한국어 합성 데이터로
SPEC → paired 학습 → held-out 합성 → SIM/CER 측정**을 실행했다.
**KSS 실제 음성의 SPEC·학습은 실행하지 않았다.** 커밋·푸시는 하지 않았다.

## 데이터 사용 근거와 KSS 상태

[별도 라이선스 확인 기록](KSS_LICENSE_SCOPE_20261118_KO.md)에 원 배포처·미러·CC 약관과 작업별 판단을 기록했다.
비상업 TTS 연구에 관한 공개 근거를 확인했지만, 팀 승인·별도 화자 모사 허가가 있다는 기록은 만들지 않았다.
라이선스가 모든 인격권이나 합성물 공개까지 일괄 허용한다고 결론내리지 않았다.

KSS 200개의 WAV·대본·해시·train 140 / validation 30 / test 30은 재검증 PASS다.
원본 manifest는 변경하지 않았으며 사용 확인 0/200, 청취 검수 0/200 상태다.
한국어 전처리·실제 BERT 특징 200개, 조건별/분할별 4열 metadata.list·7열 cleaned 목록,
clean/SPEC_1/SPEC_10 config.json을 별도 생성했다.

위치: `runs/kss_training_review_3koit4g9/`.
clean 목록은 원본 WAV를 참조한다. **protected 목록의 WAV는 아직 생성되지 않았으며 경로만 예정되어 있다.**
세 config 모두 실행 허가 gate가 false이고, protected 입력의 미생성 상태를 기록했다.
이 준비물을 학습 가능한 완성 데이터셋으로 보고하지 않는다.
`review_status.json`은 `METADATA_FEATURES_CONFIG_PREPARED_EXECUTION_BLOCKED`다.

실제 KSS 입력으로 `run_authorized_korean_training.py`를 호출하면
`Require reviewed transcript declaration`로 모델 실행 전에 거부되는 것을 확인했다.
이는 프로그램이 현재 검수 요건을 적용한 결과이며, 라이선스가 모든 연구를 금지한다는 판단은 아니다.
없는 검수·허가 기록을 대신 생성하거나, 실행을 위해 bool을 임의로 켜지 않았다.

## 전용 학습 환경

- 새 환경 `/home/safespeech/.venvs/melo-training-20261118`을 만들었다.
- 기존 CUDA 환경의 site-packages를 `.pth`로 참조하고, 추가 패키지는 새 환경에만 설치했다.
  기존 환경의 실행 전후 `pip freeze`가 동일함을 비교 확인했다.
- 시스템 패키지 상속으로 나타난 pygobject/pycairo 경고는 새 환경의 시스템 상속을 끄는 것으로 해결했다.
  최종 `pip check`: `No broken requirements found`.
- Python 3.10.12, torch/torchaudio 2.7.1+cu128, RTX 5060 Ti.
- 추가 패키지 고정 목록: `requirements-melo-training-wsl.txt`.
  전체 환경: `runs/training_20261118/training_freeze.txt`.
- Melo 공식 소스 `209145371cff8fc3bd60d7be902ea69cbdb7965a`와 한국어 모델
  `0207e5adfc90129a51b6b03d89be6d84360ed323`는 이전과 동일하다.
- 한국어 BERT: `kykim/bert-kor-base`, `1779cc0982ada0216dd6de0dd4e86fb78201926d`.
  실제 가중치 SHA-256 `ae43a392e533ccb9fd38e5c65130aeee50381b87e81e795f9d90469accd78236`.
- ECAPA: `speechbrain/spkrec-ecapa-voxceleb`, 리비전 `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`.
- ASR: openai-whisper 20240930의 `small` 다국어 모델, 언어 `ko`.
  평가 모델 위치/파일 해시는 `/home/safespeech/models/melo-evaluation-20261118/model_provenance.json`.

전체 다국어 `melo.api` 설치 성공을 주장하지 않는다. 공식 한국어 G2P와 공식 BERT 추출기를 직접 연결했다.
가짜 음소·영벡터 한국어 BERT·무작위 생성기 가중치를 넣지 않았다.
공식 설계상 한국어에서 사용하지 않는 중국어용 1024차원 특징만 영벡터다.
pooler/문장 관계 분류 head의 미사용 경고는 masked-LM 특징 추출 경로의 경고로 로그에 남겼다.

## 실제 합성·SPEC 입력

`runs/synthetic_melo_training_o_g56dze/`에 공개 사전학습 모델의 한국어 합성 4개를 만들었다.
train 2 / validation 1 / test 1을 고정했고 길이는 약 4.447, 3.170, 3.367, 3.193초다.
생성 프롬프트를 보존했으며 **사람의 청취 검수는 하지 않았다**.
합성 입력의 `transcript_verified`는 기존 runner가 요구하는 생성 프롬프트 출처 일치 선언이며,
별도의 `listening_review=NOT_PERFORMED`를 기록했다. KSS 검수와는 관계없다.

SPEC 결과: `runs/korean_melo_spec_eeuge8ay/`.
4발화 × 1/10회 = 8개 보호 WAV의 epsilon·PCM16·해시·유한/비영 gradient·STOI 검증 PASS.
동일 원본과 대본/특징을 사용한 paired 입력: `runs/paired_melo_experiment_9xuhnebf/`.
조건별 config.json과 split별 metadata.list/cleaned 목록 및 dataset.json이 실제 WAV에 연결되어 있다.
원본과 보호 데이터의 ID·대본·split·BERT 특징 해시 동일성을 실행 전에 확인했다.

기존 runner의 `NONHUMAN_NOT_FOR_FINETUNING` handoff 상태는 그대로 두었다.
이번에 새로 만든 준비기는 사용자가 요청한 **합성 데이터 학습 스모크만 별도 scope**로 다룬다.
합성 데이터를 실제 사람 복제 공격 데이터로 표시하지 않는다.

## 학습 구현과 실제 실행

`train_melo_bounded.py`는 공식 Melo 생성기, MultiPeriodDiscriminator, DurationDiscriminator,
공식 mel/KL/duration/feature/adversarial 손실을 사용하는 단일 GPU의 제한된 학습 루프다.
**공식 `train.py` 전체를 수정 없이 실행한 결과는 아니다.** DDP·epoch 스케줄·평가 루프 대신
고정 optimizer step 수와 명시적 실패/완료 기록을 사용한다. 자동 resume 실행은 아직 구현하지 않았다.
재시작에 필요한 생성기/판별기/optimizer 및 일부 RNG 상태 파일은 보존했다.

- 공통 조건: batch 1, segment 8192, seed 52, learning rate 0.0001, AdamW, BF16 autocast.
- 생성기는 동일 사전학습 checkpoint에서 strict 로딩한다. 256개 화자 embedding 크기를 유지하고 target을 slot 0에 연결한다.
- 파형/길이 판별기는 같은 seed의 무작위 초기화다. 사전학습 판별기를 사용했다고 주장하지 않는다.
- 매 step loss와 parameter gradient 유한성을 검사하고, 실제 생성기 parameter 변경을 확인했다.
- 최초 clean 1-step 시도는 probe가 weight_norm의 `weight_v`라고 가정해 실패했다.
  실제 Conv1d의 `weight`를 검사하도록 고쳐 재실행에 성공했고 실패 로그도 보존했다.
- 이후 세 조건을 각각 새 프로세스로 3-step 학습했다. 모두 완료됐으며 체크포인트 strict 재로딩·합성까지 통과했다.
- 첫 crop에 무음이 포함된 step이 있어 RMS를 로그에 추가했다. 후속 step에는 유성 구간이 포함되었다.
  짧은 step들의 loss를 수렴 또는 보호 성능으로 해석하지 않는다.

실행 summary: `runs/paired_melo_training_063mlxyb/summary.json`.

| 조건 | 실제 step | 실행 시간¹ | PyTorch peak allocated | 체크포인트 |
|---|---:|---:|---:|---|
| clean | 3 | 21.76초 | 2.048GiB | `runs/melo_train_clean_xv6kzym4/G_3.pth` |
| SPEC_1 | 3 | 23.10초 | 2.048GiB | `runs/melo_train_SPEC_1_563v75l6/G_3.pth` |
| SPEC_10 | 3 | 21.12초 | 2.048GiB | `runs/melo_train_SPEC_10_3_oywkp4/G_3.pth` |

¹ 모델 구성과 체크포인트 저장 포함. 설치/다운로드 제외. 각 생성기 파일은 207,822,652 bytes이며,
각 실행 폴더의 `training_state.pth`는 판별기와 optimizer 상태다.
peak reserved는 약 2.143GiB이며 Windows 데스크톱의 VRAM 사용량은 이 수치에 포함되지 않는다.
KSS 길이/대규모 학습에서도 같은 메모리라고 일반화하지 않는다.

## held-out 합성 및 실제 SIM·CER

평가 폴더: `runs/melo_smoke_evaluation_29va9n1c/`.
모든 조건에서 같은 held-out 합성 프롬프트 1개와 seed를 사용했다.
SIM은 같은 clean train 기준 WAV 2개에 대한 ECAPA cosine 평균이다.
CER는 Whisper small에 한국어를 지정하고 공백·문장부호를 제외한 문자로 계산했다.
화자 복제 성공 threshold는 적용하지 않았다.

| 조건 | SIM | CER |
|---|---:|---:|
| 사전학습 모델 | 0.804136 | 15.3846% |
| clean 3-step | 0.780215 | 15.3846% |
| SPEC_1 3-step | 0.788476 | 15.3846% |
| SPEC_10 3-step | 0.796549 | 15.3846% |

**실측값이지만 방어 효과를 입증하는 실험이 아니다.** 학습 데이터가 합성이고,
train 2개·test 1개·3-step의 경로 점검이다. 조건 간 작은 차이에서 유의성·공격 성공률을 추론하면 안 된다.
정확한 합성 WAV·checkpoint 해시와 ASR 전사문은 per_file.csv, 모델 provenance는 evaluation.json에 있다.
이전 수학 신호 결과와도 구분한다. 이번 실제 한국어 합성 입력의 STOI는 별도 spec_input_quality.csv에 있다.

## 최종 검증·발표 준비물·다음 작업

- 추가로 KSS 200개를 Whisper small로 읽기 전용 ASR 대조했다.
  `runs/kss_asr_screening_20gow51e/listening_priority.csv`에 불일치가 큰 순서로 정렬했다.
  정규화 후 142/200개가 ASR와 일치했고 발화별 CER 단순 평균은 약 4.918%였다.
  이것은 합성 후 평가 CER가 아니라 원본 녹음의 대본 점검용 ASR 수치다.
  ASR 일치는 청취 검수가 아니며, 불일치는 원래 대본 오류의 증명도 아니다. 검수 완료는 여전히 0/200이다.
  우선 청취 후보: `kss_hf_04094`, `kss_hf_04634`, `kss_hf_02674`, `kss_hf_09588`, `kss_hf_12842`.
- 기존 62개 + 학습 경계 6개 = **68개 테스트 PASS**, 8.736초.
  거짓 config gate, 실제 사람 미검수, WAV/특징 변조, split/대본 변경, 중복 ID 거부를 포함한다.
- KSS 원본 재검증 PASS. 기존 CUDA 환경 freeze 동일. 통합 diff 공백 검사 PASS.
- 표·JSON·PNG·SVG: `runs/training_20261118/presentation_artifacts/`.
  그림 제목과 CSV에 synthetic smoke 한계를 표시했다. 그래프를 눈으로 확인했다.
- `runs/training_20261118/integrity/integrity.json`: 생성기 checkpoint·판별기/optimizer 상태 파일의 해시,
  공식 모델/config·KSS 200개 원본 해시를 최종 확인했다. 최종 구현의 실행 후 소스 스냅샷도 보존했다.
- 기존 `docs/DESKTOP_INTEGRATION_20261118_KO.md`는 앞 단계 시점의 기록이다.
  이 문서는 그 이후 실제 합성 학습·SIM/CER 실행을 추가한 최신 기록이다.

주요 실행 스크립트 순서 (모든 명령은 이 작업에서 직접 실행했다):

1. `bootstrap_melo_training.py`, `download_korean_training_bert.py`, `prepare_melo_evaluation_models.py`.
2. `prepare_synthetic_melo_training.py` → `run_korean_melo_spec.py --allow-nonhuman --device cuda`.
3. `prepare_paired_melo_experiment.py` → `run_paired_melo_smoke.py --steps 3`.
4. `evaluate_paired_melo_smoke.py` → `build_melo_smoke_report.py`.
5. KSS 준비 전용 `prepare_kss_training_review.py`. 실제 사람용 실행기는
   `run_authorized_korean_training.py`이며 검수·허가 요건 충족 전에는 거부한다.

11월 21일 핵심 결과 목표를 위해 남은 우선 작업은 **KSS 사용 범위에 대한 팀 기록과 발화별 청취 검수**다.
확인한 발화만 원래 split을 유지한 입력으로 만들고, 실제 SPEC → 제한 학습 → 학습량 확대 →
held-out test 평가로 확장한다. 11월 24일 PPT에는 실제 KSS 결과가 생기기 전까지
이번 자료를 기술 검증으로만 제시해야 한다. 외부 WAV·가중치·합성물 공개는 별도 미확인 항목으로 남긴다.
