# 평가·강건성·모델 공통 기반 점검 (2026-10-27)

## 검토 기준과 이전 작업

검토 브랜치: `feature/integration-setup`, 커밋 `be448f2`.
GitHub 원격 브랜치의 해시도 `git ls-remote`로 동일함을 확인했습니다.
기존 `SETUP_VALIDATION_SUMMARY_KO.md`, `QUICKSTART_KO.md`, 커밋 내역과
실행 코드를 먼저 검토했습니다. WSL 환경 구축과 SPEC 1회/10회 실행은 이전
완료 사항이며 이번에 재실행하지 않았습니다. 이전 WSL 검증 결과를 Mac 검증으로
간주하지 않습니다.

## 코드 준비 상태

| 영역 | 기존 구현 | 남은 작업 및 실제 실행 조건 |
| --- | --- | --- |
| 섭동 | `protect.py` SPEC/SafeSpeech, 음성 저장, BERT 특징 생성, 기존 SPEC smoke | 전체 CLI·SafeSpeech·전체 데이터 검증은 이전에도 미완료 |
| 평가 | `evaluate.py` 합성 → 목록 생성 → MCD/WER/SIM | 모델·학습 결과·여러 평가 패키지가 필요하며 지표만 독립 실행하기 어려움 |
| STOI | `protect.compute_stoi()`의 `NegSTOILoss` | 평가 CLI의 독립 STOI는 없었음. 이번 CPU smoke에 `pystoi` 추가 |
| 강건성 | 별도 변환·실험 runner를 저장소에서 찾지 못함 | 변환 종류·파라미터·seed·평가 쌍·결과 형식부터 담당자와 합의 |
| 모델 | `train.py`, `toolbox.py`, BERT-VITS2 코드, 데이터셋별 JSON 설정 | 체크포인트·BERT 특징·의존성 필요. Mac 학습 실행은 미검증 |

`evaluate.py`의 WER 루프는 짝수 행(원본)을 건너뛰므로 `WER_gt`가 항상 0으로
출력되는 결함이 있습니다. 평가 담당자가 수정·검증할 항목으로 남겼으며 원본 코드는
변경하지 않았습니다. SIM은 ECAPA 화자 검증 점수이며 일반적인 파형 유사도로
대체할 수 없습니다. 코드의 ASR은 화자 검증 판정 비율이고 Whisper 음성 인식과
구분해서 해석해야 합니다.

## Mac 환경 및 모델 설정 점검

- `.venv`: Python 3.10.11, macOS arm64. `.venv-1`은 Python 3.13이므로 이번에 사용하지 않았습니다.
- 설치된 torch 2.14.1 / torchaudio 2.11.0은 버전이 다릅니다. 둘 다 import는 되었고
  `pip check`도 통과했지만 전체 모델 실행 호환성을 입증하지 않습니다. 이번 지표는
  두 라이브러리를 사용하지 않습니다. 모델 담당자가 별도 환경에서 버전을 맞춰 검증해야 합니다.
- 이번 실행 환경에서 CUDA와 MPS 가용성은 모두 False였습니다. 실제 계산은 CPU로만 수행했습니다.
- LibriTTS JSON은 24kHz, 화자 수 850, 100 epochs, 설정 batch size 16입니다.
  `train.py` CLI 기본 batch size 64가 이를 덮어씁니다. WavLM 경로는
  `./bert_vits2/slm/wavlm-base-plus`이며 저장소 루트 기준입니다.
- 학습 목록 108개 WAV는 모두 존재하지만 대응 `.bert.pt`는 0개입니다. 테스트 목록은 26행입니다.
- Mac에는 `checkpoints/**/*.pth`, DeBERTa 가중치/`spm.model`, WavLM 가중치가 없습니다.
  ECAPA 디렉터리에는 `hyperparams.yaml`만 있고 가중치가 없습니다.
- `evaluate.py`는 `checkpoints/LibriTTS/BERT_VITS2_<mode>_LibriTTS_100.pth`를
  기대합니다. 기본 사전학습 모델 다운로드만으로 평가 전체가 준비되지는 않습니다.
- 기존 문서에 기록된 `G_0.pth`의 `emb_g.weight` 누락과 화자 임베딩 구성은 미해결입니다.
- `train.py`는 CUDA AMP/backend 설정을 사용하며 MPS 선택 옵션이 없습니다.
  `--gpu -1` CPU 경로는 있지만 학습 전체 Mac 호환성을 보장하지 않습니다.
  `assert os.listdir(hps.model_dir) != 4`도 파일 수 검사로 작동하지 않습니다.
- `bert_vits2/config.yml`은 CUDA 및 예시 데이터 경로를 포함합니다. 루트 `train.py`가
  사용하는 데이터셋별 JSON과 구분해야 합니다. CMU 설정은 있지만 대응 filelist는 없습니다.

## 이번에 추가한 공통 테스트

원본 코드·기존 안내서는 그대로 두고 다음 파일만 추가했습니다.

- `requirements-evaluation-smoke.txt`: 모델 학습 환경과 분리한 작은 CPU 평가 의존성 목록
- `scripts/smoke_evaluation.py`: mono/유한값/진폭/길이 검사, 동일 음성 STOI,
  고정 seed 잡음 대조군 또는 사용자 지정 WAV 쌍, JSON 및 입력 복사본 저장
- `scripts/test_smoke_evaluation.py`: 동일 음성·잡음 대조, 길이/샘플레이트 불일치,
  무음·스테레오·짧은 음성·NaN·진폭 초과 거부 검증

실행 방법 (저장소 루트, Python 3.10 환경):

```bash
.venv/bin/python -m pip install -r requirements-evaluation-smoke.txt
.venv/bin/python scripts/smoke_evaluation.py
.venv/bin/python -m unittest discover -s scripts -p 'test_smoke_evaluation.py' -v
```

실제 보호 음성을 전달받으면 다음처럼 같은 쌍 인터페이스를 사용할 수 있습니다.
경로는 실제 파일로 바꿉니다.

```bash
.venv/bin/python scripts/smoke_evaluation.py --reference path/to/clean.wav --candidate path/to/protected.wav
```

샘플레이트·샘플 수가 다르면 실패합니다. 자동 자르기·리샘플링·정렬을 하지 않습니다.
1초 이상 mono 음성을 요구하며, STOI가 발화 부족 경고를 내면 실패 처리합니다.
합성 음성은 길이가 달라질 수 있어 이 인터페이스를 곧바로 합성 평가에 적용할 수 없습니다.
반환 `status: PASS`는 STOI smoke 범위만 의미하며 SIM은 항상 `NOT_RUN`입니다.
`pystoi`의 표준 STOI와 기존 `torch_stoi` 미분 손실의 숫자를 동일시하지 않습니다.

## 실행 결과

입력은 기존 SPEC 검증에 사용한 원본 WAV입니다. SPEC는 재생성하지 않았습니다.

| 검사 | 결과 |
| --- | --- |
| CPU STOI 실행 | PASS, 24kHz / 24,481 samples |
| 원본 대 원본 STOI | 0.9999999999999997 |
| seed 42, 표준편차 0.005 Gaussian 잡음 대조 STOI | 0.970284020573383 |
| 같은 seed 재실행 | 같은 점수 확인 |
| 입력 계약 및 대조군 테스트 | 3개 테스트 모두 통과 |
| 가상환경 의존성 | `pip check` 통과 |
| SIM/MCD/WER, 학습·추론, 강건성 실험 | 미실행 |

기록: `runs/evaluation_smoke_kq1kaodw/report.json`,
재실행: `runs/evaluation_smoke_hgvksa6g/report.json`.
`runs/`는 기존 Git 제외 규칙이 적용됩니다. 보고서에 OS·Python·패키지 버전·seed·입력 경로를
기록했습니다. 잡음 대조군은 SPEC가 아니며, 이 점수는 보호 효과나 논문 재현 결과가 아닙니다.

## 회의 전/후 우선순위 제안

| 우선순위 | 작업 | 완료 기준 / 담당 경계 |
| --- | --- | --- |
| P0 (이번 완료) | CPU 음성 쌍·STOI smoke와 결과 기록 | 다른 담당자 코드 없이 재현 가능, 입력 오류 거부 |
| P1 (회의 전 후속 후보) | 모델 없는 사전 점검 CLI | 파일 목록·체크포인트·특징 존재 검사만 JSON화. 이번에는 문서로 점검 결과 제공 |
| P1 (회의에서 합의) | 공통 평가 쌍 manifest | 원본/보호/합성 역할, sample rate, seed, 변환 이름, 결과 필드 확정 |
| P2 (평가 담당자 연결) | SIM 단일 쌍 CPU 검증 | 별도 의존성, 고정 ECAPA 리비전·가중치 준비 후 동일 음성 sanity check; WER 원본 누락 수정 |
| P2 (강건성 담당자 연결) | 변환 1종의 입력/출력 smoke | 파라미터와 seed 기록, 원본 불변, 변환 결과를 공통 평가로 전달. 보호 효과 결론은 별도 |
| P3 (모델 담당자 연결) | CUDA 환경의 학습/추론 최소 검증 | 임베딩 구성 해결, `.bert.pt`, 사전학습 파일 검증 후 작은 배치 실행 |

회의에서는 평가 대상(보호 음질 vs 합성 보호 효과), SIM 모델 리비전, 강건성 변환 및
학습 환경을 먼저 합의하는 것이 좋습니다. 기존 대형 실행 파일 수정이나 Push는 하지 않았습니다.
