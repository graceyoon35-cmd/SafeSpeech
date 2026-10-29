# SPEC 생성 환경 검증 및 STOI 연동 준비

검증일: 2026-10-30 KST. 실제 실행 로그·산출물에 근거한 기록입니다.

## 결과

원본 `protect.perturb()`의 SPEC 모드로 동일 영어 입력을 1회 및 10회 업데이트했고,
두 실행 모두 종료 코드 0, 저장 산출물 검증 PASS입니다. 원본 알고리즘은 수정하지 않았습니다.
전체 `protect.py` CLI, 음질 평가, 한국어 처리, XTTS-v2 보호 효과를 검증한 결과는 아닙니다.
STOI 수치는 이번 작업에서 계산하지 않았습니다.

## 브랜치와 작업 경로

요청 시 실제 체크아웃은 `feature/integration-setup` (`be448f2`)이었고,
`docs/QUICKSTART_KO.md`가 미커밋 변경으로 표시되었습니다.
요청한 브랜치는 로컬에는 없고 `origin/feature/spec-perturbation` (`3345f42`)에 있었습니다.
기존 작업 트리를 보존하기 위해 다음을 실행했습니다.

```powershell
git worktree add -b feature/spec-perturbation C:/Users/<USER>/SafeSpeech-spec-perturbation origin/feature/spec-perturbation
```

- 원래 폴더: `C:\Users\<USER>\SafeSpeech`, 계속 `feature/integration-setup`.
- 이번 작업 폴더: `C:\Users\<USER>\SafeSpeech-spec-perturbation`, `feature/spec-perturbation`.
- `master`, 다른 팀원 브랜치 변경·병합, fetch, stage, commit, push는 수행하지 않았습니다.
- `feature/metrics` 분석은 로컬 원격 추적 참조 `27ae06d1fe304320c671eeeb15338e8c23b36ac6`을
  `git show`로 읽었습니다. 원격 서버의 최신 상태를 새로 조회한 것은 아닙니다.
- 원래 QUICKSTART 파일 SHA-256은 실행 전후
  `e88ab9628a13d9cec24840041227b0715c14dcdffb1bc3f7e955c2cd64c923ed`로 동일합니다.

## 실행 환경

| 항목 | 실제 확인 결과 |
| --- | --- |
| Windows GPU | NVIDIA GeForce RTX 5060 Ti, 8151 MiB |
| Windows 드라이버 | NVIDIA-SMI / KMD 610.88, CUDA UMD 13.3 표시 |
| Windows Python | 3.15, torch 미설치. 이 인터프리터로 SPEC를 실행하지 않음 |
| 실제 실행 OS | WSL2 Ubuntu 22.04.5 LTS, Linux 6.18.54.1 |
| 사용자 / Python | safespeech / 3.10.12 |
| 가상환경 | `/home/safespeech/.venvs/safespeech` |
| torch / torchaudio | 2.7.1+cu128 / 2.7.1+cu128 |
| PyTorch CUDA 런타임 | 12.8, `torch.cuda.is_available() == True` |
| GPU capability | 12.0, torch 빌드에 sm_120 포함 |
| 오디오 패키지 | NumPy 1.23.5, SciPy 1.13.1, SoundFile 0.12.1, librosa 0.10.0 |
| 기타 | torch_stoi 0.2.1, pystoi 0.4.1, Transformers 4.46.3, huggingface-hub 0.36.0 |
| CUDA Toolkit | Windows / WSL에서 nvcc를 찾지 못함. 설치 여부 전체를 조사한 것은 아님 |

드라이버의 CUDA 표시값과 PyTorch 런타임 버전은 서로 다른 항목입니다.
CUDA 행렬곱, Conv1d 역전파, STFT, WAV 메모리 저장·복원, 리샘플링, mel 계산을
`scripts/check_gpu_audio.py`로 실행했고 PASS입니다. `pip check`도 충돌 없이 종료했습니다.
패키지나 시스템 설정을 새로 설치·변경하지 않았습니다.

## 파일·실행 경로 분석

루트 `models/`, `configs/` 디렉터리는 없습니다. 실제 구현은 다음과 같습니다.

| 파일 | 역할 및 주의점 |
| --- | --- |
| `protect.py` | CLI는 G/D/DUR/WD 및 WavLM을 준비하고 데이터 배치별 `perturb()` 실행 후 `.noise` 저장 |
| `protect.py:perturb` | 13개 항목 배치 입력, mel + beta × (NR + KL) 손실, sign gradient 갱신, epsilon 투영 |
| `toolbox.py` | 생성기·판별기 및 optimizer 생성 |
| `bert_vits2/models_noise.py` | 섭동 생성용 `SynthesizerTrn`; gradient가 입력 파형까지 전달됨 |
| `bert_vits2/models.py` | 학습·평가용 모델 구현 |
| `bert_vits2/configs/libritts_bert_vits2.json` | 24kHz, FFT/window 2048, hop 512, mel 128, seed 42 |
| `bert_vits2/data_utils.py` | cleaned 목록, PCM 파형 정규화, BERT 특징 및 13개 항목 collate |
| `bert_vits2/text/english_bert_mock.py` | 이름에 mock이 있으나 실제 로컬 DeBERTa 특징을 생성 |
| `scripts/smoke_perturbation.py` | 1초 이상 중 최단 EN 샘플을 새 폴더에 복사, 생성기와 원본 perturb만 실행 |
| `save_audio.py` | 전체 배치 `.noise`를 WAV로 변환하며 기존 filelist를 다시 씀. 이번에는 실행하지 않음 |
| `evaluate.py` | fine-tuning 모델 기반 합성 및 MCD/WER/SIM. 이번 파일 쌍 STOI용 진입점이 아님 |

SPEC 손실은 `loss_mel + 10 * (loss_nr + loss_kl)`입니다. SafeSpeech 모드의
지각 손실은 이번 실행에서 사용하지 않았습니다. `compute_stoi()`는 미분용 `NegSTOILoss`이며
팀 metrics 브랜치의 최종 파일 쌍 STOI 점수 함수와 구분해야 합니다.
반환 손실은 마지막 업데이트 직전 값으로, 업데이트 후 최종 파형의 재평가 값이 아닙니다.

문서에서 찾은 기존 Windows 실행 경로는 **PowerShell → WSL → 가상환경 Python**입니다.
Windows 네이티브 SPEC 실행법이나 네이티브 성공 기록은 확인하지 못했습니다.
원래 폴더의 `docs/QUICKSTART_KO.md`, `docs/WSL_SETUP_KO.md`,
`docs/SETUP_VALIDATION_SUMMARY_KO.md`가 기존 1회·10회 실행을 설명합니다.

## 모델과 코드 보존

integration-setup에만 있던 `.gitignore`, 두 WSL requirements 파일,
`check_gpu_audio.py`, `download_setup_models.py`, `smoke_perturbation.py`를
내용 변경 없이 이번 worktree로 복사했습니다. 브랜치 전체를 병합하지 않았습니다.
추가 작성 파일은 `scripts/validate_spec_pair.py`와 이 보고서입니다.

8개 모델 파일은 NTFS 하드 링크로 기존 로컬 파일을 연결했습니다. 이 파일들은 같은
실제 파일 내용을 공유하므로 향후 이 경로에서 모델 파일을 직접 수정하면 원래 경로에도
반영될 수 있습니다. 이번 실행에서는 읽기만 했습니다. 대용량 복사·재다운로드는 없었습니다.
`scripts/download_setup_models.py`를 실행하여 고정 리비전의 크기·해시를 모두 재확인했습니다.

| 모델 | 고정 리비전 | 파일 |
| --- | --- | --- |
| OedoSoldier/Bert-VITS2-2.3 | 3eac455e350982b84267125c370713948b011694 | G_0, D_0, DUR_0, WD_0.pth |
| microsoft/deberta-v3-large | 64a8c8eab3e352a784c658aef62be1662607476f | pytorch_model.bin, spm.model |
| microsoft/wavlm-base-plus | 4c66d4806a428f2e922ccfa1a962776e232d487b | pytorch_model.bin, config.json |

생성기 체크포인트에는 `emb_g.weight`가 없습니다. 기존 스모크와 원본 로더의 fallback대로
seed 기반 초기값을 유지하며, 스모크는 다른 누락·예상 밖 키가 없는지 검사합니다.
학습된 화자 임베딩이 준비되었다는 의미가 아닙니다. 전체 CLI의 판별기·WavLM 로딩 경로는
이번 실행 범위에 포함되지 않았으며, PyTorch 2.7의 torch.load 기본값 변화와 전체 모델
GPU 메모리 사용량도 해당 경로에서 별도 확인해야 합니다.
원본의 체크포인트 개수 assertion은 실제 개수를 검증하는 표현이 아니므로 준비 여부의
근거로 사용하지 않았습니다. 기존 코드를 수정하지 않았습니다.

## 음성 사용 권한과 출처

저장소 README는 제공 WAV를 LibriTTS train-clean-100의 화자 5339로 명시합니다.
공식 [OpenSLR SLR60](https://www.openslr.org/60/)의 공개 배포 라이선스는
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)입니다.
이 공개 라이선스의 출처 표시·라이선스 링크·변경 사항 표시 조건을 근거로 로컬 검증을 진행했습니다.
로컬 WAV가 원본 배포 아카이브와 바이트 단위로 동일한지, 화자 개인의 별도 동의서는
확인하지 않았습니다. 저장소 파일명과 README에 근거한 데이터 출처 확인입니다.

- 출처: LibriTTS, Heiga Zen, Viet Dang, Rob Clark, Yu Zhang, Ron J. Weiss,
  Ye Jia, Zhifeng Chen, Yonghui Wu, 2019.
- 입력: `data/LibriTTS/5339/5339_14134_000092_000003.wav`, EN, “There was no reply.”
- 원본 SHA-256: `155e50f0fb0596c794acc88394533c2f66a57d3915072521e84d771f8a22212e`.
- 변경: 원본은 그대로 보존하고 복사본에 SPEC 섭동을 더한 `protected.wav`를 별도 저장.
- 각 새 실행 폴더에 `DATA_ATTRIBUTION.md`를 추가해 출처와 변경 사항을 함께 보관합니다.

## 실제 실행 명령

PowerShell에서 실행한 감사 wrapper:

```powershell
wsl -d Ubuntu-22.04 -u safespeech -- /home/safespeech/.venvs/safespeech/bin/python -B /mnt/c/Users/<USER>/SafeSpeech-spec-perturbation/runs/audit_20261030/run_audit.py
```

wrapper가 worktree 루트에서 아래 명령을 실제 순서대로 실행했습니다.
원본과 과거 결과 해시를 실행 전후 비교하며, 1회 생성·검증 실패 시 10회를 실행하지 않습니다.

```bash
python -m pip check
python scripts/check_gpu_audio.py
python scripts/download_setup_models.py
python scripts/smoke_perturbation.py --iterations 1
python scripts/smoke_perturbation.py --iterations 10
```

여기서 python은 모두 `/home/safespeech/.venvs/safespeech/bin/python`입니다.
재실행 시 wrapper의 고정 감사 폴더는 재사용하지 말고 새 감사 폴더를 만듭니다.
스모크 스크립트 자체는 실행마다 마이크로초 단위로 새 `runs/smoke_*` 폴더를 만듭니다.

독립 파일 검사 재현 예시(아래 CLI 형식 자체는 안내이며 실제 감사에서는 동일한
`validate()` 함수를 직접 호출했습니다):

```bash
python scripts/validate_spec_pair.py runs/smoke_20261030_004820_253549 \
  --expected-source-sha256 155e50f0fb0596c794acc88394533c2f66a57d3915072521e84d771f8a22212e \
  --output runs/audit_20261030/validation_10_recheck.json
```

검사 결과 파일은 배타 생성(`x`)하므로 이미 존재하는 출력 파일을 덮어쓰지 않습니다.

## 생성 결과와 파일 검사

공통 설정: SPEC, batch 1, cuda:0, seed 42, epsilon 8/255 = 0.03137254901960784,
step alpha = epsilon/10, weights [0.05, 10], BERT (1024, 33).
config의 train.batch_size=16, CLI 기본 batch-size=27과 별도로 스모크는 1을 사용합니다.

| 항목 | 원본 | 1회 보호 | 10회 보호 |
| --- | ---: | ---: | ---: |
| 샘플레이트 | 24000 | 24000 | 24000 |
| 샘플 수 | 24481 | 24481 | 24481 |
| 길이 (초) | 1.0200416667 | 1.0200416667 | 1.0200416667 |
| 채널 | mono | mono | mono |
| WAV subtype | PCM_16 | FLOAT | FLOAT |
| 최소 진폭 | -0.2567138672 | -0.2598511279 | -0.2808842659 |
| 최대 진폭 | 0.3531799316 | 0.3563171923 | 0.3589661717 |
| RMS | 0.0600019685 | 0.0600761970 | 0.0645752631 |
| NaN / Inf 개수 | 0 / 0 | 0 / 0 | 0 / 0 |
| 원본 대비 최대 절대 변화량 | 0 | 0.003137260675430298 | 0.0313725508749485 |

허용 오차 `epsilon + 1e-7` 기준으로 두 출력 모두 통과했습니다.
저장 noise 텐서 shape `(1, 1, 24481)`와 유한값·진폭을 검사했고,
저장 WAV가 `clip(original_float32 + noise, -1, 1)`과 정확히 일치했습니다.
모든 source와 입력 복사본의 SHA-256이 일치합니다.
원래 폴더 데이터·기존 결과·미커밋 문서 및 새 worktree 데이터 총 **353개 파일**의
실행 전후 해시를 비교하여 변경 0개를 확인했습니다.

| 항목 | 1회 | 10회 |
| --- | ---: | ---: |
| 스모크 전체 프로세스 시간 | 13.913초 | 14.557초 |
| loss_mel | 30.598785 | 21.056269 |
| loss_nr | 252.145569 | 155.501984 |
| loss_kl | 114.721588 | 15.144611 |
| PyTorch peak allocated GPU | 602.591 MiB | 614.391 MiB |
| 프로세스 종료 코드 / 저장 파일 검증 | 0 / PASS | 0 / PASS |

시간은 모델·BERT 로딩과 출력 저장을 포함합니다. GPU 수치는 전체 GPU 점유량이 아닙니다.
두 손실을 비교하는 것만으로 일반적인 수렴·음질·보호 성능을 주장할 수 없습니다.

새 결과 경로는 이번 worktree 기준입니다.

- 1회: `runs/smoke_20261030_004806_282821/`
- 10회: `runs/smoke_20261030_004820_253549/`
- 각 폴더: 원본 이름 WAV, `protected.wav`, `perturbation.pt`, `report.json`,
  `input.cleaned`, `5339_14134_000092_000003.bert.pt`, `DATA_ATTRIBUTION.md`.
- 감사: `runs/audit_20261030/`의 `baseline.json`, `environment.json`, `outcome.json`,
  `validation_1.json`, `validation_10.json`, 과거 결과 검사 JSON, 단계별 `.log`, 실행 wrapper.
- 평가 준비: `runs/audit_20261030/spec10_metrics_manifest.json` (실행하지 않은 입력 설정).

원래 worktree의 과거 결과도 읽기만 하여 확인했습니다.

| 기존 폴더 | 이번 확인 |
| --- | --- |
| `smoke_20261025_024005_466049` | 1회 report, 원본·보호 WAV·noise 존재, 같은 검사 PASS |
| `smoke_20261025_024217_002294` | 10회 report, 원본·보호 WAV·noise 존재, 같은 검사 PASS |
| `smoke_20261025_023926_423049` | report·protected 없음, 미완료로 분류, 성공 집계 제외 |
| `smoke_20261025_023934_495419` | report·protected 없음, 미완료로 분류, 성공 집계 제외 |

이번 1회·10회 스모크의 실행 오류는 없었습니다. 과거 미완료 폴더의 실패 원인은
이번 잔여 파일만으로 확정하지 않았습니다. 모든 기존 폴더는 보존했습니다.

## feature/metrics 연동 계약과 다음 단계

`feature/metrics`에서 확인한 실제 인터페이스:

```python
# scripts/audio_metrics.py
load_audio(path) -> (mono_float64_array, sample_rate)
score_pair(reference, candidate, reference_rate, candidate_rate)
# -> {"stoi": float, "max_abs_delta": float}
```

필요 파일은 `scripts/audio_metrics.py`, `scripts/run_experiment.py`,
`scripts/robustness_transforms.py`, `requirements-evaluation-smoke.txt`와 실험 manifest입니다.
STOI는 `pystoi.stoi(..., extended=False)`를 사용합니다. 의존성 목록은 NumPy 1.26.4,
SciPy 1.13.1, SoundFile 0.12.1, pystoi 0.4.1이며 기존 GPU 환경의 NumPy 1.23.5와 다르므로
평가 담당자 환경을 분리하거나 호환성을 확인해야 합니다. 이번 GPU 환경은 변경하지 않았습니다.

엄격 계약은 mono, 1초 이상, 비무음, 유한값, 진폭 [-1,1], 원본과 보호 WAV의 같은
샘플레이트·샘플 수입니다. 자동 리샘플링·자르기·정렬은 없습니다. 이번 파일 쌍은 이 계약을
충족합니다. 이번 검사 PASS가 STOI 계산 성공을 대신하지는 않습니다.

manifest v1은 `audio.original`/`protected`/`synthetic` 경로,
`evaluation.reference_role=original`, `target_role=protected`, `alignment=strict`를 사용합니다.
경로는 manifest 디렉터리 기준입니다. 준비한 manifest는 새 10회 출력만 가리키며
`synthetic=null`, Gaussian seed 42/std 0.005를 지정합니다.
`report.json` 경로 같은 임의 키는 엄격 스키마에 추가하지 않고 생성 출처 보고서를 별도로 전달합니다.

runner는 원본 대 보호 WAV의 `metrics.before` STOI와, 보호 WAV에 Gaussian을 적용한 뒤의
`metrics.after` STOI를 모두 계산합니다. Gaussian 후 값은 보호 WAV 단독 점수가 아닙니다.
단순 파일 쌍 평가만 필요하면 `load_audio` 및 `score_pair`를 호출하는 작은 평가 진입점으로
연결할 수 있습니다. 이번에는 어느 평가 경로도 실행하지 않았습니다.

다음 실행은 평가 파일이 준비된 metrics 작업 트리에서 수행할 안내이며 미실행입니다.

```bash
python scripts/run_experiment.py /path/to/spec10_metrics_manifest.json
```

평가 담당자에게 같은 실행 폴더의 원본 WAV·보호 WAV·생성 report 및 validation JSON,
출처 문서를 전달하고 SHA-256을 비교합니다. noise는 선택 자료이며 모델과 BERT 특징은
CPU STOI 계산에 필요하지 않습니다. 다른 PC에서는 manifest 경로를 해당 파일 위치에 맞춥니다.
Git 업로드나 외부 전송은 수행하지 않았습니다.

이후 우선순위는 (1) 실제 파일 쌍 STOI 측정, (2) 학습된 화자 임베딩/체크포인트 합의,
(3) 한국어 데이터 권한 및 전처리·모델 언어 지원, (4) 별도 XTTS-v2 추론·평가 계약입니다.
현재 BERT-VITS2 언어 매핑은 ZH/JP/EN이며 KO 지원이 없어 한국어를 직접 넣는 경로는
별도 개발·검증이 필요합니다. XTTS-v2 합성 WAV는 길이가 달라질 수 있으므로 strict STOI에
자동 연결하지 말고 평가 목적과 정렬 정책을 먼저 정해야 합니다.

## 최종 Git 상태

이번 브랜치 HEAD는 `3345f42`로 원격 추적 브랜치와 같습니다. 새 파일만 미추적 상태입니다.
기존 알고리즘·설정·데이터·평가 코드는 변경하지 않았습니다.
새 `.gitignore`로 모델·실행 결과·BERT 특징·캐시가 Git 후보에서 제외됨을 확인했습니다.
커밋·Push는 사용자 승인 후 진행합니다.
