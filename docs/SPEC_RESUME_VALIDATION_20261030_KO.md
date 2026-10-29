# SPEC 개발 재개: 보존·환경 재검증·STOI 연동 준비

실행일: 2026-10-30 KST. 아래 성공 결과는 실제 실행 로그와 JSON에 근거합니다.
작업 폴더는 `C:\Users\<USER>\SafeSpeech-spec-perturbation`,
브랜치는 `feature/spec-perturbation`, HEAD는 `3345f42`입니다.

## 기존 작업 보존과 변경

기존 미추적 8개 파일을 모두 읽고,
`runs/audit_resume_20261030_005532_823644/preserved/` 아래 같은 상대 경로로
원본 내용을 복사 보관했습니다. 수정 전 SHA-256은 같은 감사 폴더의
`preservation.json`에 있습니다. 백업 파일은 기존 파일과 하드 링크로 연결하지 않았습니다.

| 파일 | 이번 처리 |
| --- | --- |
| `.gitignore` | 내용 유지 |
| `requirements-wsl-base.txt` | 내용 유지 |
| `requirements-wsl-perturbation.txt` | 내용 유지 |
| `scripts/check_gpu_audio.py` | 내용 유지, 실제 실행 |
| `scripts/download_setup_models.py` | 내용 유지, 다운로드 실행 안 함 |
| `scripts/smoke_perturbation.py` | 내용 유지, 실제 1회·10회 실행 |
| `scripts/validate_spec_pair.py` | 내용 유지, 두 새 실행에 CLI 검증 수행 |
| `docs/SPEC_VALIDATION_20261030_KO.md` | 원본 백업 후 기존 10회 손실값 3개만 정정 |

기존 보고서의 대상은 `smoke_20261030_004820_253549/report.json`입니다.
해당 실제 report와 로그에 맞춰 mel/NR/KL을 각각
`21.056269`, `155.501984`, `15.144611`로 정정했습니다.
이번 새 10회 실행의 손실값은 아래 별도 표에 기록합니다.

integration-setup의 환경 구축 커밋 `be448f2`는 현재 HEAD보다 1개 앞서며,
기존 파일 수정 없이 9개 파일을 추가한 커밋입니다. 위 환경 관련 6개 파일은
그 커밋과 Git 콘텐츠 해시가 같습니다. 현재 SPEC 보고서와 검증 스크립트는
그 커밋에 없는 추가 작업입니다.

누락 문서 3개도 `git show`로 비교했습니다. 파일 복사나 병합은 하지 않았습니다.

| integration-setup에만 있는 문서 | 내용 및 현재 보고서와의 관계 |
| --- | --- |
| `docs/QUICKSTART_KO.md` | 신규 설치·일상 실행 안내. 현재 SPEC 보고서는 이 설치 안내를 대체하지 않음 |
| `docs/WSL_SETUP_KO.md` | 상세 WSL 구축 이력, 오류 해결, 과거 스모크. 과거 SafeSpeech 폴더 기준 |
| `docs/SETUP_VALIDATION_SUMMARY_KO.md` | 2026-10-25 환경·검증 총정리. 현재 보고서는 별도 worktree에서 10월 30일 실행을 기록 |

추가 문서는 이 보고서와 `docs/SPEC_STOI_HANDOFF_KO.md`입니다.
원본 `protect.py`, 설정, filelist, requirements.txt는 수정하지 않았습니다.

## 재검증 환경

| 항목 | 실제 확인 결과 |
| --- | --- |
| OS | WSL2 Ubuntu 22.04.5 LTS |
| Linux | 6.18.54.1-microsoft-standard-WSL2 |
| Python | 3.10.12 |
| 실행 인터프리터 | `/home/safespeech/.venvs/safespeech/bin/python` |
| GPU | NVIDIA GeForce RTX 5060 Ti, 총 8151 MiB |
| CUDA | `torch.cuda.is_available() == True`, PyTorch 런타임 12.8 |
| capability | 12.0, torch 빌드에 sm_120 포함 |
| NVIDIA-SMI | 610.57.01, KMD 610.88, CUDA UMD 표시 13.3 |
| torch / torchaudio | 2.7.1+cu128 / 2.7.1+cu128 |
| NumPy / SciPy | 1.23.5 / 1.13.1 |
| SoundFile / librosa | 0.12.1 / 0.10.0 |
| torch_stoi / pystoi | 0.2.1 / 0.4.1 |
| transformers / huggingface-hub | 4.46.3 / 0.36.0 |
| tqdm / PyYAML / loguru / sentencepiece | 4.64.1 / 6.0.2 / 0.7.3 / 0.2.0 |

`pip check`는 종료 코드 0, `No broken requirements found.`입니다.
GPU 행렬곱, Conv1d 역전파, STFT, WAV 메모리 저장·복원,
리샘플링과 mel 계산을 `check_gpu_audio.py`로 실행해 PASS입니다.
드라이버의 CUDA 표시값과 PyTorch 런타임은 서로 다른 항목입니다.
패키지 설치, 가상환경 재생성, requirements 재설치는 수행하지 않았습니다.

## SPEC 재실행 결과

원본 스모크 명령은 저장소 루트에서 기존 WSL Python으로 다음을 실행하는 방식입니다.

```bash
/home/safespeech/.venvs/safespeech/bin/python -B scripts/smoke_perturbation.py --iterations 1
/home/safespeech/.venvs/safespeech/bin/python -B scripts/smoke_perturbation.py --iterations 10
```

실제 실행은 감사 폴더의 `run_resume.py`에서 위 두 프로세스와 검증 CLI를
순서대로 호출했습니다. 1회 실행과 파일 검증이 성공한 뒤 10회를 실행했습니다.
스모크는 매번 새 마이크로초 타임스탬프 폴더를 생성합니다.

입력은 `data/LibriTTS/5339/5339_14134_000092_000003.wav`,
SHA-256은 `155e50f0fb0596c794acc88394533c2f66a57d3915072521e84d771f8a22212e`입니다.
SPEC, batch 1, seed 42, epsilon 8/255, alpha epsilon/10,
가중치 [0.05, 10], BERT 특징 (1024, 33)을 사용했습니다.

| 항목 | 새 1회 실행 | 새 10회 실행 |
| --- | ---: | ---: |
| 실행·검증 종료 코드 | 0 / 0 | 0 / 0 |
| 스모크 프로세스 시간 | 20.001초 | 18.012초 |
| loss_mel | 30.598785 | 20.847090 |
| loss_nr | 252.145569 | 155.301666 |
| loss_kl | 114.721588 | 15.090028 |
| 최대 절대 변화량 | 0.003137260675430298 | 0.0313725508749485 |
| 최고 GPU 텐서 할당량 | 602.591 MiB | 614.391 MiB |
| 파일 검증 | PASS | PASS |

시간은 모델 로딩·BERT 처리·저장을 포함하며 GPU 수치는 전체 점유량이 아닙니다.
같은 seed여도 이번 10회 손실과 WAV는 이전 실행과 다릅니다. 비트 단위
재현성은 보장·검증하지 않았고 원인도 확정하지 않았습니다.
반환 손실은 마지막 업데이트 직전 값이므로 최종 WAV의 재평가 손실이 아닙니다.

## 출력 경로와 파일 검증

경로는 현재 저장소 루트 기준입니다.

- 1회: `runs/smoke_20261030_005658_023201/`
- 10회: `runs/smoke_20261030_005719_318250/`
- 감사: `runs/audit_resume_20261030_005532_823644/`

각 스모크 폴더에는 `5339_14134_000092_000003.wav` (원본 복사본),
`protected.wav`, `perturbation.pt`, `report.json`, `input.cleaned`,
BERT 특징 파일, `DATA_ATTRIBUTION.md`가 있습니다.
감사 폴더에는 `environment.json`, `baseline.json`, `outcome.json`,
`validation_1.json`, `validation_10.json`, 실제 명령별 `.log`,
원본 8개 백업과 재검증 wrapper가 있습니다. wrapper는 기존 출력과 충돌하면
배타 생성으로 실패하므로 그대로 재실행하지 말고 다음에는 새 감사 폴더를 사용합니다.

두 결과 모두 원본과 보호 WAV가 24kHz, mono, 24,481샘플, 약 1.02004초입니다.
원본은 PCM_16, 보호 WAV는 FLOAT입니다. NaN/Inf가 없고 진폭 [-1,1],
비무음 조건과 `epsilon + 1e-7` 범위를 통과했습니다.
noise shape은 `(1, 1, 24481)`이고 유한값이며 같은 섭동 범위를 통과했습니다.
WAV는 `clip(original_float32 + saved_noise, -1, 1)`과 정확히 일치합니다.
원본과 입력 복사본은 SHA-256이 같습니다.

수정 허용된 보고서를 정정한 뒤 실행 전후 **1,057개 파일**의 SHA-256을 비교했고,
변경은 **0개**입니다. 대상은 두 worktree의 기존 소스·미추적 파일,
원본 WAV, 기존 실행 결과, 로컬 모델 8개 파일 등을 포함합니다.
기존 8개 중 보고서 외 7개는 작업 시작 시 백업 해시와도 동일합니다.
공유 `G_0.pth` 하드 링크는 `fsutil hardlink list`로 확인했으며
해시가 보존되었습니다. 모델을 재다운로드하거나 원격 해시 재검증을 수행한 것은 아닙니다.

## STOI 준비 상태

`origin/feature/metrics`의 `27ae06d1fe304320c671eeeb15338e8c23b36ac6`을
읽기만 했습니다. 필요한 코드·버전·입력 계약·실행 예시는
`docs/SPEC_STOI_HANDOFF_KO.md`에 정리했습니다.
감사 폴더에 실제 새 결과를 가리키는 `spec1_metrics_manifest.json`,
`spec10_metrics_manifest.json`을 생성하고 입력 파일 존재를 확인했습니다.

실제 STOI, Gaussian 변환 평가, metrics runner는 실행하지 않았습니다.
`metrics.before`는 원본 대비 보호 음성, `metrics.after`는 원본 대비
보호 음성+Gaussian(std 0.005) 점수가 됩니다.
평가 환경은 NumPy 1.26.4를 요구하므로 기존 GPU 환경과 분리하는 것이 다음 작업입니다.

## 오류 및 미완료 범위

첫 감사 시도에서 WSL Git이 Windows 형식의 worktree gitdir를 해석하지 못해
`fatal: not a git repository`로 종료했습니다. 스모크 시작 전 실패였습니다.
저장소 설정을 변경하지 않고 Windows Git으로 파일 목록을 기록하고 WSL Python이
그 목록을 읽도록 감사 wrapper를 수정한 뒤 재실행해 성공했습니다.
이 초기 실패를 성공으로 집계하지 않았습니다.

공개 G_0 체크포인트의 `emb_g.weight` 누락은 여전하며 seed 기반 초기화 값을
유지합니다. 학습된 화자 임베딩 및 보호 효과는 확인되지 않았습니다.
전체 protect CLI의 판별기·WavLM 초기화, PyTorch 2.7 torch.load 호환성,
전체 데이터·200회·긴 음성·SafeSpeech 모드·한국어·XTTS-v2는 이번 검증 대상이 아닙니다.
원본 `save_audio.py`는 filelist를 다시 쓰므로 이번에 실행하지 않았습니다.

## 커밋 후보와 제외 대상

승인 후 커밋 후보는 기존 8개와 새 문서 2개, 총 10개입니다.

- `.gitignore`
- `requirements-wsl-base.txt`, `requirements-wsl-perturbation.txt`
- `scripts/check_gpu_audio.py`, `scripts/download_setup_models.py`
- `scripts/smoke_perturbation.py`, `scripts/validate_spec_pair.py`
- `docs/SPEC_VALIDATION_20261030_KO.md`
- `docs/SPEC_STOI_HANDOFF_KO.md`
- `docs/SPEC_RESUME_VALIDATION_20261030_KO.md`

누락 integration 문서 3개는 현재 복사하지 않았으므로 후보에 포함하지 않습니다.
`runs/` 전체(새 WAV, noise, report, validation, manifest, 감사 wrapper,
백업 포함), `checkpoints/`, 모델 가중치, BERT 특징, 캐시와 가상환경은 제외합니다.
실제 파일에 대해 `git check-ignore`로 runs와 G_0.pth 제외를 확인했습니다.
현재 manifest는 로컬 실행 경로를 담은 감사 자료이므로 추적하지 않습니다.

커밋·Push·stage·Fetch·브랜치 병합·다른 브랜치 파일 복사는 수행하지 않았습니다.
현재 브랜치 HEAD와 master는 `3345f42`, integration-setup은 `be448f2`로
유지됩니다. 추적 파일 수정과 staged 변경은 없고 위 후보는 모두 미추적입니다.
원래 SafeSpeech worktree도 깨끗한 상태입니다.
