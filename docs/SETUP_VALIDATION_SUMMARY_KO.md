# SafeSpeech 초기 설정·검증 총정리

작성일: 2026-10-25

## 현재 상태

공통 개발 환경 구축을 완료했고, 짧은 영어 음성 하나로 원본 SPEC 섭동 생성 함수를
1회 및 10회 실행했습니다. 전체 논문 재현이나 보호 성능 검증을 완료한 상태는 아닙니다.

이 작업의 담당 범위는 공통 실행 환경 준비와 **2학년 A의 섭동 생성 개발 준비**입니다.
평가 지표 구현, 강건성 실험, 모델 학습 코드는 수정하거나 실행하지 않았습니다.

## 설정한 개발 환경

| 항목 | 설정·확인 결과 |
| --- | --- |
| 운영체제 | Windows + WSL2 Ubuntu 22.04 |
| Windows 저장소 | `C:\Users\윤채영\SafeSpeech` |
| WSL 저장소 | `/mnt/c/Users/윤채영/SafeSpeech` |
| 작업 브랜치 | `feature/integration-setup` |
| Ubuntu 사용자 | `safespeech`, sudo 그룹 포함 |
| 기본 사용자 설정 | `/etc/wsl.conf`에 `default = safespeech` 저장 |
| 설정 백업 | `/etc/wsl.conf.before-safespeech-setup` |
| Python | 3.10.12 |
| 가상환경 | `/home/safespeech/.venvs/safespeech` |
| GPU | NVIDIA GeForce RTX 5060 Ti 8GB |
| PyTorch / TorchAudio | `2.7.1+cu128` / `2.7.1+cu128` |
| PyTorch CUDA 런타임 | 12.8 |
| GPU compute capability | 12.0 |

WSL 재시작은 자동 실행하지 않았습니다. 재시작 이후 기본 사용자 적용 확인은 남아 있습니다.
`wsl -d Ubuntu-22.04 -u safespeech`로 사용자를 직접 지정한 실행은 확인했습니다.

시스템에는 `python3-venv`, `python3-pip`, `build-essential`, `ffmpeg`,
`libsndfile1`을 설치했습니다. CUDA Toolkit이나 WSL 안의 NVIDIA 드라이버는
별도로 설치하지 않았습니다.

Python 패키지는 별도 목록으로 관리합니다.

- `requirements-wsl-base.txt`: NumPy 1.23.5, librosa 0.10.0, SoundFile 0.12.1, SciPy 1.13.1
- `requirements-wsl-perturbation.txt`: 위 목록과 GPU 패키지, torch_stoi, tqdm,
  Transformers 4.46.3, Hugging Face Hub, PyYAML, loguru, sentencepiece

원본 `requirements.txt`와 `bert_vits2/requirements.txt`는 수정하지 않았습니다.
현재 패키지 목록은 이번 실행 범위를 위한 것이며 다른 담당자의 전체 실행 환경까지
검증한 목록은 아닙니다.

## 완료한 검증

| 검증 대상 | 결과 | 범위 |
| --- | --- | --- |
| WSL 사용자 | 정상 | `safespeech`로 직접 명령 실행 |
| WSL GPU 인식 | 정상 | `nvidia-smi`에서 RTX 5060 Ti 확인 |
| Python 패키지 의존성 | 통과 | `pip check`: 충돌 없음 |
| GPU 계산 | 통과 | 행렬곱, 합성곱 역전파, STFT |
| 오디오 처리 | 통과 | 메모리 내 WAV 저장·복원, 리샘플링, mel 계산 |
| 모델 파일 무결성 | 통과 | 8개 파일의 크기 및 원격 해시 일치 |
| 원본 코드 import | 통과 | 오프라인으로 `protect.py` import |
| BERT 특징 생성 | 통과 | 약 1.02초 영어 음성, 특징 크기 `(1024, 33)` |
| SPEC 단일 업데이트 | 통과 | 원본 `protect.perturb()` 1회 실행 |
| SPEC 반복 업데이트 | 통과 | 같은 입력으로 10회 실행, 최종 손실 유한값 확인 |
| 저장 음성 확인 | 통과 | 길이·24kHz 유지, 진폭 및 최대 변화량 범위 충족 |
| Git 제외 규칙 | 통과 | 다운로드 모델·실행 결과가 커밋 후보에서 제외됨 |

음성 파일은 목록의 108개가 모두 존재합니다. 메타데이터는 모두 영어·24kHz·모노·PCM_16이며,
길이는 0.61~9.94초, 총 약 491.23초입니다. 108개 전체의 섭동을 생성한 것은 아닙니다.

## 다운로드한 모델

| 모델 | 저장 위치 | 파일 수 | 크기 |
| --- | --- | ---: | ---: |
| BERT-VITS2 | `checkpoints/base_models` | 4 | 약 1.32GB |
| DeBERTa-V3 Large | `bert/deberta-v3-large` | 2 | 약 0.88GB |
| WavLM Base Plus | `bert_vits2/slm/wavlm-base-plus` | 2 | 약 0.38GB |

총 2,575,092,166바이트, 약 2.58GB입니다. 캐시 공간은 별도입니다.
고정 리비전과 파일 목록은 `scripts/download_setup_models.py`에 기록했습니다.
LFS 파일은 SHA-256, WavLM 설정 파일은 Git blob SHA-1으로 검증했습니다.
ECAPA-TDNN은 이번 섭동 생성 함수 검증에 사용하지 않아 다운로드하지 않았습니다.

## 섭동 생성 실행 결과

입력: `data/LibriTTS/5339/5339_14134_000092_000003.wav`

| 항목 | 1회 업데이트 | 10회 업데이트 |
| --- | ---: | ---: |
| 음성 길이 | 약 1.02초 | 약 1.02초 |
| 배치 크기 | 1 | 1 |
| 모드 | SPEC | SPEC |
| epsilon | `8/255` | `8/255` |
| 최대 섭동 절댓값 | 약 0.00313726 | 약 0.03137255 |
| 최고 GPU 텐서 할당 메모리 | 약 602.59MiB | 약 614.39MiB |

float32 오차를 고려한 허용 기준은 `epsilon + 1e-7`입니다.
저장된 10회 결과 음성은 24,481샘플, 24kHz이며 원본 대비 최대 변화량도 기준 안에 있습니다.
GPU 수치는 PyTorch의 텐서 할당량으로, 전체 GPU 사용량이나 긴 음성의 요구량이 아닙니다.

성공 실행 폴더:

- `runs/smoke_20261025_024005_466049`: 1회
- `runs/smoke_20261025_024217_002294`: 10회

각 폴더에 복사한 입력, BERT 특징, `perturbation.pt`, `protected.wav`, `report.json`이 있습니다.
이 결과물은 로컬 파일이며 Git 공유 대상에서 제외했습니다.

## 확인된 제약과 미검증 항목

공개 생성기 체크포인트 `G_0.pth`에 **`emb_g.weight`가 없습니다.**
원본 로더는 이 값이 없으면 초기화된 값을 유지합니다. 검증 스크립트도 같은 방식으로
이 한 항목만 허용했고, 결과 보고서에 기록했습니다. 모델 담당자와 사용할 화자 임베딩 및
체크포인트 구성을 확인해야 합니다. 이를 해결하기 위한 모델 학습은 수행하지 않았습니다.

이번 검증은 원본 생성기와 `protect.perturb()`의 실행 확인입니다.
전체 `protect.py` CLI에서 판별기·WavLM을 초기화하는 경로까지 검증한 것은 아닙니다.
다음 항목은 아직 확인하지 않았습니다.

- 전체 108개 음성의 BERT 특징·섭동 생성
- 더 긴 음성에서의 메모리 요구량과 실행 안정성
- 200회 업데이트 및 SafeSpeech 모드
- 보호 효과, 음질 평가, 강건성 실험, 모델 학습
- 다른 팀원 PC에서의 신규 설치

원본 함수가 반환하는 손실은 마지막 업데이트 직전 값입니다.
손실 감소나 보호 효과를 이번 성공 결과만으로 판단할 수 없습니다.

## 추가한 공유 파일과 사용 방법

| 파일 | 역할 |
| --- | --- |
| `.gitignore` | 모델·캐시·실행 결과·가상환경 제외 |
| `requirements-wsl-base.txt` | 공통 오디오 라이브러리 목록 |
| `requirements-wsl-perturbation.txt` | 섭동 생성 검증용 추가 패키지 |
| `scripts/check_gpu_audio.py` | GPU·오디오 기본 점검 |
| `scripts/download_setup_models.py` | 고정 버전 모델 다운로드·무결성 검사 |
| `scripts/smoke_perturbation.py` | 단일 영어 음성의 SPEC 실행 점검 |
| `docs/QUICKSTART_KO.md` | 팀원 설치·실행 순서 |
| `docs/WSL_SETUP_KO.md` | 세부 작업 이력과 오류 해결 기록 |
| `docs/SETUP_VALIDATION_SUMMARY_KO.md` | 현재 설정·검증 총정리 (이 문서) |

설치를 시작할 때는 [팀원용 시작 안내](QUICKSTART_KO.md)를 먼저 봅니다.
기존 프로젝트 코드와 음성 데이터는 수정하지 않았고, 실패 실행 폴더도 삭제하지 않았습니다.

## GitHub에서 팀원에게 공유하기

위 문서와 스크립트를 커밋한 뒤 원격 브랜치로 Push하면 팀원들이 GitHub에서 볼 수 있습니다.
`feature/integration-setup`으로 Push하면 GitHub에서도 **그 브랜치를 선택해야** 보입니다.
기본 브랜치에서는 병합 전까지 보이지 않습니다. 비공개 저장소라면 팀원에게 저장소 접근 권한이 필요합니다.

모델·가상환경·실행 결과는 Push되지 않습니다. 팀원은 안내서에 따라 자기 PC에 설치하고
모델 다운로드 스크립트를 실행해야 합니다. 현재 로컬 설치 자체가 Git으로 전달되지는 않습니다.
이 문서 작성 시점에는 stage·커밋·Push를 수행하지 않았습니다.
