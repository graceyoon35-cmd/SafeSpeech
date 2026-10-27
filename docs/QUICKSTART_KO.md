# SafeSpeech 팀원용 시작 안내

> 이 문서는 2026-10-25 WSL CUDA 환경의 설치·검증 안내입니다. 최신 누적 현황과
> 새 팀원의 브랜치 선택·환경별 시작 순서는 [통합 개발 보고서](DEVELOPMENT_PROGRESS_KO.md)를
> 먼저 읽습니다. Mac CPU 담당자는 [CPU 실험 안내](EXPERIMENT_PIPELINE_KO.md)를 따릅니다.
> 아래 완료 상태와 커밋·Push 관련 표현은 최초 작성 시점의 기록입니다.

공통 개발 환경과 2학년 A의 섭동 생성 실행 준비를 위한 안내입니다.
평가 지표·강건성 실험·모델 학습은 각 담당자 영역이며 이 설정에서 구현하지 않았습니다.

## 현재 완료 상태

- WSL2 Ubuntu 22.04, Python 3.10, 일반 사용자 `safespeech`
- RTX 5060 Ti 8GB에서 PyTorch/TorchAudio 2.7.1 + CUDA 12.8 연산 확인
- 공통 오디오 처리와 패키지 의존성 검사 통과
- 모델 3종, 8개 파일 다운로드 및 해시 확인
- 짧은 영어 음성 하나로 BERT 특징 생성 및 SPEC 1회·10회 업데이트 확인

환경 구축은 완료했습니다. 전체 논문 재현과 보호 성능 검증은 완료하지 않았습니다.
현재 설치는 이 PC에서 확인한 조합이며 다른 GPU에서는 별도 호환성 확인이 필요합니다.

## 팀원 PC 설치 순서

WSL2 Ubuntu 22.04와 NVIDIA GPU의 WSL 인식이 준비된 상태에서 진행합니다.
Windows PowerShell과 Ubuntu 터미널을 구분합니다. 경로와 사용자 이름은 PC에 맞춥니다.
이미 만든 환경에서는 가상환경을 다시 생성할 필요 없이 활성화합니다.

1. Ubuntu에서 `whoami`, `python3 --version`, `nvidia-smi`로 계정·Python·GPU를 확인합니다.
2. 관리자 승인을 받아 시스템 패키지를 설치합니다.

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip build-essential ffmpeg libsndfile1
```

3. Ubuntu 일반 사용자로 가상환경을 만들고 활성화합니다.

```bash
python3 -m venv "$HOME/.venvs/safespeech"
source "$HOME/.venvs/safespeech/bin/activate"
python -m pip install --upgrade pip
```

4. 저장소 루트로 이동하고 GPU 패키지를 먼저 설치합니다.

```bash
cd /mnt/c/Users/윤채영/SafeSpeech
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-wsl-perturbation.txt
python -m pip check
python scripts/check_gpu_audio.py
```

마지막 스크립트의 `status`가 `PASS`인지 확인합니다.
원본 `requirements.txt`나 `bert_vits2/requirements.txt`를 추가 설치하면
PyTorch·librosa 버전이 바뀔 수 있으므로 이번 환경과 혼합하지 않습니다.
CUDA Toolkit이나 WSL 안의 NVIDIA 드라이버는 이번 설정에서 별도로 설치하지 않았습니다.

5. 약 2.58GB의 모델을 다운로드하고 검증합니다. 캐시 등을 위한 추가 공간도 필요합니다.

```bash
python scripts/download_setup_models.py
```

스크립트는 고정 리비전을 사용하고 기존 파일은 덮어쓰지 않습니다.
기존 파일의 크기나 해시가 다르면 실패하며 해당 파일을 보존합니다.
메타데이터 조회와 다운로드에는 인터넷 연결이 필요합니다.

6. 단일 음성으로 섭동 생성 실행을 확인합니다.

```bash
python scripts/smoke_perturbation.py --iterations 10
```

실행마다 새 `runs/smoke_*` 폴더가 생깁니다. 원본 음성 대신 복사본을 사용합니다.
결과는 `protected.wav`, `perturbation.pt`, `report.json`입니다.
이 스크립트는 영어 특징 함수와 원본 `protect.perturb()`를 사용합니다.
전체 `protect.py` CLI의 판별기·WavLM 초기화나 학습·평가는 실행하지 않습니다.

## 일상적인 개발 시작

VS Code에서 WSL 창으로 저장소를 열고 Python 인터프리터를
`/home/safespeech/.venvs/safespeech/bin/python`으로 선택합니다.
Ubuntu 터미널에서 가상환경을 활성화한 뒤 필요한 명령을 실행합니다.

```bash
source "$HOME/.venvs/safespeech/bin/activate"
cd /mnt/c/Users/윤채영/SafeSpeech
```

기본 사용자 설정은 `/etc/wsl.conf`에 저장했습니다. WSL 재시작은 자동 수행하지 않았습니다.
기본 사용자 적용 확인이 남아 있더라도 PowerShell의
`wsl -d Ubuntu-22.04 -u safespeech`로 해당 사용자에 직접 접속할 수 있습니다.

## 확인된 제약과 다음 담당 작업

- 공개 `G_0.pth`에 `emb_g.weight`가 없습니다. 원본 로더와 동일하게 초기화 상태를
  유지해 실행했습니다. 모델 담당자와 사용할 화자 임베딩·체크포인트를 확인해야 합니다.
- 실행 성공은 보호 효과를 입증하지 않습니다. 긴 음성, 전체 108개, 200회 최적화,
  SafeSpeech 모드와 전체 CLI는 아직 검증하지 않았습니다.
- 검증한 단일 음성은 약 1.02초이며 10회 실행의 GPU 텐서 메모리 최고치는 약 614MiB입니다.
  전체 GPU 사용량이나 다른 입력의 메모리 요구량을 뜻하지 않습니다.
- 원본 음성 108개의 목록 경로와 파일 존재를 확인했습니다. 메타데이터는 모두 24kHz·모노·PCM_16입니다.
  원본 데이터 옆에 전체 BERT 특징을 생성한 상태는 아닙니다.

## 공유할 파일

`.gitignore`, `requirements-wsl-*.txt`, `scripts/`, `docs/`가 이번 준비 작업의 공유 대상입니다.
모델·실행 결과·캐시는 `.gitignore`로 제외합니다. 기존에 추적된 음성은 유지합니다.
파일 삭제, stage, 커밋, Push는 수행하지 않았습니다. 팀원에게 Git으로 공유하려면
변경 목록을 검토하고 사용자 확인 후 커밋·Push해야 합니다.

세부 설치 이력과 검증 기록은 [WSL_SETUP_KO.md](WSL_SETUP_KO.md)에 있습니다.
설정·검증 결과와 미검증 범위는 [총정리 문서](SETUP_VALIDATION_SUMMARY_KO.md)에 있습니다.
