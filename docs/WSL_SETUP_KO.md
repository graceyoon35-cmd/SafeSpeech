# SafeSpeech 공통 개발 환경 구축 기록

이 문서는 공통 개발 환경과 2학년 A의 섭동 생성 개발 준비를 다룹니다.
평가 지표, 강건성 실험, 모델 학습 코드는 각 담당자 영역입니다.

## 확인한 환경 (2026-10-25)

- Windows 저장소: `C:\Users\윤채영\SafeSpeech`
- WSL 저장소: `/mnt/c/Users/윤채영/SafeSpeech`
- 브랜치: `feature/integration-setup`
- WSL2 Ubuntu 22.04, Python 3.10
- Ubuntu 사용자: `safespeech` (sudo 그룹 포함)
- GPU: NVIDIA RTX 5060 Ti 8GB. WSL의 `nvidia-smi`에서 인식 확인

## Ubuntu 기본 사용자

사용자가 없다면 root 터미널에서 `adduser safespeech`로 생성하고,
`usermod -aG sudo safespeech`로 sudo 그룹에 추가합니다.
현재 PC에서는 두 조건 모두 충족되어 다시 실행하지 않았습니다.

`/etc/wsl.conf`의 기존 `[boot]` 설정을 유지하고 다음을 추가했습니다.

```ini
[user]
default = safespeech
```

변경 전 파일은 `/etc/wsl.conf.before-safespeech-setup`에 보관합니다.
이미 실행 중인 WSL 세션에는 즉시 반영되지 않을 수 있습니다.
현재 세션을 종료할 필요 없이 PowerShell에서 다음으로 해당 사용자로 접속할 수 있습니다.

```powershell
wsl -d Ubuntu-22.04 -u safespeech
```

기본 접속 계정의 변경을 적용하려면 작업을 저장한 뒤 해당 배포판을 종료하고
다시 시작해야 합니다. 배포판 종료는 실행 중인 작업을 중단하므로 자동 수행하지 않습니다.
재시작 후 Ubuntu에서 `whoami`가 `safespeech`인지 확인합니다.

## 공통 시스템 패키지

아래는 Ubuntu에서 실행하는 재현용 명령입니다. 시스템을 변경하므로 관리자 승인이 필요합니다.

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip build-essential ffmpeg libsndfile1
```

- `python3-venv`: 프로젝트별 Python 가상환경 생성
- `python3-pip`: Python 패키지 설치 도구
- `build-essential`: 일부 라이브러리 설치에 필요한 컴파일 도구
- `ffmpeg`, `libsndfile1`: 오디오 처리 도구 및 라이브러리

## Python/GPU 환경

가상환경 경로는 `/home/safespeech/.venvs/safespeech`입니다.
Windows용 Python과 분리하며, 설치 명령은 `safespeech` 사용자로 실행합니다.
다음은 팀원용 재현 순서입니다. 각 설치 성공을 확인하고 다음으로 진행합니다.

```bash
python3 -m venv /home/safespeech/.venvs/safespeech
source /home/safespeech/.venvs/safespeech/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
cd /mnt/c/Users/윤채영/SafeSpeech
python -m pip install -r requirements-wsl-base.txt
python -m pip check
python scripts/check_gpu_audio.py
```

위 경로의 사용자 이름은 팀원 PC에 맞게 변경합니다.
가상환경은 저장소 밖에 있으므로 Git에 포함되지 않습니다.
VS Code에서는 WSL 창을 열고 Python 인터프리터로
`/home/safespeech/.venvs/safespeech/bin/python`을 선택합니다.
Windows 창에서 Linux 인터프리터를 선택하는 방식은 사용하지 않습니다.

`requirements-wsl-base.txt`는 공통 오디오 처리용 최소 패키지 목록입니다.
전체 연구 코드 실행용 의존성 목록은 아니며, 섭동 생성에 필요한 추가 의존성은 별도 검증합니다.
`scripts/check_gpu_audio.py`는 GPU 행렬곱, 합성곱 역전파, STFT와
메모리 내 WAV 저장·복원, 리샘플링, mel 계산을 확인합니다.
데이터셋, 학습, 평가 또는 모델 다운로드를 실행하지 않습니다.

기존 `requirements.txt`는 `torch==2.3.1`, `torchaudio==2.3.1`을 지정합니다.
RTX 5060 Ti의 Blackwell 지원을 위해 이 파일을 그대로 설치하기 전에 버전 검토가 필요합니다.
[PyTorch 2.7 공식 발표](https://pytorch.org/blog/pytorch-2-7/)에서
Blackwell과 CUDA 12.8 빌드 지원을 확인했습니다.
이는 전체 프로젝트가 해당 버전에서 검증되었다는 의미는 아닙니다.

원본 의존성 파일을 임의로 수정하지 않고, 별도 환경의 설치 계획과 호환성을 확인합니다.
루트와 `bert_vits2/requirements.txt`의 librosa 버전도 서로 다르므로 두 파일을
무작정 함께 설치하지 않습니다.

`nvidia-smi`의 CUDA 표시는 드라이버가 지원하는 버전이며,
Python 환경에 PyTorch나 CUDA Toolkit이 설치되었다는 뜻은 아닙니다.

PyTorch 2.7.1과 같은 버전의 TorchAudio를 CUDA 12.8 인덱스에서 설치합니다.
[공식 설치 조합](https://pytorch.org/get-started/previous-versions/)을 참고했습니다.
이 단계에서는 별도 CUDA Toolkit이나 WSL용 NVIDIA 드라이버를 설치하지 않습니다.

`protect.py`는 import 과정에서 `bert_vits2/text/__init__.py`의
`check_bert_models()`를 호출할 수 있습니다. 파일이 없으면 모델 다운로드가 시작되므로,
`--help`도 순수 환경 점검 용도로 실행하지 않습니다.

## 공통 기반 검증 기록

2026-10-25 공통 기반 검증 결과:

- `safespeech` 사용자 소유의 가상환경 생성 및 패키지 설치 완료
- PyTorch/TorchAudio `2.7.1+cu128`, CUDA 런타임 `12.8`
- GPU `NVIDIA GeForce RTX 5060 Ti`, compute capability `12.0`
- `pip check`: `No broken requirements found.`
- GPU 행렬곱, 합성곱 역전파, STFT, WAV 입출력, 리샘플링, mel 계산 통과

초기 점검 당시의 진행 계획 (아래 후속 기록에서 단일 샘플 검증 완료):

1. 섭동 생성에 필요한 나머지 의존성과 체크포인트 로딩 호환성 확인
2. 데이터·체크포인트의 존재 및 섭동 생성 진입점 확인
3. 8GB VRAM에 맞춘 작은 입력으로 섭동 생성 실행 검증

## 섭동 생성 준비 초기 점검 (2026-10-25, 다운로드 전 상태)

- `filelists/libritts_train_text.txt.cleaned`의 음성 108개 모두 존재합니다.
- 첫 음성은 24,000Hz, 모노, 약 6.05초로 설정의 샘플링 주파수와 일치합니다.
  전체 음성의 형식 검증은 아직 수행하지 않았습니다.
- 대응하는 `.bert.pt` 특징 파일은 0개입니다. 모델 다운로드 후 별도 생성이 필요합니다.
- `checkpoints/base_models` 및 `bert_vits2/slm/wavlm-base-plus`는 없습니다.
- DeBERTa 디렉터리에는 설정 파일만 있고 가중치와 `spm.model`은 없습니다.
- `requirements-wsl-perturbation.txt`의 추가 패키지를 가상환경에 설치했습니다.
  먼저 위의 CUDA PyTorch 설치를 완료한 환경에서 이 파일을 설치합니다.
- `torch_stoi`, `transformers`, `yaml`, `loguru`, `tqdm` import 및
  Transformers의 AutoModel, WavLMModel, DebertaV2Model import를 확인했습니다.
- `pip check`는 통과했습니다. 전체 `protect.py` import/실행과
  BERT 특징 생성용 추가 의존성은 아직 검증 전입니다.

Hugging Face 메타데이터로 확인한 다운로드 목록 (이후 승인받아 다운로드 완료):

| 모델 | 파일 | 합계 바이트 |
| --- | --- | ---: |
| OedoSoldier/Bert-VITS2-2.3 | DUR_0.pth, D_0.pth, G_0.pth, WD_0.pth | 1,321,334,640 |
| microsoft/deberta-v3-large | pytorch_model.bin, spm.model | 876,137,869 |
| microsoft/wavlm-base-plus | pytorch_model.bin, config.json | 377,619,657 |

합계 2,575,092,166바이트 (약 2.58GB, 2.40GiB)이며 캐시와 생성물에 추가 공간이 필요합니다.
확인한 리비전은 순서대로 `3eac455e350982b84267125c370713948b011694`,
`64a8c8eab3e352a784c658aef62be1662607476f`,
`4c66d4806a428f2e922ccfa1a962776e232d487b`입니다.
원본 `download_models.py`는 추가 모델도 받으므로 그대로 실행하지 않았습니다.
ECAPA-TDNN은 현재 확인한 `protect.py` 경로에서 필요하지 않아 이번 계획에서 제외합니다.

원본 README의 섭동 생성 예시는 80GB GPU 기준 batch size 27입니다.
현재 GPU에서 그대로 실행하지 않고 작은 설정부터 검증해야 합니다.
위 모델 다운로드는 완료했습니다. 이후 단일 음성의 섭동 생성은 아래 기록과 같이 검증했습니다. 추가 모델·데이터 다운로드와 전체 데이터 실행은 수행하지 않았습니다.
Git 커밋과 Push는 별도 확인 후 진행합니다.

## 모델 다운로드 완료 및 검증 (2026-10-25)

위 초기 점검 이후 사용자 승인을 받아 목록의 모델 3종, 총 8개 파일을 다운로드했습니다.
표에 기록한 리비전을 지정했으며 기존 파일은 덮어쓰지 않았습니다.
8개 파일 모두 원격 메타데이터와 크기 및 해시가 일치합니다.
LFS 파일은 SHA-256, WavLM의 config.json은 Git blob SHA-1로 검증했습니다.
Windows 마운트 경로에서 Linux 파일 권한 설정 경고가 발생했지만 다운로드는
완료되었고 검증은 모두 통과했습니다. 기존 추적 파일의 변경은 없습니다.

음성 108개의 메타데이터도 모두 확인했습니다. 모두 영어, 24,000Hz, 모노, PCM_16이며
길이는 0.61~9.94초, 총 약 491.23초입니다.

다운로드한 모델은 Git 미추적 상태이며 stage/커밋/Push하지 않았습니다.
모델·캐시·실행 결과는 루트 .gitignore로 제외했습니다. 파일은 디스크에 유지됩니다.
이후 BERT 특징 생성과 작은 입력의 섭동 생성 검증을 완료했습니다. 아래 실행 기록을 참고합니다.

## 첫 섭동 생성 실행 검증 (2026-10-25)

`sentencepiece==0.2.0`을 추가 설치하고 `protect.py`의 오프라인 import를 확인했습니다.
새 `scripts/smoke_perturbation.py`는 영어 음성 하나를 고유한 `runs/smoke_*` 폴더에
복사하고 원본 영어 BERT 특징 함수와 `protect.perturb()`를 실행합니다.
원본 음성, 파일 목록, 모델·학습·평가 코드는 수정하지 않습니다.

재현 명령 (저장소 루트, WSL 가상환경):

```bash
python -m pip install -r requirements-wsl-perturbation.txt
python scripts/smoke_perturbation.py
```

다운로드한 모델이 필요하며 실행 중 네트워크 모델 다운로드는 비활성화합니다.
테스트는 원본 생성기만 사용하며 전체 `protect.py` CLI의 판별기·WavLM 초기화는
포함하지 않습니다. 영어 단일 샘플의 실행 검증으로 범위를 제한합니다.

성공 실행 폴더: `runs/smoke_20261025_024005_466049`

- 입력: `data/LibriTTS/5339/5339_14134_000092_000003.wav`, 약 1.02초
- BERT 특징: `(1024, 33)`
- SPEC, 배치 1개, 업데이트 1회
- epsilon `8/255`, 최대 섭동 절댓값 약 `0.00313726` (범위 내, 0이 아님)
- 측정된 최고 GPU 텐서 할당 메모리 약 602.59MiB (전체 GPU 사용량과 다름)
- 결과: `perturbation.pt`, `protected.wav`, `report.json`

중요한 재현 제약: 공개 G_0.pth에 `emb_g.weight`가 없습니다.
원본 로더는 누락 시 초기화된 값을 유지합니다. 검증 스크립트도 이 항목만
초기화 상태로 허용하며 그 외 누락·추가 키와 크기 불일치는 실패 처리합니다.
이 사실을 report.json에 기록합니다. 학습된 화자 모델이나 보호 효과를
검증한 결과로 해석할 수 없습니다. 화자 임베딩·체크포인트 구성은 모델 담당자와
확인이 필요하며 이를 해결하려고 임의로 모델 학습을 수행하지 않습니다.

Windows 마운트에서 copy2의 파일 시각 보존이 권한 오류를 내어 새 스크립트에서
파일 내용만 복사하도록 했습니다. 실패 실행 폴더는 삭제하지 않았습니다.
이후 10회 업데이트 검증을 완료했습니다. 체크포인트 구성 확인은 모델 담당자와 남아 있습니다.
전체 108개 실행, 200회 최적화, SafeSpeech 모드 및 보호 효과는 아직 검증 전입니다.

## 10회 업데이트 실행 검증

검증 스크립트에 `--iterations` 옵션을 추가했습니다. 기본값은 1이며 양수만 허용합니다.
다음 명령으로 동일한 짧은 음성에 대해 SPEC 업데이트 10회를 완료했습니다.

```bash
python scripts/smoke_perturbation.py --iterations 10
```

- 결과 폴더: `runs/smoke_20261025_024217_002294`
- 최대 섭동 절댓값: `0.0313725508749485` (float32 오차를 고려해 `8/255 + 1e-7` 이내)
- 마지막 업데이트의 mel/NR/KL 손실은 모두 유한한 값입니다.
- GPU 텐서 할당 메모리 최고치: 약 614.39MiB
- 저장한 WAV를 다시 읽어 24,481샘플, 24kHz, 유한한 값,
  출력 진폭 [-1,1] 및 원본 대비 최대 변화량의 범위 충족을 확인했습니다.

화자 임베딩의 초기화 상태, 단일 영어 샘플 및 생성기만 검증한 제약은 동일합니다.
이 결과는 전체 반복 과정의 손실 감소나 보호 효과를 입증하지 않습니다.
원본 함수는 마지막 업데이트 직전의 손실과 업데이트 이후 섭동을 반환합니다.
