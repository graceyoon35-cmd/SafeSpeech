# SPEC 앱 연동 인계 — 2026-11-20

`scripts/protect_wav.py`를 독립 프로세스로 실행하면 실제 WAV를 SPEC 보호 WAV로 변환한다. 엄격한 ε는 최종 PCM16 파일을 기준으로 검사한다. 연구 결과는 [최종 보고서](SPEC_FINAL_REPORT_20261120_KO.md), 이전 검증 상세는 [11월 19일 명세](README_SPEC_HANDOFF_20261119_KO.md)에 있다.

## 검증 환경과 의존성

Windows + Ubuntu-22.04 WSL2, RTX 5060 Ti, Python 3.10 전용 환경에서 검증했다.

```bash
cd '/mnt/c/Users/윤채영/SafeSpeech-desktop-integration-20261118'
PY=/home/safespeech/.venvs/melo-training-20261118/bin/python
ASSETS=/home/safespeech/models/melo-spec-pinned-20261118
"$PY" -m pip check
```

현재 환경을 재사용하면 추가 설치 없이 실행한다. 설치 명세는 `requirements-spec-app-wsl.txt`, 전체 버전 기록은 `docs/spec_app_environment_20261120.txt`다. 새 환경 설치 시 torch/torchaudio `2.7.1+cu128`을 CUDA 12.8 wheel 저장소에서 먼저 설치한 뒤 해당 명세를 설치한다. 시스템 FFmpeg·libsndfile과 고정 backend 자산도 필요하다. 새 기기에서 처음부터 설치하는 전 과정은 이번에 재검증하지 않았다.

기존 `requirements-spec-stoi.txt`는 독립 CPU 테스트 환경용이며 NumPy 1.26.4를 지정한다. 이 GPU 환경의 NumPy 1.23.5와 섞어 설치하지 않는다.

새 전용 환경을 만드는 설치 명령 예시는 다음과 같다. 시스템 FFmpeg·libsndfile과 고정 자산은 별도로 준비한다. 아래 설치 절차 자체는 새 기기에서 재검증하지 않았다.

```bash
python3.10 -m venv /absolute/new-spec-venv
/absolute/new-spec-venv/bin/python -m pip install \
  torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
/absolute/new-spec-venv/bin/python -m pip install \
  -r docs/spec_app_environment_20261120.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128
/absolute/new-spec-venv/bin/python -m pip check
```

`ASSETS` 아래 `model/checkpoint.pth`, `model/config.json`, `effective_loss_config_verified.json`, 고정 MeloTTS 및 팀 backend 소스가 있어야 한다. 자산 내부 구조를 보존한다. 실행 시 고정 해시·revision을 검사하고 모델을 자동 다운로드하지 않는다. 초기 가중치 SHA256: `48e3ff3fd0b5348e095f0468e60ae727507564100f58142ef3a922ead6e0a4d0`. 데이터·가중치는 Git에 포함하지 않으며 로컬 자산과 최신 백업을 통해 인계한다. 자산 준비 과정은 `scripts/prepare_melo_spec_assets.py`와 기존 Melo 인계 문서를 참고한다.

## 실제 실행

다음 명령은 **WSL bash**에서 실행한다. 출력 디렉터리는 존재하지 않는 새 경로를 사용한다.

```bash
"$PY" scripts/protect_wav.py \
  --input runs/final_validation_20261120/app_handoff_examples/input.wav \
  --output-dir runs/spec-app-example-new \
  --assets "$ASSETS" --device cuda --iterations 10 --epsilon 8 --seed 52

# 같은 실행을 제공하는 쉘 예제
bash scripts/example_spec_protect.sh \
  "$PWD/runs/final_validation_20261120/app_handoff_examples/input.wav" \
  "$PWD/runs/spec-app-example-another-new"
```

실제 검증 입력·출력·진단 예제는 `runs/final_validation_20261120/app_handoff_examples/{input.wav,protected.wav,result.json}`에 있다. 이 음성 파일은 로컬 연구용이며 Git에 올리지 않았다. 최신 보완 백업에 포함한다. 임의의 정상 입력 WAV 절대 경로로 바꾸어도 된다. 연구에 사용한 원본 예제는 `runs/desktop_20261118/data/data_set/kss_part_200/train/wavs/kss_hf_00078.wav`다.

Python 앱에서 호출하는 예:

```python
import json
import subprocess

proc = subprocess.run([
    '/home/safespeech/.venvs/melo-training-20261118/bin/python',
    'scripts/protect_wav.py', '--input', '/absolute/input.wav',
    '--output-dir', '/absolute/new-output',
    '--assets', '/home/safespeech/models/melo-spec-pinned-20261118',
    '--device', 'cuda', '--iterations', '10', '--epsilon', '8', '--seed', '52',
], cwd='/mnt/c/Users/윤채영/SafeSpeech-desktop-integration-20261118',
   capture_output=True, text=True)
if proc.returncode:
    raise RuntimeError(proc.stderr)
result = json.loads(proc.stdout.strip().splitlines()[-1])
if result['status'] != 'PASS':
    raise RuntimeError(result)
protected_path = result['output']
```

Windows 앱은 `wsl.exe -d Ubuntu-22.04 -- <Python 경로> <스크립트 WSL 절대 경로> ...`로 호출하고 입력·출력도 WSL 경로로 전달한다. 앱에서 GPU 작업을 비동기 실행하고 취소·오류 표시·배포를 연결하는 작업은 앱팀 담당이다.

## 입출력 계약

| 항목 | 명세 |
|---|---|
| 입력 | mono, WAV/PCM16, 44,100Hz, 1~10초, 완전 무음 제외, 유한 샘플 |
| 전처리 | 호출자가 리샘플·다운믹스·분할. 자동 변환하지 않음 |
| 텍스트 | 대본 입력 불필요 |
| ε | `--epsilon 8`은 정규화 진폭 8/255. 최종 WAV는 정수 code 차이 1028 이하 |
| 반복·seed | iterations 1~200, seed 0~2³²−1 |
| 출력 | `protected.wav`: 입력과 같은 길이·44.1kHz mono PCM16 |
| 진단 | `result.json`, `generation*.json`, `validation.json`, `stoi.json`, backend·gradient 기록 |
| 중간 WAV | `optimized.wav`는 정수 투영 전 진단용. 앱 전달 파일은 `protected.wav` |
| 성공 | 종료 코드 0 **및** 마지막 stdout JSON의 `status=PASS` |
| 오류 | 종료 코드 1. 일부 생성 후 실패하면 `result.json`에 FAIL 기록. 사전 입력 검사 실패 시 결과 폴더가 없을 수 있음 |
| 덮어쓰기 | 기존 출력 디렉터리 거부. 실패 시 일부 결과가 남으면 새로운 경로로 재시도 |
| stdout | backend 로그가 앞에 올 수 있으므로 전체 stdout을 단일 JSON으로 파싱하지 않음 |

`result.json`에는 입출력 경로·SHA256, seed·반복·ε, 최종 `strict_linf`·`stoi`·`snr_db`·길이·형식을 기록한다. 진단 JSON의 세부 구조는 예제 파일을 따른다.

## 검증과 알려진 한계

```bash
"$PY" -m unittest discover -s scripts -p 'test_*.py' -v
"$PY" -m pip check
```

오늘 전체 자동 테스트 99개, pip check, 실제 WAV GPU 변환, 최종 ε·SNR 독립 검사, 기존 출력 재사용 거부와 해시 보존을 확인했다. 증거는 `runs/final_validation_20261120/`에 있다. 보호 입력 200개 보정본의 ε 초과는 0이며 별도 1,000-step 모델 학습·평가 결과를 최종 보고서에 수록했다.

GPU 기본 모드는 동일 입력·seed에서도 WAV 바이트 동일성을 보장하지 않는다. `--deterministic`은 현재 `reflection_pad1d_backward_out_cuda` 지원 부재로 실패하는 진단 옵션이다. 성공 보장 옵션으로 표시하지 않는다. CPU 경로의 완료 시간·성능은 미검증이다.

KSS 단일 화자 내부 비상업적 연구다. SPEC_10의 입력 품질 비용이 있으므로 무손실·불가청·복제 완전 차단으로 표시하지 않는다. 사람 청취·별도 화자 동의·외부 음성/가중치 공개 승인 상태는 기존 기록을 유지한다.
