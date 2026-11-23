# Voice Protect Windows 앱

WAV·MP3·MP4 파일을 최대 10개까지 선택해 음성을 보호합니다. 완료 후 원본과 결과를 재생하고 결과를 별도 파일로 저장할 수 있습니다. 설정에서 보호 강도(ε)를 지정합니다. 기본값은 모델팀이 검증한 8입니다.

## 실행 환경

Windows에 Python 3.10 이상과 FFmpeg·ffprobe가 필요합니다. 모델은 WSL2와 NVIDIA GPU에서 실행합니다. [모델팀의 실행 가이드](https://github.com/graceyoon35-cmd/SafeSpeech/blob/research/spec-final-validation-20261119/docs/README_SPEC_HANDOFF_20261120_KO.md)에 따라 전용 WSL 환경과 고정 모델 자산을 준비하고, 같은 브랜치의 SPEC 저장소를 Windows 로컬 드라이브에 둡니다.

PowerShell에서 실행합니다.

```powershell
cd ui
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

첫 실행 전 `main.py` 옆에 `voiceprotect.ini`를 저장합니다. 실행 파일로 배포할 때는 `VoiceProtect.exe` 옆에 둡니다. 아래 경로를 해당 PC에 맞게 바꾸세요. `repository`는 Windows 경로이고 `python_executable`과 `assets`는 선택한 WSL 배포판 안의 경로입니다.

```ini
[spec]
repository = C:\SafeSpeech\spec-runtime
distribution = Ubuntu-22.04
python_executable = /home/safespeech/.venvs/melo-training-20261118/bin/python
assets = /home/safespeech/models/melo-spec-pinned-20261118
epsilon = 8
```

환경 변수 `VOICEPROTECT_SPEC_REPO`, `VOICEPROTECT_WSL_DISTRO`, `VOICEPROTECT_SPEC_PYTHON`, `VOICEPROTECT_SPEC_ASSETS`, `VOICEPROTECT_SPEC_EPSILON`가 있으면 해당 설정에 우선합니다. 결과·진단 파일의 기본 위치는 `%LOCALAPPDATA%\VoiceProtect\results`입니다. INI의 `output_root` 또는 `VOICEPROTECT_RESULT_DIR`로 변경할 수 있습니다.

## 파일 처리

김소희의 [오디오 파이프라인](https://github.com/sohui2/SafeSpeech-audio-pipeline/blob/c439d6f2f086eac6d8df5749b0f8ae8ef1267708/audio_pipeline.py)이 음성을 44.1 kHz 모노로 추출하고, 기본 8초 단위로 0.5초씩 겹쳐 나눕니다. 앱은 각 조각을 PCM16 WAV로 만들어 [모델팀의 브리지](https://github.com/graceyoon35-cmd/SafeSpeech/blob/research/spec-final-validation-20261119/scripts/spec_desktop_bridge.py)를 통해 WSL GPU 실행기를 호출합니다. 브리지가 출력 해시와 길이·강도를 검증한 후 파이프라인이 결과를 병합합니다.

원본 전체가 1초보다 짧으면 무음으로 1초까지 채워 처리하고 원래 길이로 자릅니다. 완전 무음 조각은 그대로 유지하고, 파일 전체가 무음이면 실패로 표시합니다. MP3·MP4 재인코딩 후의 보호 성능은 별도 평가가 필요합니다. 앱에는 모델 신뢰도 수치가 제공되지 않습니다.

## Windows 실행 파일

```powershell
.\.venv\Scripts\python.exe -m pip install "pyinstaller>=6.12,<7"
.\build_windows.ps1
```

배포 단위는 `dist\VoiceProtect` 폴더 전체입니다. 사용 중인 Windows FFmpeg·ffprobe를 함께 묶습니다. WSL 실행 환경과 모델 자산은 각 PC에서 준비합니다.

나눔스퀘어 글꼴의 라이선스는 [assets/fonts/OFL.txt](assets/fonts/OFL.txt)에 있습니다.
