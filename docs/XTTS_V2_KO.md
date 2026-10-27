# XTTS-v2 한국어·영어 음성 복제 연동 준비

작성일: 2026-10-27 · 개발 브랜치: `feature/xtts-v2`

## 1. 이번 완료 범위

Mac에서 **모델 없이 검증 가능한 연동 코드와 Mock 테스트**를 구현했습니다.
실제 XTTS 로딩·CUDA VRAM 측정·한국어/영어 합성은 모두 미실행입니다.
사용 허락받은 실제 참조 WAV 경로가 아직 없으며, Windows PC에 원격 접속할 수 없습니다.
Mock 신호를 실제 합성 또는 SPEC 보호 음성이라고 표시하지 않습니다.

| 항목 | 상태 |
| --- | --- |
| `ko` / `en` 추론 인터페이스 | 구현, Mock 검증 |
| clean/protected 동일 조건 비교 | 구현, Mock 검증 |
| 입력 검사·출력 덮어쓰기 방지·입력 해시 | 구현, Mock 검증 |
| CUDA 부재·모델 로딩 OOM 변환·실패 JSON | 구현, Mock 검증 |
| 의존성 파일·Windows 설치 절차 | 작성, 설치 조합 실제 검증 대기 |
| 실제 CUDA 로딩·VRAM·음질·한국어/영어 WAV | 미실행 |
| SPEC의 XTTS 전이 보호 효과 | 미검증 |

## 2. 브랜치와 기존 BERT-VITS2 구조

원격 fetch 후 `feature/bert-vits2`의 로컬·원격이 모두 `3345f42`임을 확인했습니다.
`master` 대비 고유 커밋, `feature/metrics`에 미포함된 커밋은 없습니다.
미커밋 변경도 없었습니다. 공통 오디오·해시 도구를 재사용하기 위해 이를 포함하는
`feature/metrics`의 `27ae06d`에서 새 `feature/xtts-v2`를 만들었습니다.
`master` 및 다른 팀원 브랜치의 커밋은 변경하지 않습니다. 초기 구현 후 승인을 받았으며, 이 문서는 해당 브랜치의 마감 커밋·Push 대상입니다.

기존 구조는 다음과 같이 유지합니다.

| 영역 | 기존 연결 | XTTS와의 관계 |
| --- | --- | --- |
| 섭동 생성 | `protect.py`, `toolbox.py`, `bert_vits2/models_noise.py` | BERT-VITS2 대리 모델을 계속 사용 |
| 학습 | `train.py`, `bert_vits2/`, 데이터셋별 JSON | 삭제·수정하지 않음 |
| 기존 통합 평가 | `evaluate.py`가 BERT-VITS2 합성 후 MCD/WER/SIM | 수정하지 않고 XTTS 별도 runner 추가 |
| 텍스트·특징 | `preprocess_text.py`, `bert_gen.py`, `bert/` | XTTS 추론 입력으로 `.bert.pt`를 요구하지 않음 |
| 모델 파일 | `checkpoints/base_models`, DeBERTa, WavLM, ECAPA | XTTS checkpoint와 공유하지 않음 |

이번 전환은 **음성 복제 평가 대상 모델을 추가하는 작업**입니다. SPEC 최적화 모델을
XTTS로 바꾸거나 BERT-VITS2 학습 코드를 이식한 것은 아닙니다. 원본 SPEC는 현재 영어
초기 실행만 검증되어 있어, 한국어 SPEC 생성 자체는 별도 과제로 남습니다.

## 3. 공식 지원·라이선스·호환성 조사

공식 [XTTS-v2 모델 카드](https://huggingface.co/coqui/XTTS-v2)는
한국어 `ko`, 영어 `en`을 포함한 다국어 음성 복제와 24kHz 출력을 명시합니다.
참조 음성은 모델 카드에서 약 6초를 예시로 듭니다. 1초 입력을 코드가 허용하더라도
짧은 참조에서의 합성 품질을 보장하지 않습니다.

유지보수 구현은 [idiap/coqui-ai-TTS](https://github.com/idiap/coqui-ai-TTS)의
PyPI 패키지 `coqui-tts`를 선택했습니다. 기존 구 패키지 `TTS`와 이름은 다르지만
Python import namespace는 `TTS`이므로 둘을 같은 환경에 설치하지 않습니다.
[0.28.0 패키지 정의](https://github.com/idiap/coqui-ai-TTS/blob/v0.28.0/pyproject.toml)는
Python 3.10 이상 3.15 미만, NumPy 1.26 이상, librosa 0.11 이상,
Transformers 4.57 이상을 요구합니다. 코드 라이선스는 MPL-2.0입니다.

[공식 설치 문서](https://coqui-tts.readthedocs.io/en/latest/installation.html)에 따르면
PyTorch는 따로 설치해야 하며 2.2 이상을 대상으로 합니다. PyTorch 2.9 이상은
torchcodec도 필요합니다. 이번 후보는 기존 GPU 검증 기록이 있는 torch/torchaudio
2.7.1 + cu128을 사용하여 torchcodec 의존 경로를 피합니다.
이는 **XTTS 설치·실행 성공을 확인한 조합이 아니라 검증할 후보**입니다.
문서의 테스트 OS는 Ubuntu 24.04이며 팀의 WSL Ubuntu 22.04는 후속 실행 확인이 필요합니다.

SafeSpeech 기존 WSL 환경의 NumPy 1.23.5·librosa 0.10·Transformers 4.46.3과
충돌하므로 별도 `.venv-xtts`가 필요합니다. Mac의 기존 torch 2.14.1/torchaudio
2.11.0 조합도 XTTS 설치에 재사용하지 않습니다. 이번에 설치·변경한 패키지는 없습니다.
`requirements-xtts.txt`는 주요 패키지 버전을 지정한 목록이며 전체 전이 의존성을
고정한 lock 파일은 아닙니다. Windows 검증 후 별도 lock을 만드는 것이 후속 작업입니다.

### 모델 CPML 조건

[공식 CPML 1.0.0 원문](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt)을
확인했습니다. 모델과 출력은 비상업적 사용에 한정되며, 모델이나 출력 전달 시
해당 조건 또는 라이선스 URL을 함께 제공해야 합니다. 코드 MPL 라이선스가 모델의
사용 제한을 없애지 않습니다. 수익·보수·연구비 등 연구 조건에 따라 단순히 “학교 연구”라는
명칭만으로 적합성을 단정할 수 없습니다. 팀의 실제 용도에 맞게 원문 조건을 확인합니다.
상업 배포 가능 모델로 취급하지 않으며 모델·합성 출력도 무조건 자유 배포하지 않습니다.

참조 음성 사용 허락과 CPML 수락은 별도입니다. 예제는 둘 다 `false`이고,
사용자가 실제 확인 후에만 설정합니다. 코드가 자동으로 약관 수락 환경변수를 설정하거나
라이선스 확인을 우회하지 않습니다. 모델 자동 다운로드도 하지 않습니다.

## 4. 새 코드와 입력 구조

| 경로 | 역할 |
| --- | --- |
| `scripts/xtts_inference.py` | 지연 모델 로딩, `XttsSynthesizer.synthesize()` 및 단일 합성 CLI |
| `scripts/run_xtts_pair.py` | clean/protected × prompt 비교와 결과 JSON |
| `scripts/test_xtts.py` | 모델 없는 테스트 8개 |
| `experiments/xtts_pair_pending.json` | 실제 참조·라이선스 확인 전 대기 설정 |
| `requirements-xtts.txt` | 독립 환경용 의존성 후보 |
| `.gitignore` | `.venv-xtts/` 제외 추가; 기존 `runs/`, `checkpoints/` 제외 유지 |

재사용 API:

```python
from xtts_inference import XttsSynthesizer

synth = XttsSynthesizer(
    model_dir="checkpoints/xtts-v2", device="cuda:0",
    consent_confirmed=True, cpml_accepted=True,
)
result = synth.synthesize(
    reference_wav="runs/xtts_inputs/clean.wav",
    text="안녕하세요. 음성 보호 기술을 연구합니다.",
    language="ko", output_path="runs/xtts_single/ko.wav", seed=42,
)
```

이 예시는 실제 참조·모델·권한 확인 후 사용합니다. 모듈을 직접 import할 때는
`scripts`가 Python 검색 경로에 있어야 합니다. 아래 CLI는 저장소 루트에서 바로 실행합니다.

Manifest의 참조 경로와 model_dir는 manifest 디렉터리 기준입니다.
원본/보호 경로를 바꿔도 텍스트·언어·seed·decoding/conditioning은 같은 값을 씁니다.
기본 prompt는 `ko` 한 문장과 `en` 한 문장으로 총 출력 4개를 준비합니다.
clean/protected는 같은 길이·sample rate의 별도 파일이어야 합니다. 실제 SPEC 출처는
생성 보고서와 파일 전달 이력으로 확인해야 하며 이름만으로 증명되지 않습니다.
테스트의 protected 파일은 합성한 수학 신호일 뿐 실제 SPEC가 아닙니다.

실제 백엔드는 공식 `XttsConfig`, `Xtts.load_checkpoint()`,
`get_conditioning_latents()`, `inference()` API를 사용합니다.
모델은 한 번 로드하고 참조마다 conditioning을 다시 계산하여 참조 간 캐시 혼입을 피합니다.
seed는 각 합성 전에 재설정하고 greedy decoding(`do_sample=False`)을 사용합니다.
설정은 코드 상수와 출력 runtime에 기록합니다. GPU 연산까지 비트 단위 결정성을
보장하는 설정은 아니므로 실제 반복 검증이 필요합니다.
참조 처리는 22,050Hz 내부 로딩·최대 30초·GPT conditioning 6초를 명시하며,
이는 기존 strict STOI 오디오 처리와 달리 XTTS 내부의 리샘플링/부분 사용입니다.
이 전처리가 섭동에 미치는 영향도 후속 실험 대상입니다.

## 5. Windows WSL CUDA 설치 및 실행 순서

아래 명령은 **후속 Windows 실행용이며 이번 Mac 작업에서 실행하지 않았습니다.**
저장소의 이 브랜치 소스가 Windows에 있어야 합니다. 아직 Push하지 않았으므로
지금 `git pull`만으로 새 파일이 전달되지는 않습니다.

### 독립 환경

WSL Ubuntu 터미널에서 저장소 루트로 이동하고 기존 SPEC 환경과 분리합니다.

```bash
python3.10 -m venv .venv-xtts
source .venv-xtts/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-xtts.txt
python -m pip check
nvidia-smi
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)"
```

RTX 5060 Ti 8GB에서 기존 SPEC의 CUDA 연산 성공 기록은 있지만 XTTS VRAM 사용량은
측정하지 않았습니다. 8GB면 무조건 성공한다고 가정하지 않습니다.
드라이버 변경·Deepspeed 설치·기존 가상환경 재설정은 이번 범위에 포함하지 않습니다.

### 모델과 참조 준비

라이선스를 검토·수락한 후 공식 저장소의 모델을 별도 폴더에 준비합니다.
다음은 버전을 고정한 다운로드 예시이며 실제 다운로드 검증은 남아 있습니다.

```bash
hf download coqui/XTTS-v2 --revision v2.0.2 --local-dir checkpoints/xtts-v2
```

최소 필요한 파일은 `config.json`, `model.pth`, `vocab.json`입니다.
공식 snapshot의 LICENSE와 관련 파일을 함께 보관합니다. runner는 모델 파일 해시를
기록하지만 원격 해시와 자동 대조하는 다운로드 검증기는 아직 구현하지 않았습니다.
유지보수 로더는 일부 구 checkpoint를 변환하며 파일을 다시 저장할 수 있으므로
원본 snapshot을 보관하고 별도 작업 사본에서 실행하는 것이 좋습니다.
JSON의 `model_hashes`/`model_hashes_after`로 로딩 중 변화도 확인할 수 있습니다.

사용 허락받은 실제 원본·SPEC 음성을 `runs/xtts_inputs/clean.wav`,
`runs/xtts_inputs/protected.wav`에 준비합니다. 한국어 참조를 우선 확보하고,
교차 언어 합성과 영어 참조 조건은 결과에서 구분합니다. 기존 1.02초 SPEC 음성만으로
한국어 음성 복제 품질이나 보호 효과를 판단하지 않습니다.

### 단일 한국어·영어 합성

실제 권한·약관 확인 후에만 아래 두 확인 옵션을 지정합니다. 기존 출력 파일이 있으면
덮어쓰지 않으므로 새 경로를 사용합니다.

```bash
python scripts/xtts_inference.py \
  --model-dir checkpoints/xtts-v2 --device cuda:0 \
  --reference-wav runs/xtts_inputs/clean.wav \
  --text "안녕하세요. 오늘은 음성 보호 기술을 연구합니다." \
  --language ko --output-path runs/xtts_single/ko.wav \
  --consent-confirmed --cpml-accepted

python scripts/xtts_inference.py \
  --model-dir checkpoints/xtts-v2 --device cuda:0 \
  --reference-wav runs/xtts_inputs/clean.wav \
  --text "Hello. Today we are studying voice protection technology." \
  --language en --output-path runs/xtts_single/en.wav \
  --consent-confirmed --cpml-accepted
```

예상 경로는 `runs/xtts_single/ko.wav`, `runs/xtts_single/en.wav`입니다.
**현재 실제 생성된 파일은 아닙니다.**

### 원본·보호 비교

`experiments/xtts_pair_pending.json`을 검토하고 실제 참조 허락·CPML 확인 후
`consent_confirmed`와 `cpml_accepted`를 `true`로 변경합니다.

```bash
python scripts/run_xtts_pair.py experiments/xtts_pair_pending.json
```

새 `runs/xtts_pair_*` 안에 `0_ko_clean.wav`, `0_ko_protected.wav`,
`1_en_clean.wav`, `1_en_protected.wav`, `report.json`을 저장하는 구조입니다.
결과에는 역할·문장·언어·seed·참조/출력/모델 해시·패키지 버전과 모델 로딩 시/합성 시
메모리 사용량을 기록합니다. 메모리는 PyTorch 할당량·예약량·최대 할당량이며
시스템 전체 GPU 사용량과 다릅니다.

## 6. 기존 평가·강건성 연결 지점

`audio_metrics.load_audio()`의 입력 검사를 그대로 재사용하고,
`run_experiment.require_keys()`와 `sha256()`을 manifest·해시 도구로 재사용합니다.
추론 모듈은 `protect.py`, `evaluate.py`, `train.py`를 import하거나 수정하지 않습니다.

연결 순서는 다음과 같습니다.

1. SPEC 담당자가 원본/보호 WAV와 생성 설정을 제공합니다.
2. 강건성 담당자가 원본·보호 참조에 동일 Gaussian/MP3/리샘플링 조건을 적용합니다.
3. XTTS runner가 각 참조 조건에서 같은 문장을 합성합니다.
4. 평가 담당자가 합성 결과의 화자 SIM·KO/EN WER 등을 비교합니다.

2~4의 전체 통합 실험은 아직 미실행이며 자동 연결 코드를 완성한 것은 아닙니다.
현재 `run_experiment.py`의 strict STOI는 원본 대 보호/변환 파형의 음질 점검에
사용할 수 있습니다. 합성 음성은 길이·내용·정렬이 달라 기존 STOI runner에 무조건
넣지 않습니다. 기존 WER 경로는 영어 Whisper 모델·원본 행 누락 문제도 있어
한국어를 평가하려면 별도 ASR 모델·문자 정규화·정렬/지표 정책이 필요합니다.
SPEC 효과가 XTTS에 전이되는지는 이 실험을 통해 확인할 가설입니다.

## 7. 오류 처리와 테스트 결과

모델을 불러오기 전에 파일 누락·지원하지 않는 언어·빈 문장·잘못된 seed·기존 출력
경로·입력 오디오를 검사합니다. 언어는 소문자 `ko`/`en`만 허용합니다.
권한/CPML 확인이 없으면 시작하지 않습니다. 실제 모델 로딩 import 오류에는
독립 환경 설치 안내를 반환합니다. CUDA 부재는 CPU로 자동 전환하지 않습니다.
모델 로딩/추론의 CUDA OOM은 `CudaMemoryError`로 변환하며, pair runner 실행 중
실패하면 `status: FAIL`과 오류 종류를 JSON으로 남깁니다. 사전 입력 검증 실패는
결과 폴더 생성 전 종료합니다. 부분 WAV가 있을 수 있어 실패 JSON을 확인해야 합니다.

Mac에서 실행한 명령:

```bash
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py' -v
.venv/bin/python scripts/run_xtts_pair.py experiments/xtts_pair_pending.json
```

자동 테스트는 **기존 9개 + XTTS Mock 8개 = 총 17개 통과**했습니다.
기본 manifest 실행은 동의 확인값이 없어 의도대로 거부되었습니다.
테스트는 수학적으로 생성한 mono 신호만 사용하며 실제 사람의 음성을 복제하지 않습니다.
출력은 임시 디렉터리에 생성 후 정리되고 `MOCK_PASS`로 구분합니다.
별도 Mock 실행 기록은 `runs/xtts_pair_6kox8ut3/report.json`에 남겼습니다.
동일 폴더의 `0_ko_clean.wav`, `0_ko_protected.wav`, `1_en_clean.wav`,
`1_en_protected.wav`는 모두 Mock 수학 신호이며 실제 한국어/영어 음성이 아닙니다.
수동 확인 중 Mock 백엔드 주입을 빠뜨린 첫 실행은 모델 누락으로 실패했고,
백엔드를 명시한 재실행에서 `MOCK_PASS`를 확인했습니다. 실제 모델 로딩 성공은 아닙니다.
KO/EN 전달, clean/protected 동일 설정, 입력 불변성, 누락/오류 거부, 덮어쓰기 방지,
CUDA 부재·로딩 OOM 변환·실패 JSON을 검증합니다.
Mock 결과는 실제 모델 API 실행·GPU 메모리·음질·KO/EN 발음 성공을 입증하지 않습니다.

## 8. 실제 결과와 다음 순서

| 요청 결과 | 실제 확인 |
| --- | --- |
| 설치 패키지 | 없음. 기존 `.venv`로 Mock 테스트만 실행 |
| CUDA | 현재 Mac의 `torch.cuda.is_available()` False, CUDA 런타임 없음 |
| XTTS VRAM | 측정하지 않음 |
| 한국어 합성 | 미실행, 실제 출력 없음 |
| 영어 합성 | 미실행, 실제 출력 없음 |
| 실제 보호 효과 | 미검증 |
| Git | 별도 `feature/xtts-v2`에 관련 7개 파일만 커밋·공유; 다른 브랜치 변경 없음 |

다음 순서는 Windows 전용 환경 설치·pip check → snapshot/라이선스·참조 허락 확인 →
CUDA 로딩 및 VRAM 실측 → 한국어·영어 단일 합성 청취/검사 → 실제 clean/SPEC 쌍 비교 →
반복 seed·화자 SIM·다국어 WER → 강건성 조건별 보호 효과 검증입니다.

## 9. 개발 마감 확인 (2026-10-27)

기존 9개와 신규 Mock 8개, 총 17개 테스트를 마감 시 재실행하여 통과했습니다.
실제 모델 로딩·CUDA VRAM 측정·한국어/영어 음성 합성은 여전히 미실행입니다.
내일 Windows WSL에서는 독립 환경 설치와 의존성 확인부터 시작하고,
라이선스·참조 허락 확인 후 모델 로딩·VRAM 실측·단일 KO/EN 합성·쌍 비교 순서로 진행합니다.
모델·참조 음성·Mock 결과·가상환경은 커밋에 포함하지 않습니다.
