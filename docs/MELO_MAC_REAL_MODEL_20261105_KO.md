# Mac 실제 MeloTTS 로딩·한국어 합성·SPEC 기울기 검증

실행일: 2026-11-05 · 브랜치: `feature/spec-perturbation`
기준 HEAD: `7c2ec7eaf0691dfe851ce5e7b85e80073d2a4dec`

## 결론과 범위

공식 한국어 체크포인트를 사용하여 **CPU strict 로딩, 한국어 1문장 합성,
입력 파형 gradient 계산, SPEC 1회 생성, 저장 산출물 검증·STOI 평가**를 완료했습니다.
랜덤 가중치나 Mock으로 산출한 성공 결과가 아닙니다.

다만 전체 `melo.api`는 Mac MeCab 패키지 충돌로 import에 실패했습니다.
한국어 합성은 공식 한국어 전처리와 공식 모델의 `infer()`를 연결한 실행용
한국어 전용 import harness에서 성공했습니다. 또한 배포 config의 누락된 손실 설정을
공식 학습 config에서 보충했습니다. 따라서 원본 설치 명령만으로 수정 없이 전체
워크플로가 동작했다고 주장하지 않습니다.

한국어 합성과 SPEC 입력은 별도입니다. SPEC gradient 입력은 기존 영어 LibriTTS의
약 1.02초 발화이며, 한국어 모델의 음향 경로를 검증하는 용도입니다.
한국어 녹음에 대한 보호 효과·복제 공격 방어 성능·파인튜닝은 검증하지 않았습니다.

## 환경과 격리

| 항목 | 실제 확인 |
| --- | --- |
| 운영체제 | macOS 26.6.2, Apple Silicon arm64 |
| Python | 새 전용 환경의 3.10.11 |
| PyTorch / TorchAudio | 2.7.1 / 2.7.1, Mac 패키지 |
| 연산 장치 | 모델 로딩·합성·gradient·SPEC 모두 CPU, 스레드 4개 |
| CUDA | 없음 |
| 디스크 | 시작 시 약 590GiB 여유, 이번 실행 폴더 약 2.5GiB |
| 격리 위치 | `runs/melo_mac_actual_20261105_z1muic00/venv` |

기존 프로젝트 `.venv`는 변경하지 않았으며 실행 전후 `pip freeze`가 일치했습니다.
새 환경은 별도 Python으로 생성했고 새 `runs/` 폴더 아래에만 모델·소스·캐시·결과를
모았습니다. 원본 연구 WAV와 기존 실행 폴더는 덮어쓰지 않았습니다.

**MPS 확인을 정정합니다.** 샌드박스 안에서는 `is_available=False`였지만,
호스트 권한으로 확인하면 `is_built=True`, `is_available=True`였습니다.
작은 MPS 텐서와 유한 gradient 연산도 통과했습니다. Melo 모델의 MPS 실행은
이번에 시도하지 않았으며 CPU 결과를 MPS 성공으로 해석하면 안 됩니다.
근거는 `host_device_result.json`과 각 CPU stage 결과의 장치 기록입니다.

## 사용 소스와 모델

| 대상 | 고정 리비전 |
| --- | --- |
| 공식 `myshell-ai/MeloTTS` 소스 | `209145371cff8fc3bd60d7be902ea69cbdb7965a` |
| 모델 담당 `feature/melo-kr` | `b6db484367d38909b4ffd74e06fcf1166cd1e0b9` |
| `myshell-ai/MeloTTS-Korean` | `0207e5adfc90129a51b6b03d89be6d84360ed323` |
| 한국어 BERT `kykim/bert-kor-base` | `1779cc0982ada0216dd6de0dd4e86fb78201926d` |

공식 출처:

- [MeloTTS 소스](https://github.com/myshell-ai/MeloTTS/tree/209145371cff8fc3bd60d7be902ea69cbdb7965a)
- [한국어 모델](https://huggingface.co/myshell-ai/MeloTTS-Korean/tree/0207e5adfc90129a51b6b03d89be6d84360ed323)
- [한국어 BERT](https://huggingface.co/kykim/bert-kor-base/tree/1779cc0982ada0216dd6de0dd4e86fb78201926d)

| Melo 파일 | 크기 | SHA-256 |
| --- | ---: | --- |
| `config.json` | 3,405 bytes | `74543376976dfadde45ba34336fa79c7e95509f43a7c2e701b22c0f71fd7695c` |
| `checkpoint.pth` | 207,860,748 bytes | `48e3ff3fd0b5348e095f0468e60ae727507564100f58142ef3a922ead6e0a4d0` |

체크포인트는 원격 LFS SHA-256과 일치했고 로딩·gradient 실행 후에도 해시가 같습니다.
한국어 BERT 가중치의 원격 LFS 해시도 확인했습니다. 상세 파일 기록은
`model_source.json`, `ko_bert_source.json`에 있습니다.
모델 화자는 `KR: 0`, 모델 샘플레이트는 44,100Hz, 파라미터 수는 51,874,097개입니다.
`strict=True` state dict 검사에서 누락·추가 키가 모두 0개였습니다.

## 의존성과 발생 오류

핵심 패키지는 NumPy 1.26.4, SciPy 1.13.1, SoundFile 0.12.1,
librosa 0.10.2.post1, Transformers 4.27.4, Hugging Face Hub 0.36.0,
pystoi 0.4.1입니다. 전체 목록은 `melo_environment_freeze.txt`에 보관했습니다.
새 환경의 `pip check`도 통과했습니다.

공식 Melo 소스를 `pip install -e`로 설치하지 않고 고정 checkout을 Python 경로에
지정했습니다. 이 환경은 합성·gradient 검증용 최소 조합이며 공식 requirements 전체를
그대로 검증한 설치는 아닙니다. 특히 담당자 손실 함수가 가져오는 저장소 mel 함수와
맞추기 위해 공식 요구 librosa 0.9.1 대신 0.10.2.post1을 사용했습니다.

발생 오류와 처리:

1. 모델 다운로드 중 연결이 끊겼습니다. 재시도하여 전체 파일과 원격 해시 일치를 확인했습니다.
2. `melo.api`는 전체 언어 모듈을 eager import하며 일본어 MeCab에서 실패했습니다.
   Mac의 대소문자 경로 충돌 때문에 `mecab-python3`와 `python-mecab-ko`가 겹쳤습니다.
   새 환경에서 일본어 패키지를 제거하고 한국어 패키지를 복구했습니다.
3. 합성에서는 공식 `melo.text.korean`을 직접 사용하고 공식
   `get_text_for_tts_infer()` 함수의 AST를 변경 없이 실행했습니다. 언어 선택을 한국어로
   제한한 연결 함수와 로컬 한국어 BERT 경로 매핑만 사용했습니다. 가짜 특징·음소·가중치는
   넣지 않았습니다. 따라서 한국어 합성 성공과 전체 다국어 API 실패를 별도로 기록합니다.
4. 한국어 BERT에서 문장 관계 분류 head 두 키가 사용되지 않는 경고가 있었습니다.
   `BertForMaskedLM` 특징 추출에는 해당 head를 사용하지 않으며 나머지 실제 가중치를 로딩했습니다.

## 배포 config와 SPEC 손실 호환성

담당자의 `common.py`·`protect_melo.py`를 원격 커밋에서 실행 폴더로 복사하여
사용했고, 원격 Git 콘텐츠와 일치함을 실행 후 확인했습니다. 원본을 수정하지 않았습니다.

배포 config 그대로 `linear_spec()`을 호출하면 다음 오류가 발생합니다.

```text
AttributeError: 'types.SimpleNamespace' object has no attribute 'win_length'
```

공식 학습 config에서 다음 누락값만 **메모리상의 실행용 설정**에 보충했습니다.

| 필드 | 보충값 |
| --- | --- |
| `data.win_length` | 2048 |
| `data.n_mel_channels` | 128 |
| `data.mel_fmin` | 0.0 |
| `data.mel_fmax` | null |
| `train.c_mel` | 45 |

원본 `model/config.json`이나 모델 가중치를 변경하지 않았습니다.
`effective_loss_config_verified.json`에 보충 설정과 공식 학습 config 해시를 기록했습니다.
seed 52, beta 10, epsilon 8/255, 1회 반복은 호출 인자로 명시했습니다.
담당자와 이 호환성 설정의 정식 인터페이스를 합의해야 합니다.

## 한국어 합성 결과

문장: **안녕하세요. 음성 보호를 연구합니다.**

| 항목 | 결과 |
| --- | --- |
| 상태 | 실제 가중치 CPU 합성 PASS |
| 출력 | `korean_synthesis.wav` |
| 샘플레이트·채널 | 44,100Hz·모노 |
| 샘플 수·길이 | 196,096 samples / 약 4.4466초 |
| NaN/Inf | 없음 |
| 최대 절대 진폭 | 약 0.714928 |
| RMS | 약 0.145722 |
| 실행 시간 | 약 5.22초 (새 프로세스 내 로딩·전처리 포함) |
| 최대 프로세스 RSS | 약 2.20GiB, GPU VRAM 수치가 아님 |

합성 결과 JSON은 `korean_synthesis_result.json`입니다.
파일·신호 검사는 완료했지만 청취 품질·발음 정확도·한국어 CER는 측정하지 않았습니다.
이 WAV는 모델이 생성한 합성 음성이며, 사람의 원본 녹음이나 보호 WAV가 아닙니다.

## 입력 gradient 및 SPEC 1회 결과

원본은 기존 `data/LibriTTS/5339/5339_14134_000092_000003.wav`입니다.
원본 자체는 읽기만 했으며, 새 FLOAT WAV 복사본을 입력으로 만들었습니다.
기존 원본 해시 `155e50f0fb0596c794acc88394533c2f66a57d3915072521e84d771f8a22212e`가
이전 Windows 기록과 일치했습니다. 새 검증 실행에서는 복사본의 생성 전·후 해시도 일치했습니다.

경로는 담당자 구현 그대로 입력 파형 → 미분 가능한 24k→44.1k 리샘플링 →
스펙트로그램 → `enc_q` → `dec` → SPEC 손실입니다.
`torch.autograd.grad()`를 감싸 실제 반환값을 관찰했으며 기울기나 갱신값을 대체하지 않았습니다.

| 항목 | 결과 |
| --- | --- |
| gradient 상태 | None 아님, 전체 유한, 0 아님 |
| gradient shape | `(1, 1, 24481)` |
| 0이 아닌 원소 | 24,481개 |
| gradient 최대 절댓값 | 약 2062.2166 |
| gradient L2 norm | 약 22967.0430 |
| 손실 | 약 3969.9771 |
| SPEC 반복 | 1회, 실제 Melo 가중치, random_init=False |
| 저장 보호 WAV | `spec1_verified/short_input.wav`, 24kHz·PCM16 |
| 저장 후 최대 변화량 | 0.003143310546875 |
| 검증기 | PASS, PCM16 오차 포함 epsilon 검사 |
| STOI | **0.9610571370019663** |
| 원본 대 원본 STOI | 0.9999999999999997 |
| gradient·SPEC 실행 시간 | 약 2.52초, 로딩 포함 |
| 최대 프로세스 RSS | 약 1.37GiB |

선택한 음향 경로에서는 gradient 단절을 관찰하지 못했습니다.
공식 전체 모델의 alignment 경로에는 `.data.cpu().numpy()` 연산이 있지만
이번 담당자 경로는 해당 연산을 사용하지 않습니다.

이는 영어 단일 발화의 **명료도·미분 경로 확인**입니다. Melo 한국어 보호 효과나
파인튜닝 공격 방어의 성공 결과가 아닙니다. 과거 BERT-VITS2의 1회/10회 STOI와도
실험 조건이 달라 같은 표본군·동등 성능으로 비교하지 않습니다.

## 실행 기록과 재실행 위치

아래 모든 상대 경로의 기준 실행 폴더는
`runs/melo_mac_actual_20261105_z1muic00/`입니다. 약 2.5GiB이며 Git에서 제외됩니다.

| 파일 | 증거 |
| --- | --- |
| `audit_summary.json` | 소스·가중치·단계별 결과·명령·보존 확인 |
| `load_result.json` | 실제 가중치 strict 로딩 |
| `api_import.json` | 전체 Melo API import 실패와 traceback |
| `host_device_result.json` | 호스트의 CPU/MPS 점검 |
| `gradient_verified_result.json` | 사전 해시·실제 gradient·1회 생성 결과 |
| `effective_loss_config_verified.json` | 공식 학습 기본값을 보충한 실행 설정 |
| `spec1_verified/report.json` | 담당자 형식의 실제 생성 보고서 |
| `spec1_validation.json` | 검증기 PASS |
| `spec1_evaluation/melo_stoi_9helwebs/report.json` | 실제 STOI 결과 |
| 같은 평가 폴더의 `comparison.csv` | CSV 결과 |
| `korean_synthesis_result.json` | 실제 한국어 합성 검사 |
| `original_venv_freeze.txt`, `melo_environment_freeze.txt`, `pip_check.txt` | 환경 보존·의존성 |

실제로 실행한 핵심 명령 형태는 다음과 같습니다. 설치·캐시 준비 후 사용한 명령이며,
이미 있는 출력에 재실행하면 배타 생성 때문에 실패할 수 있습니다. 재실험은 새 실행
폴더·출력 경로를 사용합니다.

```bash
runs/melo_mac_actual_20261105_z1muic00/venv/bin/python -B \
  runs/melo_mac_actual_20261105_z1muic00/run_model_checks.py --stage load
runs/melo_mac_actual_20261105_z1muic00/venv/bin/python -B \
  runs/melo_mac_actual_20261105_z1muic00/run_model_checks_verified.py --stage gradient
```

합성은 실행 폴더의 `run_korean_synthesis.py`를 CPU에서 실행했으며
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, 실행 폴더의 `HF_HOME`·`NLTK_DATA`를
지정했습니다. 한국어 BERT는 사전 다운로드한 로컬 고정 리비전만 사용했습니다.
검증·평가는 기존 프로젝트 CPU 환경에서 새 스크립트를 실제 호출했습니다.
정확한 파일 인자는 JSON의 경로·해시에 기록되어 있습니다.

## 보존 상태와 Windows에 넘길 작업

모델 checkpoint·config 해시, 공식 소스 checkout의 깨끗한 상태,
담당자 코드와 원격 콘텐츠 일치, 기존 `.venv` freeze 일치를 확인했습니다.
프로젝트 핵심 소스나 다른 브랜치를 변경·병합하지 않았고 Commit·Push도 하지 않았습니다.
추가 추적 후보는 이 문서뿐이며 모든 실행물은 새 `runs/` 폴더에 있습니다.

Windows 후속 검증:

1. 공식 소스·가중치 리비전과 누락 설정 보충 정책을 모델 담당자와 확정합니다.
2. 별도 CUDA 환경에서 동일 입력·1회 결과·유한 gradient·VRAM을 검증합니다.
3. 허락받은 실제 한국어 녹음에서 1회/10회·긴 발화·여러 화자로 확장합니다.
4. clean/protected 분할과 대본을 동일하게 맞춰 Melo 파인튜닝 공격을 실행합니다.
5. 한국어 합성 청취·CER·화자 SIM·강건성 조건을 평가합니다.
6. 전체 다국어 API 설치 문제와 Mac MPS 모델 연산은 별도 호환성 과제로 남깁니다.

이번 결과는 위 Windows 단계의 대체 결과가 아니라, 실제 가중치 경로를 CPU에서
먼저 확인한 개발 검증입니다.
