> **보관 문서**: 작성 당시 개발 상태·수치를 담은 기록입니다. 현재 실행은 [앱팀 실행 가이드](../README_SPEC_HANDOFF_20261120_KO.md), 연구 결론은 [최종 보고서](../SPEC_FINAL_REPORT_20261120_KO.md)를 확인하세요.

> 11월 19일 시점의 기록입니다. ε 보정본 추가 학습·평가를 포함한 최신 결과는 [2026-11-20 최종 보고서](../SPEC_FINAL_REPORT_20261120_KO.md)와 [앱팀 명세](../README_SPEC_HANDOFF_20261120_KO.md)를 확인하세요.

# SPEC 보호 기능 앱팀 인계 — 2026-11-19

인계 범위는 기존 SPEC 알고리즘을 호출하는 오프라인 WAV 실행기와 검증 결과다. 앱 UI·서버·프론트엔드 개발 및 변경은 수행하지 않았다.

## 실제 실행 명령

검증 환경: Windows + Ubuntu-22.04 WSL2, RTX 5060 Ti, Python 3.10 계열의 기존 전용 가상환경. 아래 명령은 **WSL bash**에서 실행한다. 설치된 정확한 버전은 `runs/final_validation_20261119/environment_freeze.txt`, 기존 설치 명세는 `requirements-melo-training-wsl.txt`, `requirements-spec-stoi.txt`를 참고한다.

```bash
cd '/mnt/c/Users/윤채영/SafeSpeech-desktop-integration-20261118'
PY=/home/safespeech/.venvs/melo-training-20261118/bin/python
ASSETS=/home/safespeech/models/melo-spec-pinned-20261118
"$PY" scripts/protect_wav.py \
  --input '/absolute/path/input.wav' \
  --output-dir '/absolute/path/new-output-directory' \
  --assets "$ASSETS" --device cuda \
  --iterations 10 --epsilon 8 --seed 52
```

직접 검증한 입력은 `runs/desktop_20261118/data/data_set/kss_part_200/train/wavs/kss_hf_00078.wav`이다. 이 파일과 모델은 로컬 연구 자산이며 Git에 포함하지 않는다. 같은 명령의 출력 경로는 매번 새 이름을 지정한다. 실행 예제 스크립트 `scripts/example_spec_protect.sh`는 입력과 새 출력 디렉터리를 인자로 받는다.

## 입력·출력 계약

| 항목 | 명세 |
|---|---|
| 입력 | 실제 파일 경로, mono WAV, PCM16, 44,100Hz, 1~10초, 유한 값, 완전 무음 제외 |
| 전처리 | 호출 측에서 명시적으로 수행. 자동 리샘플·다운믹스·분할 없음 |
| 텍스트 | 이 보호 명령에는 대본 불필요. 한국어 Melo 고정 대리모델의 SPEC 목적함수 사용 |
| ε | CLI 단위 `/255`. `--epsilon 8` = 정규화 진폭 8/255 |
| 반복 | `--iterations` 1~200; 1과 10은 기존 조건 SPEC_1·SPEC_10에 대응 |
| seed | 0~2³²−1. 모델·환경·입력 및 설정도 함께 보관해야 재현 가능 |
| 결정론 | 기본 GPU 모드의 바이트 단위 재현은 미보장. `--deterministic`은 현재 GPU backend에서 실패하는 검증용 옵션 |
| 출력 경로 | **존재하지 않는 디렉터리**. 기존 경로는 거부하여 덮어쓰기 방지 |
| 앱이 사용할 파일 | `protected.wav`: 입력과 같은 샘플레이트·길이의 mono PCM16 WAV |
| 진단 파일 | `result.json`, `generation.json`, `validation.json`, `stoi.json`, `gradients.json`, `backend.json`, 고정 backend 소스 사본 |
| 중간 파일 | `optimized.wav`: 정수 샘플 ε 보정 전 진단용. 앱 전달 파일은 `protected.wav` |
| 성공·실패 | 종료 코드 0 + 마지막 stdout 줄의 JSON `status=PASS`. 실패는 종료 코드 1; 일부 생성 후 실패하면 `result.json`에 FAIL·원인 기록 |

호출 측은 종료 코드와 PASS를 함께 확인한다. GPU 작업이므로 비동기 프로세스로 호출하고 충분한 실행 시간을 허용한다. 폴더 일부나 WAV 존재만으로 성공 처리하지 않는다. 원본 파일은 읽기만 하며 모델 가중치는 변경하지 않는다.

## ε 저장 오차 수정과 실제 스모크 테스트

기존 SPEC_10 200개는 부동소수점 섭동을 PCM16으로 저장할 때 최대 `0.031402587890625`까지 증가했다. 기존 validator의 `ε + 1/32768` 허용 기준에는 맞지만 엄격한 `8/255=0.03137254901960784`에는 초과한다. 기존 연구 파일은 보존했다.

새 실행기는 정수 샘플에서 원본 대비 `floor(ε×32768)` 이내로 투영한다. 검증된 스모크 테스트는 `runs/final_validation_20261119/wav_smoke_strict/`에 있다. 해당 실행의 최종 파일 최대 차이는 `0.0313720703125`, STOI는 `0.8923985786`이며 141,123샘플·44.1kHz PCM16을 유지했다. 같은 실행에서 투영 전의 STOI는 `0.8927472340`이다. `result.json`의 `pcm16_adjusted_samples`, `strict_linf`, `output_sha256`으로 확인한다. 최종 실행기에는 투영 전 `generation_unprojected.json`과 투영 후 `generation.json`을 구분하며, `result.json`의 SNR은 최종 WAV로 계산한다.

기본 GPU 모드로 동일 seed를 독립 재실행했을 때 WAV SHA256이 달랐다. 두 검사 출력의 STOI는 0.8923985786 / 0.8886657853이었고 76,205샘플이 달랐다. 입력·모델·seed가 같아도 바이트 단위 동일 파일을 보장하지 않는다. 결정론을 강제한 진단 실행은 `reflection_pad1d_backward_out_cuda`의 결정론 구현 부재로 종료 코드 1에 실패했다. 오류를 경고로 바꾸거나 비결정론 연산을 몰래 허용하지 않았다. `handoff_repeat_mismatch.json`, `deterministic_repeat.json`, `deterministic_smoke_a.log`에 실패 증거가 있다. 같은 GPU에서 확인한 비결정론 연산이지만 이 연산 하나가 모든 차이의 유일한 원인이라고 단정하지 않는다.

이 저장 단계 보정은 기존 KSS 학습 결과를 소급해서 바꾸지 않는다. 기존·seed53 학습은 원래 보호 WAV를 사용하며, 엄격한 ε 보정 WAV로 학습한 모델의 SIM·CER는 별도 실험이 필요하다.

## 의존성과 실행 범위

전용 환경에는 PyTorch·torchaudio CUDA, NumPy, SoundFile/libsndfile, SciPy, pystoi, Melo backend 및 해당 코드가 사용하는 패키지가 필요하다. 검증·평가에는 jiwer, openai-whisper, speechbrain, matplotlib가 추가로 필요하다. 시스템 FFmpeg는 ASR 및 오디오 평가 도구에서 사용한다. CUDA GPU 경로에서 실제 성공을 확인했으며 CPU의 속도·완료 여부는 이번 인계에서 검증하지 않았다.

이번 실행에서 확인한 버전: torch `2.7.1+cu128`, torchaudio `2.7.1+cu128`, numpy `1.23.5`, scipy `1.13.1`, soundfile `0.12.1`, pystoi `0.4.1`, transformers `4.46.3`, jiwer `4.0.0`, openai-whisper `20240930`, speechbrain `1.1.2`, matplotlib `3.7.5`. `pip check`는 통과했다. 이 목록만으로 고정 backend·모델 자산을 대체할 수 없으며 전체 환경은 freeze 파일로 대조한다.

`--assets`는 고정 `model/checkpoint.pth`, `model/config.json`, `effective_loss_config_verified.json`, 고정 MeloTTS 및 팀 backend 소스를 갖춘 기존 자산 폴더다. 실행기가 해시와 고정 Git revision을 검사한다. 가중치 SHA256은 `48e3ff3fd0b5348e095f0468e60ae727507564100f58142ef3a922ead6e0a4d0`이다. 자산 이동 시 상대 구조를 보존해야 하며, 자산 준비는 기존 `prepare_melo_spec_assets.py`와 기존 문서를 따른다. 앱 실행기가 모델을 다운로드하지 않는다.

KSS는 이번 비상업적 내부 연구 범위로 사용했다. 사람 청취 검수 완료, 별도 화자 동의, 음성·가중치 외부 공개 승인으로 바꾸어 기록하지 않는다. KSS 데이터 범위는 기존 `KSS_RESEARCH_PROTOCOL_20261118_KO.md`를 따른다. 보호 WAV의 STOI·SNR과 복제 모델의 SIM·CER는 서로 다른 평가다. SPEC_10은 원음 보존 품질도 낮추므로 앱에서 무손실·불가청·복제 차단 보장을 표시해서는 안 된다.

## 검증 명령

```bash
"$PY" -m unittest discover -s scripts -p 'test_*.py' -v
"$PY" scripts/audit_kss_final.py \
  --study runs/kss_research_study_b0msiw2t/study.json \
  --output runs/new-kss-audit-directory
```

전체 자동 테스트와 실제 GPU 스모크 테스트는 별도 증거다. 최종 결과·추가 seed 결과는 `SPEC_FINAL_REPORT_20261119_KO.md`, 실행 증거는 `runs/final_validation_20261119/`에 기록한다.

추가 seed와 발표 자료를 다시 실행할 때에는 새 저장 경로를 사용한다.

```bash
"$PY" scripts/run_kss_seed_repeat.py \
  --training-summary runs/paired_melo_training_rhp4a8t2/summary.json \
  --assets "$ASSETS" --seed 53 --steps 1000 \
  --output-root /home/safespeech/experiments/NEW-KSS-repeat
# 출력된 새 summary.json을 사용해 평가한다.
"$PY" scripts/evaluate_paired_melo_smoke.py \
  --training-summary /absolute/path/NEW-summary.json --assets "$ASSETS"
"$PY" scripts/project_kss_strict_inputs.py \
  --spec-run runs/korean_melo_spec_mqxs1q1s --output runs/NEW-strict-copies
"$PY" scripts/recompute_kss_sim.py \
  --evaluation runs/melo_smoke_evaluation_qyh7qbgy/evaluation.json \
  --audit runs/final_validation_20261119/audit_strict/audit.json \
  --output runs/NEW-sim-recomputation
"$PY" scripts/build_spec_final_report.py \
  --validation-root runs/final_validation_20261119 \
  --output-directory runs/NEW-spec-presentation
```
