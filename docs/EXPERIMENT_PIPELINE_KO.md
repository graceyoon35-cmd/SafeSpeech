# CPU 공통 실험 파이프라인 (2026-10-27)

음성 입력 → Gaussian 잡음 변환 → 변환 전후 STOI → JSON 저장을 실행하는
공통 실험 기반입니다. 평가·강건성 담당자의 전체 기능 구현이나 보호 성능 검증을
완료한 것은 아닙니다. 모델 다운로드·CUDA·PyTorch 없이 실행합니다.

## 기존 코드 재사용

기존 `scripts/smoke_evaluation.py`와 테스트를 먼저 검토했습니다.
기존 mono/길이/진폭/유한값/무음 검사와 strict 음성 쌍 STOI 계산을
`scripts/audio_metrics.py`로 옮겼습니다. 기존 Gaussian 잡음 대조군은
`scripts/robustness_transforms.py`의 `gaussian_noise()`로 옮겼습니다.
기존 smoke와 신규 runner가 같은 모듈을 사용하므로 잡음 구현을 중복하지 않습니다.
기존 smoke의 CLI와 기본 seed·표준편차·결과 형식은 유지했습니다.
`protect.py`, `evaluate.py`, `train.py`는 변경하지 않았습니다.

## Manifest v1

실행 가능한 예제는 `experiments/gaussian_smoke.json`입니다.

```json
{
  "schema_version": 1,
  "experiment_id": "libritts_original_gaussian_smoke",
  "audio": {
    "original": "../data/LibriTTS/5339/5339_14134_000092_000003.wav",
    "protected": null,
    "synthetic": null
  },
  "evaluation": {
    "reference_role": "original",
    "target_role": "original",
    "alignment": "strict"
  },
  "transform": {"name": "gaussian_noise", "seed": 42, "std": 0.005}
}
```

| 필드 | 계약 |
| --- | --- |
| `schema_version` | 정수 1 |
| `experiment_id` | 비어 있지 않은 실험 이름; 출력 경로로 사용하지 않음 |
| `audio.original` | 원본 음성 경로, 필수 |
| `audio.protected` | 보호 음성 경로, 없으면 `null` |
| `audio.synthetic` | 합성 음성 경로, 없으면 `null` |
| `evaluation.reference_role` | STOI 기준 음성 역할 |
| `evaluation.target_role` | 변환할 음성 역할; 기준 음성은 변환하지 않음 |
| `evaluation.alignment` | `strict`만 지원: 같은 sample rate·샘플 수 필수 |
| `transform` | `gaussian_noise`, 0 이상의 정수 seed, 유한한 0 이상의 std |

경로는 **manifest 파일이 있는 디렉터리 기준**으로 해석합니다. 절대 경로도 가능합니다.
각 객체의 키를 엄격히 검사해 오타를 거부합니다. 경로를 지정한 모든 음성은 존재해야 하고,
mono·1초 이상·유한값·진폭 [-1, 1]·비무음 조건을 만족해야 합니다.
없는 담당자 결과를 원본으로 대체해서 보호/합성 역할에 넣지 않습니다.
자동화 테스트에서만 역할 선택을 검증하기 위한 복사본을 사용합니다.

이번 예제는 `original`을 기준과 대상으로 선택한 정상 동작 대조 실험입니다.
보호 WAV가 준비되면 `audio.protected`에 경로를 넣고 `target_role`을 `protected`로
바꾸면 됩니다. 실제 보호 WAV를 이번에 생성하거나 평가하지는 않았습니다.
합성 WAV는 길이가 다를 수 있어 별도 정렬 정책을 합의하기 전에는 strict 평가에
실패할 수 있습니다. 자동 자르기·리샘플링·정렬은 지원하지 않습니다.

## 변환 및 결과 의미

Gaussian 변환은 로컬 `numpy.random.default_rng(seed)`로 평균 0, 지정 표준편차의
잡음을 더하고 [-1, 1]로 clip한 새 float64 배열을 반환합니다. 전역 RNG와 입력 배열을
변경하지 않습니다. `std`는 진폭 단위이며 SNR 지정 방식은 아닙니다.

- `metrics.before`: reference와 변환 전 target의 STOI 및 최대 절대 차이
- `metrics.after`: 같은 reference와 변환 후 target의 STOI 및 최대 절대 차이
- `metrics.stoi_delta`: after STOI − before STOI
- `inputs`: 역할별 절대 경로와 실행 전후 SHA-256
- `manifest`: 실행한 설정 전체
- `versions`, `python`, `platform`, `device`: 재현 환경
- `artifact`: 변환 WAV 경로와 SHA-256

매 실행마다 새로운 `runs/experiment_*` 디렉터리에 `report.json`과
`transformed.wav`를 저장합니다. WAV는 float64 형식으로 계산한 샘플을 보존합니다.
SciPy WAV writer를 사용해 SoundFile의 시간 정보가 있는 PEAK 청크를 피하므로
같은 환경·입력·seed의 파일 바이트도 재현할 수 있습니다.
실행 전후 입력 해시가 다르면 실패합니다. 읽기·설정·계산 실패는 종료 코드 1로
보고하며 성공 보고서를 생성하지 않습니다. 출력 저장 중 I/O 실패 시 부분 폴더가
남을 수 있습니다. `runs/`는 기존 Git 제외 규칙을 따릅니다.

## 실행

저장소 루트에서 Python 3.10 환경으로 실행합니다. 추가 의존성은 기존 CPU STOI
smoke 목록을 그대로 사용합니다.

```bash
.venv/bin/python -m pip install -r requirements-evaluation-smoke.txt
.venv/bin/python scripts/run_experiment.py experiments/gaussian_smoke.json
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py' -v
```

출력 루트 변경:

```bash
.venv/bin/python scripts/run_experiment.py experiments/gaussian_smoke.json --output-root runs/team_smoke
```

기존 smoke도 계속 사용할 수 있습니다.

```bash
.venv/bin/python scripts/smoke_evaluation.py
```

## 직접 확인한 결과

Mac arm64, Python 3.10.11, CPU에서 실행했습니다.
NumPy 1.26.4 / SciPy 1.13.1 / SoundFile 0.12.1 / pystoi 0.4.1을 사용했습니다.

| 항목 | 결과 |
| --- | --- |
| 실제 WAV | `5339_14134_000092_000003.wav`, 24kHz, 24,481 samples |
| 변환 설정 | Gaussian, seed 42, std 0.005 |
| 변환 전 STOI | 0.9999999999999997 |
| 변환 후 STOI | 0.970284020573383 |
| STOI 변화량 | -0.029715979426616634 |
| 입력 파일 해시 | 실행 전후 동일 |
| 자동 테스트 | 기존 3개 + 신규 6개 = 9개 통과 |
| 기존 smoke 회귀 실행 | PASS, 기존 점수 유지 |
| `pip check` | 통과 |

최종 실행 보고서: `runs/experiment_2c6x6lr2/report.json`,
재실행: `runs/experiment_m7jhzr7b/report.json`.
두 실제 실행의 WAV SHA-256·지표 일치와 저장 WAV 재평가 점수의 JSON 일치를 확인했습니다.
자동 테스트는 입력 파일 바이트 불변성, 입력 배열 불변성, 동일 seed의 지표·WAV 해시
일치, 다른 seed의 해시 차이, 상대 경로·보호 역할 선택, 잘못된 설정·파일·정렬 거부,
JSON 저장·재읽기를 확인합니다. 보호 역할 테스트의 음성은 테스트용 복사본입니다.

초기 테스트에서는 macOS `/var` → `/private/var` 경로 정규화 차이로 assertion이
실패했고, 기대 경로에도 `resolve()`를 적용한 후 9개 모두 통과했습니다.
추가 실험에서 SoundFile WAV 헤더의 생성 시각으로 파일 해시가 달라지는 것을 발견해
SciPy 저장으로 수정했습니다. 최종 재실행과 9개 테스트는 수정 후 통과했습니다.

## 남은 범위

실제 SPEC 보호 음성 연결, 합성 음성 정렬, SIM/MCD/WER, 다른 강건성 변환,
여러 발화 배치 실험·통계 집계, 모델 학습·추론은 구현하거나 실행하지 않았습니다.
단일 발화 Gaussian 결과로 보호 성능이나 강건성 결론을 내릴 수 없습니다.
`status: PASS`는 이 공통 파이프라인의 정상 실행만 의미합니다.
커밋·Push는 수행하지 않았습니다.
