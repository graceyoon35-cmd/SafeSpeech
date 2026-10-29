# SPEC 출력을 STOI 평가에 연결하는 방법

작성일: 2026-10-30 KST. 이 문서는 평가 준비이며 STOI 계산 성공 기록이 아닙니다.

## 확인한 평가 코드

로컬 원격 추적 참조 `origin/feature/metrics`의
`27ae06d1fe304320c671eeeb15338e8c23b36ac6`을 `git show`로 읽었습니다.
Fetch, 브랜치 병합, 다른 브랜치 파일 복사, 평가 실행은 하지 않았습니다.

| 필요한 파일 | 용도 |
| --- | --- |
| `scripts/audio_metrics.py` | `load_audio(path)`와 `score_pair(reference, candidate, reference_rate, candidate_rate)` |
| `scripts/run_experiment.py` | manifest 읽기, 변환 전후 STOI, 입력 해시 검사, 새 출력 폴더 생성 |
| `scripts/robustness_transforms.py` | seed로 고정한 Gaussian 잡음 추가 |
| `requirements-evaluation-smoke.txt` | CPU 평가 의존성 |

평가 의존성은 NumPy 1.26.4, SciPy 1.13.1, SoundFile 0.12.1,
pystoi 0.4.1입니다. SPEC 가상환경의 NumPy는 1.23.5이므로 평가 담당자의
별도 CPU 환경에서 고정 목록을 설치하는 방식을 권장합니다. 기존 GPU 환경을
변경하지 않았고 새 평가 환경도 아직 만들지 않았습니다.

## 입력 계약

mono, 1초 이상, 비무음(RMS >= 1e-6), 유한값, 진폭 [-1, 1], 같은
샘플레이트와 샘플 수가 필요합니다. 자동 리샘플링, 자르기, 시간 정렬은 없습니다.
`validate_spec_pair.py` 통과는 파일 계약과 섭동 범위를 확인하는 단계이며,
pystoi의 경고를 오류로 처리하는 실제 STOI 계산 성공까지 보장하지는 않습니다.

생성기의 `protect.compute_stoi()`는 SafeSpeech 모드의 미분용 NegSTOILoss입니다.
최종 파일 평가의 `pystoi.stoi(..., extended=False)` 점수와 구분합니다.

## 실제 입력 설정

이번 재검증 감사 폴더는 `runs/audit_resume_20261030_005532_823644/`입니다.
스모크 및 파일 검증 성공 후 `spec1_metrics_manifest.json`,
`spec10_metrics_manifest.json`이 생성됩니다. 각 manifest는 같은 실행의
원본 복사 WAV와 `protected.wav`를 가리킵니다. 경로는 manifest 파일의
디렉터리 기준이며 다른 PC로 전달하면 WAV와 함께 경로를 맞춰야 합니다.

스키마 v1의 정확한 키는 아래와 같습니다. 이 예시의 경로 문자열은 설명용이며,
실행에는 감사 폴더의 실제 manifest를 사용합니다.

```json
{
  "schema_version": 1,
  "experiment_id": "spec10_stoi_gaussian_ready",
  "audio": {
    "original": "../smoke_RUN/5339_14134_000092_000003.wav",
    "protected": "../smoke_RUN/protected.wav",
    "synthetic": null
  },
  "evaluation": {
    "reference_role": "original",
    "target_role": "protected",
    "alignment": "strict"
  },
  "transform": {"name": "gaussian_noise", "seed": 42, "std": 0.005}
}
```

`report.json`이나 검증 경로를 임의 키로 추가하면 엄격한 스키마 검사에 실패합니다.
생성 report, validation JSON, 출처 문서는 manifest와 별도로 전달합니다.

## 다음 작업에서 실행할 명령 (현재 미실행)

평가 코드가 이미 있는 metrics 작업 폴더와 평가 전용 Python을 사용합니다.
현재 SPEC 폴더에는 평가 스크립트가 없으므로 아래 상대 명령의 작업 디렉터리는
metrics 코드가 있는 저장소 루트여야 합니다.

```bash
python scripts/run_experiment.py \
  /mnt/c/Users/<USER>/SafeSpeech-spec-perturbation/runs/audit_resume_20261030_005532_823644/spec10_metrics_manifest.json \
  --output-root /mnt/c/Users/<USER>/SafeSpeech-spec-perturbation/runs/metrics_stoi
```

runner는 `experiment_*` 새 폴더에 결과를 저장합니다.
`metrics.before.stoi`가 원본 대비 SPEC 보호 음성의 점수이고,
`metrics.after.stoi`는 보호 음성에 Gaussian(std 0.005)을 추가한 점수입니다.
후자는 SPEC 보호 음성 단독 점수가 아닙니다. 1회 결과도 같은 방식으로
`spec1_metrics_manifest.json`을 전달할 수 있습니다.

변환 없이 파일 쌍만 평가하려면 metrics 환경에서 다음 인터페이스를 사용할 수 있습니다.
이는 호출 예시이며 현재 실행하지 않았습니다.

```python
from audio_metrics import load_audio, score_pair  # scripts 디렉터리에서 실행
reference, sr_ref = load_audio(original_path)
candidate, sr_candidate = load_audio(protected_path)
result = score_pair(reference, candidate, sr_ref, sr_candidate)
```

공유 자료는 원본·보호 WAV, 생성 report, validation JSON, DATA_ATTRIBUTION.md,
manifest입니다. WAV는 Git에 추가하지 않고 로컬 평가 또는 별도 승인된 전달
방식을 사용합니다. CPU STOI에 모델 가중치·BERT 특징·noise 텐서는 필요하지 않습니다.

STOI는 신호 명료도 지표입니다. 이 점수만으로 음성 복제 억제 효과를 입증하지 않습니다.
XTTS-v2 합성 음성은 길이·정렬이 달라질 수 있어 별도의 평가 계약이 필요합니다.
