# 2학년 B: 강건성 변환 개발

기존 feature/cpu-experiment-pipeline의 Gaussian/STOI 코드는 그대로 유지합니다.
추가 코드는 scripts/robustness_b.py, run_robustness_b.py, test_robustness_b.py입니다.

## 범위

- Gaussian: 기존 함수를 재사용합니다. 원본·보호 입력에 같은 seed/std를 적용합니다.
- MP3: ffmpeg의 libmp3lame CBR로 인코딩하고 pcm_f64le WAV로 디코딩합니다.
  원본 샘플레이트·길이가 달라지면 자동 잘라내기 없이 실패합니다.
  중간 MP3도 보존하고 ffmpeg 버전, 비트레이트, 해시를 기록합니다.
- 리샘플링: scipy.signal.resample_poly, 기본 Kaiser 필터를 사용합니다.
  출력은 실제 목표 Hz를 유지합니다. STOI용 비교쌍 정렬은 평가 단계에서 명시해야 합니다.
- 필터/코덱 overshoot는 [-1, 1]로 클리핑하며 변경 샘플 수를 기록합니다.
- 원본/보호 파일을 덮어쓰지 않으며 매 실행 새 runs/robustness_b_* 폴더를 만듭니다.
- JSON에 입력·출력 SHA-256, 전후 Hz/길이, 조건, 패키지 버전을 남깁니다.

## 실행

CPU 전용 환경에 requirements-evaluation-smoke.txt를 설치합니다.
GPU/섭동 생성 환경과 분리해 NumPy 버전 충돌을 피합니다.

    python -m unittest discover -s scripts -p "test_*.py" -v
    python scripts/run_experiment.py experiments/gaussian_smoke.json
    python scripts/run_robustness_b.py experiments/robustness_b_pair.json

manifest 스키마 예시(오디오 경로는 manifest 파일 기준):

```json
{
  "schema_version": 1,
  "experiment_id": "spec_pair_robustness",
  "audio": {"original": "/absolute/original.wav", "protected": "/absolute/protected.wav"},
  "conditions": [
    {"name": "gaussian_noise", "seed": 42, "std": 0.005},
    {"name": "mp3", "bitrate_kbps": 64},
    {"name": "resample", "target_hz": 16000}
  ]
}
```

원본 original과 보호 음성 protected는 서로 다른 파일이며 Hz와 길이가 같은 모노 음성이어야 합니다.
역할 이름은 기존 파이프라인과 일치시킵니다.

## 解釈と残る作業

PASS는 변환·저장·재읽기·입력 보존에 성공했다는 뜻이며 보호 성능 검증이 아닙니다.
추가 테스트의 보호 입력은 0.99배한 테스트용 음성이며 실제 SPEC 생성 결과가 아닙니다.
실제 SPEC 쌍의 실행은 Colab 노트북의 마지막 셀에서 별도로 확인합니다.
SIM/MCD/WER, 모델 합성, 여러 발화의 집계·통계는 후속 작업입니다.
MP3 바이트 재현성은 동일한 ffmpeg 빌드와 설정에서 확인합니다.
작업 브랜치는 robustness-B입니다. 커밋·Push는 자동 실행하지 않습니다.
