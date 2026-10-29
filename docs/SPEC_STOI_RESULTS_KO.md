# 실제 SPEC 보호 음성 STOI 평가 결과

실행일: 2026-10-30 KST. 브랜치 `feature/spec-perturbation`, 작업 시작 HEAD
`16f46e7a525260284aa6d432dddfb663c90b69bf`.
이 문서는 실제 CUDA 생성 SPEC WAV를 CPU에서 평가한 결과입니다.
Gaussian 대조군이나 Mock 음성의 점수를 SPEC 결과로 사용하지 않았습니다.

## 결과

| 입력 | SPEC 반복 | STOI | 원본 대비 최대 절대 변화량 |
| --- | ---: | ---: | ---: |
| 원본 대 원본 sanity check | 해당 없음 | 0.9999999999999997 | 0 |
| 실제 원본 대 SPEC 보호 WAV | 1 | 0.9612467971012578 | 0.003137260675430298 |
| 실제 원본 대 SPEC 보호 WAV | 10 | 0.7754191151068554 | 0.0313725508749485 |

10회 minus 1회 STOI는 **-0.18582768199440247**입니다.
이 파일 쌍에서는 10회 출력의 STOI가 더 낮았습니다. 단일 약 1.02초 영어 발화에
대한 결과이며 반복 횟수의 일반적인 효과나 통계적 유의성을 입증하지 않습니다.
STOI는 음성 명료도 지표이며 음성 복제 방어 효과 지표가 아닙니다.
SIM, WER, MCD, 청취 품질, XTTS 전이 보호 효과를 이번에 평가하지 않았습니다.

## 입력과 출처 검증

모든 경로는 저장소 루트 기준입니다.

| 반복 | 입력 폴더 |
| --- | --- |
| 1회 | `runs/smoke_20261030_005658_023201/` |
| 10회 | `runs/smoke_20261030_005719_318250/` |

원본 복사본은 각 폴더의 `5339_14134_000092_000003.wav`, 보호 음성은
`protected.wav`입니다. 원본 출처는 저장소의 LibriTTS 화자 5339이며,
라이선스·변경 기록은 기존 `DATA_ATTRIBUTION.md`에 있습니다.

두 쌍 모두 24kHz, mono, 24,481샘플, 길이 1.0200416666666667초입니다.
원본은 PCM_16, 보호 WAV는 FLOAT입니다. 1회·10회 모두 NaN/Inf 없음,
진폭 [-1, 1], 비무음, epsilon `8/255 + 1e-7` 제한을 통과했습니다.
noise shape `(1, 1, 24481)`와 저장 WAV 재구성도 원본 검증 스크립트로 재확인했습니다.
자동 자르기·리샘플링·시간 정렬을 하지 않았습니다.

| 파일 | SHA-256 |
| --- | --- |
| 공통 원본 및 두 원본 복사본 | `155e50f0fb0596c794acc88394533c2f66a57d3915072521e84d771f8a22212e` |
| 1회 protected.wav | `e343021e98affbaf9c760643629d85bde7a31e274afde10c73a3429b2ffd50c1` |
| 10회 protected.wav | `4277dd17365e6baab6880528d08b69443ef3b39b8bd15e945eca0fd94d5f8995` |

새 validation JSON의 source·원본·보호·noise·생성 report 해시는 이전 감사
`runs/audit_resume_20261030_005532_823644/validation_1.json` 및
`validation_10.json`의 해시와 모두 일치했습니다.

기존 manifest는 각각 같은 감사 폴더의 `spec1_metrics_manifest.json`,
`spec10_metrics_manifest.json`입니다. 역할은 `original -> protected`,
정렬은 `strict`, `synthetic=null`로 올바릅니다.
manifest의 Gaussian 설정은 **실행하지 않았고**, 새 결과에는
`transform_applied=null`, `candidate_kind=actual_SPEC_protected`와
`declared_transform_not_executed`를 명시했습니다. 기존 manifest는 수정하지 않았습니다.
과거 문서의 Gaussian 점수 0.970284020573383은 이번 SPEC 점수가 아닙니다.

## 평가 환경과 재사용 코드

WSL Ubuntu 22.04.5의 Python 3.10.12로 별도 CPU 가상환경을 생성했습니다.

| 항목 | 값 |
| --- | --- |
| CPU 환경 | `/home/safespeech/.venvs/safespeech-stoi-20261030_011000` |
| NumPy | 1.26.4 |
| SciPy | 1.13.1 |
| SoundFile | 0.12.1 |
| pystoi | 0.4.1 |
| 추가 전이 의존성 | cffi 2.1.1, pycparser 3.11 |
| 점수 | `pystoi.stoi(..., extended=False)` |
| CPU 환경 pip check | 통과 |

`origin/feature/metrics`의 `27ae06d1fe304320c671eeeb15338e8c23b36ac6`을
읽고 `scripts/audio_metrics.py`를 동일 내용으로 재사용했습니다.
해당 브랜치의 CPU requirements는 `requirements-spec-stoi.txt`로 추가했으며
두 파일 모두 원본 Git 콘텐츠 해시가 일치합니다. 다른 브랜치는 수정·병합하지 않았습니다.

기존 metrics runner는 보호 음성에 Gaussian을 추가하는 평가까지 실행하므로,
이번에는 `scripts/evaluate_spec_stoi.py`를 별도 추가해 변환 없는 쌍만 평가했습니다.
검증 JSON의 해시로 실제 WAV·noise·생성 보고서를 확인한 뒤 계산하며,
길이·샘플레이트·역할 불일치, 서로 다른 원본 비교, 파일 변경은 실패 처리합니다.
새 출력 폴더와 JSON·CSV를 생성하고 입력 WAV는 다시 저장하지 않습니다.

기존 GPU 환경 `/home/safespeech/.venvs/safespeech`의 패키지는 변경하지 않았습니다.
GPU 환경 NumPy 1.23.5 및 torch 2.7.1+cu128을 유지했고,
전체 `pip freeze`가 실행 전후 일치합니다. GPU 환경에서는 기존 파일 검증과
패키지 조회만 실행했습니다. CPU 환경에는 PyTorch를 설치하지 않았습니다.

## 실제 실행 명령과 증거

새 감사 폴더는 `runs/spec_stoi_20261030_011000_995566/`입니다.
이 안의 `run_stoi_audit.py`를 기존 WSL Python으로 실행하여 아래 작업을
순서대로 수행했습니다. 실제 argv·종료 코드는 `outcome.json`에 기록되어 있습니다.
새 CPU 환경 설치는 이 환경에만 수행했습니다.

```bash
/usr/bin/python3 -m venv /home/safespeech/.venvs/safespeech-stoi-20261030_011000
/home/safespeech/.venvs/safespeech-stoi-20261030_011000/bin/python -m pip install -r requirements-spec-stoi.txt
/home/safespeech/.venvs/safespeech-stoi-20261030_011000/bin/python -m pip check
```

원본 `validate_spec_pair.py`는 기존 GPU Python으로 두 입력 폴더에 각각
`--expected-source-sha256`와 새 `--output` 경로를 지정해 실행했고 모두 종료 코드 0입니다.
정확한 인자는 `outcome.json`, 출력은 `validate_1.log`, `validate_10.log`입니다.

최종 코드로 실행한 테스트 명령은 다음과 같습니다.

```bash
/home/safespeech/.venvs/safespeech-stoi-20261030_011000/bin/python -B -m unittest discover -s scripts -p test_evaluate_spec_stoi.py -v
```

최종 평가 명령의 저장소 루트 기준 재현 형태는 다음과 같습니다.
실제 실행은 같은 파일의 절대 경로를 인자로 사용했으며 정확한 argv는
`final_outcome.json`에 있습니다. 아래 1회 실행 명령의 output-root를
`final_repeat_2`로 바꾸어 별도 프로세스에서 재실행했습니다.

```bash
/home/safespeech/.venvs/safespeech-stoi-20261030_011000/bin/python -B scripts/evaluate_spec_stoi.py \
  --manifests \
    runs/audit_resume_20261030_005532_823644/spec1_metrics_manifest.json \
    runs/audit_resume_20261030_005532_823644/spec10_metrics_manifest.json \
  --validations \
    runs/spec_stoi_20261030_011000_995566/validation_1.json \
    runs/spec_stoi_20261030_011000_995566/validation_10.json \
  --output-root runs/spec_stoi_20261030_011000_995566/final_repeat_1
```

재실행 시에도 runner가 고유한 하위 폴더를 만들므로 기존 결과는 덮어쓰지 않습니다.
로컬 감사 wrapper는 고정 이름 로그를 배타 생성하므로 같은 wrapper를 다시
실행하려면 새 감사 폴더·환경 이름으로 준비해야 합니다.

## 결과 파일과 검증

아래 경로는 새 감사 폴더 기준입니다.

- `comparison_final.json`: 최종 1회·10회 점수, 차이, 반복 일치, 소스 해시
- `final_repeat_1/spec_stoi_kks0r0ny/report.json`: 최종 첫 평가
- `final_repeat_2/spec_stoi_2saasmms/report.json`: 최종 재평가
- 각 report 폴더의 `comparison.csv`: 표 형식 결과
- `validation_1.json`, `validation_10.json`: 재확인한 SPEC 파일 무결성
- `final_contract_tests.log`: 최종 7개 테스트 통과 기록
- `final_evaluate_1.log`, `final_evaluate_2.log`: 실제 평가 프로세스 출력
- `final_outcome.json`: 최종 실행 argv·종료 코드·파일 보존 검사
- `outcome.json`, `comparison.json`, `evaluate_1.log`, `evaluate_2.log`: 초기 두 번 실행 기록
- `gpu_freeze_before.txt`, `gpu_freeze_after.txt`, `cpu_freeze.log`: 환경 확인 자료

초기 5개 테스트와 두 번 실제 평가가 통과한 뒤, 저장 실패 처리와 같은 원본 비교
조건을 보강했습니다. 최종 7개 테스트와 두 번 실제 평가를 다시 실행하여 모두 통과했습니다.
최종 두 프로세스의 쌍·점수·해시가 정확히 일치하고 초기 두 실행 결과와도 일치합니다.
점수만 배열 안에서 반복 계산한 것이 아니라 독립 프로세스에서 파일을 다시 읽었습니다.

7개 테스트는 샘플레이트 불일치, 길이 불일치, 잘못된 역할, synthetic 입력,
검증 후 파일 변경, 서로 다른 원본 비교, 저장 실패 시 FAIL·종료 코드 1을 확인합니다.
테스트의 계약용 fixture는 실제 SPEC 점수 산출에 사용하지 않았습니다.

초기 감사에서 256개 파일의 실행 전후 해시 변경이 없었습니다.
그 후 의도적으로 보강한 새 runner와 테스트 파일 2개를 제외한 **254개 보존 파일**을
최종 재평가 후 다시 확인하여 변경 0개였습니다. 대상에는 원본 WAV, 기존 실행 결과,
기존 manifest, 모델 8개, 원본 protect.py와 설정 등이 포함됩니다.
최종 새 코드 해시는 `comparison_final.json`에 별도로 기록했습니다.
공유 G_0.pth를 포함한 모델은 읽기·해시 검사만 했으며 수정하지 않았습니다.

## Git 상태와 남은 작업

새 추적 후보는 `requirements-spec-stoi.txt`, `scripts/audio_metrics.py`,
`scripts/evaluate_spec_stoi.py`, `scripts/test_evaluate_spec_stoi.py`, 이 문서입니다.
기존 추적 파일은 변경하지 않았고 stage·commit·Push도 하지 않았습니다.
`runs/` 및 가상환경은 Git에 추가하지 않습니다. 가상환경은 저장소 밖 WSL 홈에 있습니다.

남은 범위는 여러 발화·화자·길이·반복 횟수에서의 평가, 청취 품질 확인,
SIM/MCD/WER와 XTTS 실제 합성 평가, 학습된 화자 임베딩 구성 확인입니다.
이번 실행 오류는 없었으며 Gaussian 대조 실험은 추가로 실행하지 않았습니다.
