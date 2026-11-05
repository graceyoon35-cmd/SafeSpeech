# MeloTTS SPEC 산출물 검증·STOI 연결 (2026-11-05)

## 현재 상태와 Git 최신화

작업 시작 시 Mac의 `feature/xtts-v2` 작업 트리는 깨끗했습니다.
`git fetch origin --prune` 후 `feature/spec-perturbation`으로 전환하고
`git merge --ff-only origin/feature/spec-perturbation`을 실행했습니다.
HEAD는 `85ae25ed23581fc302cb741414c0f7fdc1871729`이며,
`16f46e7`의 CUDA 검증과 `85ae25e`의 실제 SPEC STOI 평가 이력을 포함합니다.
다른 팀원 브랜치를 병합하거나 수정하지 않았습니다. 아래 신규 파일 4개는
이 기준 커밋 위에 별도 SPEC 개발 커밋으로 공유합니다.

**완료 범위는 CPU 산출물 계약 검사·STOI 연결·Mock 자동 테스트입니다.**
실제 MeloTTS 가중치·한국어 보호 WAV·CUDA 환경이 이 Mac에 없으므로
모델 추론·최적화·보호 성능 성공으로 기록하지 않습니다.

## 기존 구현과 인터페이스 차이

분석 대상은 `origin/feature/melo-kr`의
`b6db484367d38909b4ffd74e06fcf1166cd1e0b9`입니다.
`melo_kr/protect_melo.py`, `common.py`, 학습 데이터·합성·평가 코드를 원격 Git 객체로
읽었으며 로컬로 병합하거나 알고리즘을 수정하지 않았습니다.

| 항목 | 기존 SPEC 검증 | 모델 담당자 Melo 출력 |
| --- | --- | --- |
| 생성 보고서 | `status`, `mode`, `iterations`, `source`, `epsilon` | `args`, `model_sampling_rate`, `files`, `total_seconds` |
| epsilon 단위 | 진폭 단위 `8/255` | `args.epsilon=8`, 생성 시 255로 나눔 |
| 출력 구성 | 단일 실행 폴더의 `protected.wav`, `perturbation.pt` | 원본 basename별 PCM16 WAV와 공통 보고서 |
| 샘플레이트 | 기존 검증은 24kHz | 모델 내부 44.1kHz 경로, 저장 WAV는 입력 Hz |
| 원본 보존 | 원본과 복사본의 해시 비교 | 원본 복사본·사전 해시는 생산자 보고서에 없음 |
| noise 재구성 | 저장 tensor로 WAV와 정확히 비교 | noise tensor가 없어 같은 검사를 할 수 없음 |

따라서 `scripts/validate_spec_pair.py`와 `evaluate_spec_stoi.py`에 Melo 보고서를
그대로 넣을 수 없습니다. 기존 코드는 보존하고 새 파일로 연결합니다.
파인튜닝이나 제로샷 합성의 음질·보호 효과 평가는 이번 범위가 아닙니다.

## 새 코드와 검사 계약

| 파일 | 역할 |
| --- | --- |
| `scripts/validate_melo_spec_pair.py` | Melo 생성 보고서와 원본/보호 WAV의 CPU 검증 |
| `scripts/evaluate_melo_spec_stoi.py` | 검증 바인딩 재확인 후 strict STOI, JSON·CSV 저장 |
| `scripts/test_melo_spec_artifacts.py` | 수학 신호 fixture 기반 자동 테스트 14개 |

추가 패키지는 설치하지 않았습니다. 기존 `requirements-spec-stoi.txt`의
NumPy·SciPy·SoundFile·pystoi 환경을 재사용합니다. 두 스크립트 모두 torch나 Melo 모델을
불러오지 않습니다. `scripts/audio_metrics.py`의 오디오 검사와 STOI를 그대로 사용합니다.

검사 내용:

- 원본·보호 WAV의 mono, 1초 이상, 같은 Hz·샘플 수
- NaN/Inf·진폭 범위 [-1, 1]·RMS 무음 검사
- 생산자 출력의 PCM16 subtype, 보고서 sample rate·파일명 연결
- SPEC mode, 양수 iteration·epsilon, 확정 seed, 유한 손실·보고 통계
- 중복 basename 거부: 생산자는 하위 폴더를 평탄화하므로 같은 이름이 있으면 모호함
- 저장 전 보고 delta와 저장 후 실제 delta의 양자화 오차 범위 확인
- 사전 기록한 원본 SHA-256과 현재 파일 비교, 검증 실행 전후 세 입력 파일 해시 비교
- 평가 직전에 검증 JSON의 입력 해시를 재확인하고 원본 보고서로 계약 검사 재수행
- STOI 계산 전후 입력·validation 해시 재확인, 원본 대 원본 STOI sanity check

### PCM16 오차와 클리핑

PCM16 양자화에서는 아래 방향 반올림도 가능하므로 반 LSB 대신 **1 LSB**를 허용합니다.

```text
epsilon = generation.args.epsilon / 255
saved_max_abs_delta <= epsilon + 1/32768 + 1e-7
abs(saved_max_abs_delta - reported_pre_quantization_delta) <= 1/32768 + 1e-7
reported_pre_quantization_delta <= epsilon + 1e-7
```

`1e-7`은 생산자의 float32 변환·연산 허용 오차입니다. epsilon을 임의 확대하는
일반 정책이 아니라 현재 PCM16 저장 경로의 보수적 상한입니다.
출력의 0 변화는 거부합니다. FLOAT WAV를 PCM16처럼 검증하지 않습니다.
진폭 초과는 거부하고 경계 진폭 샘플 수를 JSON에 기록합니다. 실제 원본이 경계값을
포함할 수 있으므로 경계 샘플만으로 클리핑 발생 여부를 단정하거나 무조건 실패시키지는 않습니다.
저장 WAV만으로 이전 단계의 클리핑이나 기울기의 유한성을 증명할 수 없습니다.

### 원본 불변성과 출처 한계

`--expected-original-sha256`는 **생성 전에 따로 기록한 값**을 전달해야 합니다.
검증 시점에 처음 해시를 계산해 전달하면 이후 실행 중 보존만 확인할 수 있고,
생성 이전부터 원본이 그대로였다는 증거는 되지 않습니다.

생산자 보고서에는 원본 해시·model checkpoint 해시·실제 소스 리비전이 없습니다.
새 검증 JSON의 `producer_schema_reference`는 분석한 형식의 기준 리비전이며 실제
실행 코드의 리비전을 검증했다는 의미가 아닙니다. `model_weights_verified`와
`producer_revision_verified`는 False입니다. 명시적 파일 인자와 보고서 basename을
연결하므로, 다른 PC에서 옮긴 경로도 읽을 수 있지만 파일명만으로 생성 출처를 증명하지 않습니다.
생성 로그·체크포인트 해시·소스 리비전을 Windows 실험 시 별도로 보관해야 합니다.

## 실행 방법

아래는 실제 Windows 실험 후 실행할 명령의 예시입니다. 입력 경로·해시는 실제 값으로
바꾸고 저장소 루트에서 실행합니다. 현재 Mac에는 이 예시의 실제 Melo 파일이 없습니다.

먼저 생성 전에 원본 해시를 외부 실험 기록에 보관합니다.

```bash
sha256sum data/korean/clean/sample.wav
```

모델 담당자 코드로 생성한 뒤 CPU 평가 환경에서 실행합니다.
`--record-index`는 공통 생성 보고서의 `files` 배열 인덱스입니다. 원본·보호 파일은
해당 인덱스에 기록된 파일명과 일치해야 합니다.

```bash
python scripts/validate_melo_spec_pair.py \
  --generation-report data/korean/protected/SPEC/report.json \
  --original data/korean/clean/sample.wav \
  --protected data/korean/protected/SPEC/sample.wav \
  --expected-original-sha256 <생성_전에_기록한_SHA256> \
  --execution-id melo_spec_gpu_run_001 \
  --input-kind melo_artifact --record-index 0 \
  --output runs/melo_validation_001.json

python scripts/evaluate_melo_spec_stoi.py \
  --validation runs/melo_validation_001.json
```

`runs/`가 없으면 먼저 `mkdir -p runs`로 준비합니다. validation 출력 파일은
새 경로여야 하며 기존 파일·입력을 덮어쓰지 않습니다. STOI 평가는 매 실행
새 `runs/melo_stoi_*`에 `report.json`, `comparison.csv`를 기록합니다.
쉘에서 `<...>` 표시는 직접 입력할 값의 자리표시자이므로 실제 해시 문자열로 바꿉니다.

JSON·CSV에는 모델 종류, 입력 종류, 실행 ID, seed, iteration, 모델/파일 샘플레이트,
epsilon, 원본·보호·생성 보고서 해시, STOI가 포함됩니다.
JSON은 validation 해시·패키지 버전·환경·계약 한계도 기록합니다.
검증 실패는 종료 코드 1이며 성공 validation을 만들지 않습니다.
평가 중 입력·저장 오류는 `FAIL` JSON과 종료 코드 1로 보고합니다.
디스크 자체가 JSON 저장까지 거부하면 보고서를 쓸 수 없고 예외로 종료될 수 있습니다.

입력 종류는 반드시 선택합니다.

| 설정 | 의미 |
| --- | --- |
| `mock_contract` | 결과 `MOCK_PASS`, 수학 신호/무작위 초기화의 입력 계약 확인만 |
| `melo_artifact` | 실제 산출물의 형식·해시·범위 확인. 가중치 실행 진위나 보호 효과 증명은 아님 |

`melo_artifact`에서 생산자 보고서가 `random_init=true`이면 실패합니다.
하지만 False라는 선언만으로 실제 사전학습 가중치를 사용했음을 입증하지는 않습니다.

## 테스트와 이번 실행 결과

```bash
.venv/bin/python -B -m unittest discover -s scripts -p 'test_*.py' -v
```

Python 3.10.11 Mac CPU에서 **기존 SPEC 평가 7개 + 새 계약 테스트 14개 = 21개 통과**했습니다.
브랜치가 달라 이전 XTTS 브랜치의 17개 테스트를 이번에 실행했다고 계산하지 않습니다.
첫 실행에서 신규 테스트 파일의 여러 줄 `with` 구문 오류가 발견되어 수정했으며,
수정 후 21개 모두 통과했습니다.

테스트는 길이·Hz·채널·짧은 음성·무음·NaN/Inf·진폭 초과·손상/누락 WAV·출력 subtype,
epsilon 초과·비정상 손실·원본 해시 불일치·실행 중 변경·검증 후 변경·중복 basename,
Mock 구분·입력 불변성·JSON/CSV·저장 오류·덮어쓰기 방지를 확인했습니다.
실제 음성을 테스트용 데이터로 생성하거나 실제 모델을 실행하지 않았습니다.

별도 CLI smoke는 새 `runs/melo_contract_mock_*` 안의 수학 신호와 가상 생성 보고서로만
검증기→STOI→JSON/CSV의 전체 입출력을 실행했습니다. 결과는 `MOCK_PASS`입니다.
이번 기록은 `runs/melo_contract_mock_dc8ya0f0/validation.json`과
같은 폴더의 `evaluation/melo_stoi_lwcv7dkq/report.json`, `comparison.csv`입니다.
최종 코드 재실행 결과는 같은 Mock 폴더의
`final_evaluation/melo_stoi_n_qrucui/report.json`과 `comparison.csv`입니다.
이 최종 CLI 실행 전후에 이미 존재하던 데이터·실행 파일 212개의 해시 일치를 확인했습니다.
Mock STOI 숫자는 Melo 음질·보호 효과 결과로 보고하지 않습니다.
기존 WAV·모델·실험 폴더는 변경하거나 삭제하지 않았습니다.

## 기존 BERT-VITS2 결과 보존

`docs/SPEC_STOI_RESULTS_KO.md`의 0.9612467971012578 (1회),
0.7754191151068554 (10회)는 **2026-10-30 BERT-VITS2 기반 실제 SPEC**의
영어 단일 발화 결과입니다. 해당 문서·검증기·평가기·원본 연구 코드는 수정하지 않았습니다.
새 Melo 결과는 `model_kind=MeloTTS-Korean`, 별도 JSON·CSV·실행 ID로 구분합니다.

## Windows CUDA 실험 준비와 다음 과제

1. SPEC checkout과 모델 담당자 Melo 코드를 별도 작업 경로에 보관합니다. 현재 SPEC
   브랜치에는 `melo_kr/`를 병합하지 않았으므로 여기에 생성 명령을 바로 실행할 수 없습니다.
2. 모델 담당자 환경에서 `third_party/MeloTTS` 소스 리비전을 고정·기록합니다.
   MeloTTS-Korean의 `config.json`, `checkpoint.pth`와 모델 파일 해시를 준비합니다.
3. 별도 Melo 생성/학습 환경을 사용합니다. Python 3.10, RTX 5060 Ti의 기존 검증 조합인
   torch/torchaudio 2.7.1+cu128을 후보로 삼되, Melo 실제 실행 호환성은 확인해야 합니다.
   모델 담당 README의 환경 구분과 Melo의 고정 Transformers/librosa 버전을 따르며
   기존 SPEC GPU 환경에 덮어 설치하지 않습니다.
4. CPU 검증·평가는 `requirements-spec-stoi.txt`의 별도 환경에서 실행합니다.
   이 새 검증기에는 torch·Melo·XTTS 가중치가 필요하지 않습니다.
5. 허락받은 한국어 mono WAV, 생성 전 SHA-256, 실제 사용한 seed·speaker·epsilon·소스
   리비전을 기록하고 새 출력 폴더에서 한 발화·1회/10회로 시작합니다.
6. GPU 손실·기울기·VRAM·실행 로그를 확인하고, **저장 후** 새 검증기와 STOI로 점검합니다.
   충분한 길이·다른 Hz·PCM16 경계 오차에 대한 실제 모델 검증이 필요합니다.
7. 이후 여러 발화·화자, 음질 청취, 같은 train/validation 분할의 clean/protected
   파인튜닝 공격, SIM·CER 평가로 확장합니다. STOI만으로 방어 효과를 결론 내리지 않습니다.

향후 모델 담당자와 합의할 항목은 저장 FLOAT/PCM16 정책, noise tensor 보관 여부,
generation report에 넣을 해시·실행 상태·가중치/소스 리비전, SafeSpeech에서 STOI 항
누락 시 중단 정책, 파일명 충돌 처리입니다. 이번에는 해당 알고리즘을 변경하지 않았습니다.
