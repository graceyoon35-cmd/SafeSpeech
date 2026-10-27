# 실제 SPEC 음성 연결 준비 (2026-10-27)

## 현재 확인 결과

Mac 저장소 전체에서 `.git`, 가상환경, Python 캐시를 제외하고 SPEC 산출물 이름을
검색했습니다. `protected.wav`, `perturbation.pt`, `.noise` 파일은 없습니다.
`runs/*/report.json` 8개도 확인했으며 모두 이전 CPU STOI smoke 또는 원본을 대상으로 한
Gaussian 대조군입니다. WSL 실행 폴더는 현재 Mac에 없습니다.
따라서 **실제 SPEC 음성의 검증과 STOI 측정은 미실행**입니다.

기존 `scripts/smoke_perturbation.py`와 WSL 검증 문서에서 확인한 10회 실행 결과는
`runs/smoke_20261025_024217_002294`입니다. 이 경로는 이전 기록에 따른 것이며,
현재 WSL 파일의 존재 여부는 Windows PC에서 확인해야 합니다.
1회 결과 `smoke_20261025_024005_466049`와 섞지 않습니다.

## 가져올 파일

WSL 저장소 `/mnt/c/Users/윤채영/SafeSpeech` 기준으로 같은 10회 실행 폴더에서 가져옵니다.

| 파일 | 용도 |
| --- | --- |
| `5339_14134_000092_000003.wav` | 생성에 실제 사용한 원본 복사본 |
| `protected.wav` | 실제 SPEC 출력. 필수 |
| `report.json` | 생성 출처 확인: `status`, `mode`, `iterations`, `source`, `epsilon` 등 |
| `perturbation.pt` | 선택: 후속 섭동 분석용. 이번 CPU 평가에는 필요 없음 |

모델 체크포인트·BERT 특징 파일·CUDA 환경은 이번 음성 평가에 필요하지 않습니다.
단순 Gaussian `candidate.wav`나 `transformed.wav`를 `protected.wav`로 바꿔 넣으면 안 됩니다.

## Windows/WSL → Mac 가져오기

아래 명령은 안내용이며 이번 작업에서 실행하지 않았습니다.
먼저 WSL 터미널에서 이전 실행 파일과 보고서를 확인합니다.

```bash
cd /mnt/c/Users/윤채영/SafeSpeech
ls -l runs/smoke_20261025_024217_002294
cat runs/smoke_20261025_024217_002294/report.json
```

보고서가 `status: PASS`, `mode: SPEC`, `iterations: 10`, 예상한 원본 `source`인지
확인합니다. 기록상 epsilon은 `8/255`, 출력은 24kHz·24,481 samples입니다.
보고서만으로 파일의 생성 출처가 암호학적으로 증명되지는 않으므로 본인의 실제 실행
폴더에서 가져와야 합니다.

동일 WSL 터미널에서 세 파일을 묶고 전송 확인용 해시를 출력합니다.

```bash
tar -czf /tmp/safespeech-spec10.tar.gz \
  runs/smoke_20261025_024217_002294/5339_14134_000092_000003.wav \
  runs/smoke_20261025_024217_002294/protected.wav \
  runs/smoke_20261025_024217_002294/report.json
sha256sum /tmp/safespeech-spec10.tar.gz
explorer.exe /tmp
```

Explorer에서 아카이브를 USB·개인 파일 전송 수단으로 Mac의 `Downloads`에 복사합니다.
원격 접속 정보가 제공되지 않아 자동 전송은 하지 않았습니다. `runs/`는 Git 제외이므로
Git pull만으로 WSL 결과가 전달되지 않습니다.

Mac 터미널에서 전송 해시를 비교하고 목록을 확인한 뒤 저장소에 풉니다.
아래 `tar -xzkf`의 `-k`는 같은 이름의 기존 파일 덮어쓰기를 거부합니다.

```bash
shasum -a 256 ~/Downloads/safespeech-spec10.tar.gz
tar -tzf ~/Downloads/safespeech-spec10.tar.gz
cd /Users/yunchaeyeong/SafeSpeech
tar -xzkf ~/Downloads/safespeech-spec10.tar.gz
```

출력 해시가 WSL에서 기록한 값과 같은지, 아카이브 목록이 위 세 상대 경로인지
확인한 후 압축을 풉니다. 원본 데이터 파일은 덮어쓰지 않습니다.

## 파일 확보 후 기존 runner로 실행

추가한 `experiments/spec_gaussian_pending.json`은 실제 파일을 기다리는 manifest입니다.
`reference_role: original`, `target_role: protected`, seed 42, std 0.005를 지정하고,
원본과 보호 파일을 위 WSL 실행 폴더에서 읽습니다. 합성 음성은 `null`입니다.

먼저 가져온 `report.json`의 출처를 위 기준으로 확인합니다. 그다음:

```bash
.venv/bin/python scripts/run_experiment.py experiments/spec_gaussian_pending.json
```

기존 runner가 중복 코드 없이 다음을 수행합니다.

1. 실제 파일 존재 및 mono·1초 이상·유한값·비무음·진폭 범위 검사
2. 원본/보호 음성의 sample rate·샘플 수 일치 확인 (자동 보정 없음)
3. `metrics.before`: 원본 대 SPEC 보호 음성 STOI
4. `metrics.after`: 원본 대 Gaussian을 적용한 SPEC 보호 음성 STOI
5. 두 점수·차이·역할별 실행 전후 SHA-256·변환 WAV 해시를 새 JSON에 기록

이 runner는 WSL 생성 보고서의 진위나 SPEC 생성 자체를 검증하지 않습니다.
파일명·manifest 역할 선택만으로 실제 SPEC 출처를 확인할 수 없습니다.
필요하면 저장소 원본과 가져온 원본의 파일 해시도 비교합니다.

```bash
shasum -a 256 \
  data/LibriTTS/5339/5339_14134_000092_000003.wav \
  runs/smoke_20261025_024217_002294/5339_14134_000092_000003.wav
```

## 검증 결과와 구분

| 실험 | 현재 결과 |
| --- | --- |
| 원본 → Gaussian 대조군 | 이전 확인: 변환 전 STOI 1.000000, 후 0.970284 |
| 실제 SPEC → Gaussian | 파일 미확보, 점수 없음 |
| 기존 자동 테스트 | 이번에 9개 모두 통과 |
| 파일 없는 SPEC manifest | 종료 코드 1, missing file 오류, 성공 결과 미생성 확인 |

기존 테스트의 보호 역할 테스트는 테스트용 복사본으로 인터페이스만 검사합니다.
실제 SPEC 검증 결과로 해석하지 않습니다. 새 manifest는 파일이 없으면 실패하며
다른 음성으로 대체하지 않습니다. 실제 파일 확보 전에는 SPEC 결과 수치를 작성하지 않습니다.

`protect.py`, `evaluate.py`, `train.py`와 runner는 이번에 변경하지 않았습니다.
CUDA 실행·SPEC 재생성·커밋·Push도 하지 않았습니다.
