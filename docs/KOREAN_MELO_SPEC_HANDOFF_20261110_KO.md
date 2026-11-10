# 한국어 Melo SPEC 실행·전달 절차 (2026-11-10)

현재 브랜치: `feature/spec-perturbation`. Commit·Push 미수행.
실제 사람의 한국어 녹음은 확인되지 않았으므로 **실제 한국어 녹음 보호 실험과 실제 파인튜닝 데이터 전달은 미완료**다.
다른 팀원 PC의 미Push 데이터·작업은 조사 범위 밖이다.

## 데이터 조사

- fetch 후 데이터 브랜치 HEAD: `724650ab77f0147dd7b2463264e86437c0dea082`.
- 데이터 CSV 170행: 로컬 LibriTTS clean 168개와 기존 영어 SPEC protected 기록 2개.
- 로컬 `data/`에는 LibriTTS만 존재. `runs/`의 한국어 음성은 11/5 Melo 합성 결과다.
- 한국어 실제 녹음에 대해 출처·연구 사용 허가·검수 대본·화자·고정 split이 함께 준비된 묶음은 발견하지 못했다.
- 합성·수학 신호를 실제 녹음으로 변경 표시하지 않았다. 자동 검증은 동의서 진위나 대본의 실제 발음 일치를 증명하지 않는다.

## 추가 파일

- `scripts/run_korean_melo_spec.py`: 입력 manifest 검증, 고정 코드·모델 검증, 실제 SPEC 반복 실행, 기존 WAV 검증·STOI 함수 연결, 전달 manifest, 실제 녹음의 split별 입력 내보내기.
- `scripts/test_korean_melo_spec.py`: 모델 없는 계약·거부·입력 보존·내보내기·실패 기록 테스트.
- `docs/templates/korean_spec_input.template.json`: 개인정보 없는 입력 양식. 허가·대본 검수 기본값 False이므로 그대로는 실행되지 않는다.

기존 생성 알고리즘·팀원 브랜치·기존 검증기·평가기는 변경하지 않았다.
모델 담당자의 `common.py`, `protect_melo.py`를 Git 객체에서 고정 리비전 그대로 새 실행 폴더에 추출한다.
`perturb()` 반환 gradient를 관찰한 후 그대로 돌려주며, 손실이나 갱신 규칙을 대체하지 않는다.

## 모델·환경·설정

| 항목 | 값 |
|---|---|
| 팀원 코드 | `b6db484367d38909b4ffd74e06fcf1166cd1e0b9` |
| 공식 Melo 소스 | `209145371cff8fc3bd60d7be902ea69cbdb7965a` |
| MeloTTS-Korean 모델 | `0207e5adfc90129a51b6b03d89be6d84360ed323` |
| checkpoint SHA-256 | `48e3ff3fd0b5348e095f0468e60ae727507564100f58142ef3a922ead6e0a4d0` |
| 배포 config SHA-256 | `74543376976dfadde45ba34336fa79c7e95509f43a7c2e701b22c0f71fd7695c` |
| 실제 실행 환경 | 기존 전용 Python 3.10.11, torch/torchaudio 2.7.1, Mac CPU, 4 threads |
| 모델 화자·Hz | `KR:0`, 44,100Hz |
| SPEC 기본 조건 | seed 52, epsilon 8/255, step epsilon/10, beta 10, 1회·10회 |

배포 설정에 없는 `win_length=2048`, `n_mel_channels=128`, `mel_fmin=0`,
`mel_fmax=null`, `c_mel=45`만 기존 검증된 effective config로 보충한다.
나머지 설정이 배포 config와 달라지면 거부한다. 모델 state dict는 strict 로딩한다.
공식 소스 HEAD·clean 상태, 가중치·설정 해시를 실행 전후 검사한다.
MPS·CUDA 실행 및 전체 다국어 `melo.api` 호환성은 이번 검증 범위가 아니다.
BERT·텍스트 경로는 담당자 SPEC 음향 손실에 사용되지 않는다.

## 실제 녹음 입력 준비

실제 WAV·개인정보·허가 원문은 Git 제외인 새 `runs/` 하위 비공개 폴더에 둔다.
템플릿을 그 폴더에 복사하고 실제 데이터로 채운다. 원본 WAV·metadata는 수정하지 않는다.

필수 필드:

- `utterance_id`: 유일한 안전 ID. `speaker_id`: 실명 대신 영문·숫자·밑줄·하이픈의 가명.
- `input_kind`: `human_recording`, `model_synthesized`, `mock` 중 실제 출처 그대로.
- `language=ko`, `model_speaker=KR`. 실제 사람의 speaker ID와 모델 화자 0은 별개다.
- `text`, `transcript_reference`, `transcript_verified`: 실제 녹음은 사람이 음성과 대본을 대조한 뒤 True로 둔다.
- `source`, `research_use_permitted`, `permission_reference`: 별도 확인한 출처와 사용 허가 기록.
- 실제 녹음은 `permission_scope=spec_and_finetuning`: 원본·보호 데이터 학습까지 허가 범위에 포함해야 한다.
- `split`: `train`, `validation`, `test` 중 미리 고정. 필요한 split을 모두 포함했는지는 팀이 결정한다.
- `original_path`: manifest 파일 기준 상대 경로 또는 절대 경로.
- `original_sha256`: 생성 이전에 기록한 해시. `sample_rate`: 실제 파일 Hz.

검사 범위는 mono·1~10초·비무음·유한값·진폭 [-1,1], 지원 Hz
16000/22050/24000/44100/48000과 선언 일치, 중복 ID·경로·동일 해시의 중복 split 거부다.
대본은 한국어 포함·metadata 구분자 금지·검수 선언을 확인한다.
파일만으로 사람 녹음 여부·허가의 진위·대본의 발음 일치를 판별하지 않는다.
합성 점검 입력의 대본 검수는 기존 합성 입력 문장과 출처 JSON의 일치 확인이며 청취/CER 검증은 아니다.

```bash
.venv/bin/python -B scripts/run_korean_melo_spec.py \
  --manifest runs/private_korean/input.json --validate-only

runs/melo_mac_actual_20261105_z1muic00/venv/bin/python -B \
  scripts/run_korean_melo_spec.py \
  --manifest runs/private_korean/input.json \
  --assets runs/melo_mac_actual_20261105_z1muic00 \
  --seed 52 --epsilon 8 --iterations 1 10 --device cpu
```

`private_korean`과 파일명은 예시다. 실제 녹음이 준비된 뒤 실행한다.
다른 컴퓨터는 동일 리비전의 공식 `MeloTTS/`, `model/`,
`effective_loss_config_verified.json`을 가진 assets 폴더와 호환 환경을 별도로 준비한다.
네트워크 다운로드나 패키지 설치를 자동으로 수행하지 않는다.

## 출력·검증 계약

매 실행 `runs/korean_melo_spec_<고유값>/`을 새로 만들고 기존 결과를 덮어쓰지 않는다.

```text
run.json                       # 실행 상태·리비전·설정·입력/모델 해시·시간
backend.json                   # Python·torch·device·팀원 코드 해시·strict 로딩
input_manifest.json            # 입력 선언 스냅샷 (비공개)
effective_config.json
team/melo_kr/                  # 고정 Git 객체에서 읽은 원본 팀원 코드
<발화ID>/clean/<발화ID>.wav      # 원본과 바이트 동일한 복사본
<발화ID>/spec_1/                # 보호 WAV·report·gradients·validation·stoi JSON
<발화ID>/spec_10/
handoff_manifest.json          # 성공한 전체 쌍의 manifest
finetune_inputs/               # 실제 녹음만 생성. 아래 참고
```

매 반복 loss·gradient 유한성 및 비영 gradient를 검사하고 loss 성분·반복 수를 확인한다.
실패 시 종료 코드 1, `run.json` FAIL을 남긴다. 입력/모델 사전 검사 실패는 실행 폴더 생성 전에 종료할 수 있다.
실패 폴더의 부분 WAV를 전달하지 않는다.

기존 `validate_melo_spec_pair.validate()`와 `evaluate_melo_spec_stoi.evaluate_validation()`을 그대로 재사용한다.
PCM16 저장 오차는 기존 `epsilon + 1/32768 + 1e-7` 규칙을 적용한다.
검증 JSON에서 평가 직전·직후 해시를 다시 확인한다. STOI는 명료도이며 복제 방어 효과가 아니다.

## 파인튜닝 전달

manifest에는 발화 ID·대본·가명 화자·split·원본/clean/protected 경로와 해시,
모델 화자 ID·입력 종류·epsilon·반복 수·seed·STOI·검증 경로·모델/소스 provenance를 기록한다.
clean/protected의 대본·화자·split을 서로 대조하며 반복 조건 사이의 동일성도 검사한다.

실제 녹음만 `finetune_inputs/<clean|SPEC_1|SPEC_10>/<split>/<speaker_id>/`에
`wavs/`와 `transcripts.csv(file,text)`를 만든다. 원본 샘플레이트와 바이트를 보존한다.
합성·Mock는 `NONHUMAN_NOT_FOR_FINETUNING` manifest만 생성하고 이 내보내기는 거부한다.

모델 담당자는 **자신의 브랜치·환경**에서 각 조건·split·화자별로 아래 형식으로 준비한다.

```bash
python melo_kr/prepare_finetune.py \
  --audio-dir <run>/finetune_inputs/clean/train/s001/wavs \
  --transcripts <run>/finetune_inputs/clean/train/s001/transcripts.csv \
  --speaker s001 --name clean_train_s001

python melo_kr/prepare_finetune.py \
  --audio-dir <run>/finetune_inputs/SPEC_10/train/s001/wavs \
  --transcripts <run>/finetune_inputs/SPEC_10/train/s001/transcripts.csv \
  --speaker s001 --name spec10_train_s001
```

`<run>`은 실제 경로로 교체한다. validation/test도 원래 split을 유지한다.
담당자 준비 코드는 44.1kHz로 변환하므로 변환 후 파일은 별도 파생 데이터다.
원래 보호 WAV 검증/STOI와 학습용 변환 후의 효과를 구분하고, 학습 변환 뒤 보호 효과도 후속 확인한다.

**호환성 한계:** `prepare_finetune.py`는 WAV·metadata.list·transcripts.csv만 생성하며
README가 참조하는 학습 `config.json`을 생성하지 않는다. 학습/검증 목록 연결,
화자 매핑·초기 가중치·동일 학습 step·seed 설정은 모델 담당자가 확정해야 한다.
최신 스크립트가 이 학습 config를 만들거나 파인튜닝을 성공했다고 보고하지 않는다.
서로 다른 반복 조건을 한 protected 학습 세트로 섞지 않는다.

경로는 로컬 절대 경로이므로 다른 PC로 옮길 때 새 경로로 재매핑하고 모든 WAV 해시를 재검증한다.
이동한 검증 JSON의 절대 경로를 그대로 신뢰하지 말고 기존 검증 CLI를 새 경로로 재실행한다.
음성·개인정보·manifest·허가 기록은 공개 GitHub에 올리지 않는다.

## 자동 테스트

```bash
.venv/bin/python -B -m unittest discover -s scripts -p 'test_*.py' -v
```

기존 21개 유지 + 신규 14개 = 35개 통과. 신규 테스트는 수학 신호·Mock 선언만 사용한다.
입력·허가·대본 선언·중복 split·잘못된 WAV·해시 변경·epsilon 위반·출력 덮어쓰기,
실제 녹음 전용 내보내기와 실패 시 FAIL 보고를 확인한다.
템플릿 기본값은 검수 False여서 validate-only에서 의도대로 거부되는 것을 확인했다.
테스트 로그: `runs/melo_spec_20261110_audit_6yy9dv6l/tests_final.log`.

## 실제 녹음 확보 후 우선 작업

1. 데이터 담당자에게 한국어 녹음·검수 대본·사용 허가·고정 split과 해시를 받는다.
2. 최소 한 발화에서 1/10회 SPEC·저장 WAV 검증·STOI를 실행한다.
3. 다발화·다화자로 늘리고 환경·시간·메모리 한계를 확인한다. 초기 runner는 1~10초로 제한한다.
4. 모델 담당자에게 실제 전달 manifest·split별 준비 입력과 config 누락 사항을 넘긴다.
5. clean/protected 동일 조건 파인튜닝·합성 후 SIM/CER·강건성 평가를 연결한다.

11/17까지 7일이며, 현재 가장 큰 지연 요인은 실제 데이터 미확보와 학습 config/공격 평가 미완료다.

## 이번 실제 가중치 실행 결과 (합성 입력)

최종 실행: `runs/korean_melo_spec_85hbcsho/`.
입력은 기존 `korean_synthesis.wav` (44.1kHz·mono·196,096 samples·약 4.4466초),
SHA-256 `ca64cfae3ff5c82c8e6080535b593bf58afc95bf191f60c6d63ff1d914d7041f`다.
전체 실행 시간은 약 48.42초이며 CPU에서 실제 가중치로 실행했다.

| 조건 | STOI | 저장 후 최대 변화량 | gradient 검사 | WAV 검증 |
|---|---:|---:|---|---|
| 1회 | 0.9961603424775992 | 0.00316776438648958 | 1회 유한·비영 | PASS |
| 10회 | 0.8757433629288034 | 0.0314030647277832 | 10회 모두 유한·비영 | PASS |

동일 입력·모델·seed로 1회와 10회를 실행했다. 10회 변화량은 epsilon 자체보다 조금 크지만
PCM16 1 LSB 허용 범위 안이다. 원본·모델·설정 해시를 실행 전후 확인했다.
이전 점검 실행 `runs/korean_melo_spec_22149dc0/`과 STOI가 동일했다.
이는 동일 환경의 반복 점검이며 다발화 통계 검증이 아니다.
11/5 실행과는 모델 호출 직전 seed 초기화 위치 등이 달라 당시 숫자의 정확한 재현으로 주장하지 않는다.

실제 가중치 실행과 입력 출처를 구분하여 실행 상태는 `NONHUMAN_EXECUTION_PASS`,
전달 manifest는 `NONHUMAN_NOT_FOR_FINETUNING`으로 기록했다.
**실제 사람의 녹음으로 한 SPEC 1회·10회 실험은 미실행**이다.
청취·한국어 CER·SIM·파인튜닝·복제 방어 효과는 이번에 측정하지 않았다.

재현 명령 (합성 입력으로만 실행):

```bash
runs/melo_mac_actual_20261105_z1muic00/venv/bin/python -B \
  scripts/run_korean_melo_spec.py \
  --manifest runs/melo_spec_20261110_audit_6yy9dv6l/synthetic_input.json \
  --assets runs/melo_mac_actual_20261105_z1muic00 \
  --allow-nonhuman --iterations 1 10
```

기존 두 실험 폴더의 가중치·WAV·JSON·팀원 소스 등 64개 파일을 사전·사후 해시 비교했고
변경 0개였다. 제외한 전용 환경·캐시·한국어 BERT·공식 소스 전체를 이 64개로 표현하지 않는다.
공식 소스는 별도 Git clean 확인, 전용 환경은 11/5 `pip freeze`와 일치했다.
관련 증거: `runs/melo_spec_20261110_audit_6yy9dv6l/preservation_after.json`.
