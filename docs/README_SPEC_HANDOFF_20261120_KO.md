# SPEC 실행·시연·연구 인계

2026-11-26 문서 통합. 최신 진입 절과11/20 상세 명세를 구분한다. 이 문서를 갱신하면서 보호 생성·학습·GUI 테스트·청취 배포를 다시 실행하지 않았다.

## 작업공간과 실행 환경

| 이름 | 이 PC의 경로·역할 |
|---|---|
| ResearchRoot | `C:\Users\윤채영\SafeSpeech-spec-perturbation` — 최신 프로필·실험·이 문서 |
| BackendRoot | `C:\Users\윤채영\SafeSpeech-desktop-integration-20261118` — 보호 CLI·브리지·학습/평가 |
| Python/Assets | `/home/safespeech/.venvs/melo-training-20261118/bin/python` / `/home/safespeech/models/melo-spec-pinned-20261118` |

아래 backend 명령은 BackendRoot에서 실행한다. 이 연구 브랜치에는 `protect_wav.py`·`spec_desktop_bridge.py`·bounded trainer가 병합돼 있지 않다. 최신 프로필은 기존 backend를 명시적으로 참조한다. 현재 문서 브랜치와 기존 연구/앱 브랜치의 역할을 섞지 않는다.

기존 WSL Python3.10·torch/torchaudio2.7.1+cu128·NumPy1.23.5 환경을 재사용한다. CPU STOI 전용 `requirements-spec-stoi.txt`의 NumPy1.26.4, 원논문 BERT requirements, 과거 Mac 패키지를 덮어 설치하지 않는다. 정확한 환경은 backend `docs/spec_app_environment_20261120.txt`, `requirements-spec-app-wsl.txt`와 아래 원문을 따른다. 새 PC 전체 설치는 미검증이다.

## 단일 WAV 보호 명령 — 기존 시연 프로필 ε2/10

다음은 **WSL bash**, 실제 실행은 필요할 때만 한다. 입력과 아직 존재하지 않는 출력 디렉터리를 직접 지정한다.

```bash
cd '/mnt/c/Users/윤채영/SafeSpeech-desktop-integration-20261118'
PY=/home/safespeech/.venvs/melo-training-20261118/bin/python
ASSETS=/home/safespeech/models/melo-spec-pinned-20261118
"$PY" -B scripts/protect_wav.py \
  --input '/absolute/path/input.wav' \
  --output-dir '/absolute/path/NEW-spec-job/result' \
  --assets "$ASSETS" --device cuda --epsilon 2 --iterations 10 --seed 52
```

입력은 mono44,100Hz PCM16 WAV, 1~10초, 유한값·비무음이다. 대본 없이 보호하며 자동 다운믹스·리샘플·분할하지 않는다. CLI ε단위는 `/255`: `--epsilon 2`=2/255, 최종 원본 대비257code 이하. 과거 ε8 프로필은1028code다.

종료 코드0과 마지막 stdout JSON의 `status=PASS`를 모두 확인한다. backend 로그가 앞에 있을 수 있다. 실제 `protected.wav`와 `result.json`의 출력 경로·입출력 SHA·요청 ε/반복/seed·길이/형식·strict ε를 확인한다. `optimized.wav`는 정수 투영 전 진단용으로 앱에 넘기지 않는다. PASS는 파일 계약 통과이며 실제 방어 확률이 아니다.

## 준비된 앱 사본과 캐시 정책

ResearchRoot의 PowerShell에서 기존 실행기를 사용한다. 아래 명령은 실행 방법이며 이번 문서 작업에서는 호출하지 않았다.

```powershell
Set-Location 'C:\Users\윤채영\SafeSpeech-spec-perturbation'
powershell -ExecutionPolicy Bypass -File '.\scripts\demo_noise_fix_20261124\Start-Demo.ps1'
```

이 실행기는 기존 WSL Qt 환경과 WSLg를 사용한다. `scripts/spec_demo_profile.py:create_client`는 명시적 ε2·10회·seed52를 전달한다. `config/spec_demo.template.json`은 기존 위치·내용을 유지한다. backend `SpecConfig`의 기본 ε8을 앱 시연 기본값으로 오인하지 않는다. silence_only·지각 가중·마스킹은 현재 앱 호출에 들어 있지 않다.

`runs/demo_noise_fix_20261124/live_app_jobs/spec-<UUID>/result/protected.wav`처럼 요청마다 새 작업을 생성하고, 검증된 `result.output_path`만 재생·저장한다. ‘가장 최신 파일’ 검색이나 과거 ε8 파일을 재사용하지 않는다. 재요청 시 이전 재생/결과를 초기화하고 저장 사본 SHA가 직접 출력과 같은지 확인한다. 처리 중 종료·강제 QThread 종료 정책은 [앱 계약](SPEC_APP_INTEGRATION_REVIEW_20261120_KO.md)을 따른다.

고정 시연 자료(ResearchRoot 기준):

- `runs/demo_noise_fix_20261124/final_demo_eps2/original.wav`
- `runs/demo_noise_fix_20261124/final_demo_eps2/protected_eps2_iter10.wav`
- `runs/demo_noise_fix_20261124/final_demo_eps2/AB_original_then_protected.wav`
- 같은 폴더 `demo_settings.json` 및 상위 `final_selection.json`: 설정·해시·한계의 원본

비교용 ε8은 `demo_wavs/01_existing_profile_eps8_iter10.wav`이며 고정 시연 파일과 구분한다. 같은 seed의 GPU 재생성은 보존된 파일과 바이트 동일성을 보장하지 않는다.

## 앱팀 반영·GUI 검증 범위

인계 패치 `patches/spec_demo_ui_20261124.patch`, 모듈 `scripts/spec_demo_profile.py`, `spec_demo_app.py`, `launch_spec_demo.py`, 전달 기록 `runs/demo_noise_fix_20261124/app_team_handoff.json`을 사용한다. **feature/ui 반영/병합은 확인되지 않았고 이번에도 실행하지 않았다.** 기준은 고정 커밋d678c2051e336b2be2a595cc37562030c7df1954다. 외부 팀원의 미Push 작업이나 현재 원격 변경은 조회하지 않았다.

11/20의 Windows→WSL·Qt 계약 검사와11/24 WSLg 창 기동/offscreen 업로드→GPU→재생 상태→저장2회는 별도 기록이다. 네이티브 Windows 전체 GUI·발표장 스피커·새 PC 설치 완료를 뜻하지 않는다. UI의 더미92%는 실제 신뢰도가 아니며 실제 결과의 미측정 값은 `None`으로 표시한다.

## 후배 블라인드 청취 평가

`runs/human_listening_20261125/`의15WAV는 동일5발화×Clean/ε2/ε2+silence_only다. 파일명/순서만 블라인드이며 사본의 BWF 메타데이터를 제거해도 PCM/음량은 유지했다. 원본 데이터나 보호 WAV를 이 문서 작업에서 변경하지 않았다.

| 자료 | ResearchRoot 기준 보존 위치 |
|---|---|
| 평가자 배포 ZIP | `runs/human_listening_20261125/SafeSpeech_listening_20261125.zip` |
| 검증 | 같은 폴더 `package_verification.json` |
| 비공개 정답·출처·음량/길이 | `organizer_private/answer_mapping.csv`, `answer_mapping.json`, `source_hashes.json`, `volume_length_audit.csv` |
| 설문 초안·음성 링크 목록 | `forms_drafts/FORM_DRAFT_KO.txt`, `question_draft.csv`, `audio_file_list.csv` |

ZIP은3,090,214bytes,15WAV+안내문1개, 보존 SHA-256은 `6b3cc2b29f341abe917e8c6c220c9cc817218f97f4817f153546f624a99076b7`다. 이 문서에서 새 CRC/오디오 재생 검사를 했다는 뜻은 아니다. 비공개 정답표·조건별 수치·추천 설정은 평가자에게 전달하지 않는다.

확인한 음성 목록의 공유 URL은15행 모두 비어 있다. 온라인 Forms·파일 접근 권한·실제 배포·응답 수집 완료를 확인하지 못했다. 다음 순서로 재개한다.

1. 패키지 검증 기록과 해시를 확인하고 Forms 초안으로 설문을 구성한다.
2. 공유 URL·평가자 권한을 채우고 평가자 입장에서 재생을 확인한다.
3. 익명 참가 코드·모집/분석 기준을 사전 고정한 뒤 ZIP과 설문만 전달한다.
4. 동일 기기·음량에서 잡음 거슬림/자연스러움/내용 이해를 기록한다. 재생 불가·미청취는 낮은 점수 대신 결측으로 둔다.
5. 응답을 확정한 뒤 진행자가 비공개 정답표와 결합하고 평가자·발화의 짝을 유지해 분석한다.
6. 유의한 차이가 없다는 사실만으로 불가청·동등을 선언하지 않는다. 발화 중 잔여 잡음과 복제 방어를 별도로 검증한다.

11/25 종료 시 로컬 커밋6205fbbc1fa0dcbf63b4448847e5b2e6705bf76b에는 `scripts/session_tools_20261125/` 소스10개가 저장됐고 Push하지 않았다. 실행 폴더 연결은 `SAFESPEECH_WORKSPACE`를 사용한다. 이와 마스킹 작업공간의 미커밋 방법 명세/코드는 보존한다. 과거 종료 문서의 ‘안전 종료’는 당시 시점의 판단이지 현재 PC 상태 점검 결과가 아니다.

해당 종료 도구 소스는 현재 연구 루트가 아니라 `runs/session_closeout_20261125/code_worktree`의 `work/low-noise-listening-closeout-20261125` 브랜치에 있다. 이 문서 통합으로 소스를 복사하거나 다른 브랜치를 병합하지 않았다.

## 재현·복구와 과거 환경 상세

새 보호 후보의 방어는 후보별 새 데이터로 별도 공격 학습해야 한다. 아래 재학습 명령·재생성 도구는 문서를 읽는 것만으로 실행하지 않는다. 모델·optimizer3개·RNG·완료 manifest를 함께 복구하고 기존 출력은 거부한다. 초기 Mock/합성 입력은 사람 녹음으로 바꾸지 않으며, 일반 입력 선언 경로와 고정 KSS 연구 정책 경로를 구분한다.

다음 원문의 상대 경로·명령은 모두 **BackendRoot와11/20 시점** 기준이다. ε8 예제는 역사적 프로필이며 최신 시연은 위 명시적 ε2 호출이다. CPU·Mac·WSL 및 GPU NumPy 환경을 혼합하지 않는다.

<details>
<summary>11/20 상세 명세 — manifest 필드·고정 자산·재개·원래 검증/백업 위치</summary>

# SPEC 앱 연동 인계 — 2026-11-20

`scripts/protect_wav.py`를 독립 프로세스로 실행하면 실제 WAV를 SPEC 보호 WAV로 변환한다. 엄격한 ε는 최종 PCM16 파일을 기준으로 검사한다. 연구 결과·KSS 사용 조건·실험 프로토콜은 [최종 보고서](SPEC_FINAL_REPORT_20261120_KO.md), Windows 앱 연결은 [앱 연동 점검](SPEC_APP_INTEGRATION_REVIEW_20261120_KO.md)에 있다. 이전 실행·검증·복구 명세는 이 문서에 통합했다.

## 검증 환경과 의존성

Windows + Ubuntu-22.04 WSL2, RTX 5060 Ti, Python 3.10 전용 환경에서 검증했다.

```bash
cd '/mnt/c/Users/윤채영/SafeSpeech-desktop-integration-20261118'
PY=/home/safespeech/.venvs/melo-training-20261118/bin/python
ASSETS=/home/safespeech/models/melo-spec-pinned-20261118
"$PY" -m pip check
```

현재 환경을 재사용하면 추가 설치 없이 실행한다. 설치 명세는 `requirements-spec-app-wsl.txt`, 전체 버전 기록은 `docs/spec_app_environment_20261120.txt`다. 새 환경 설치 시 torch/torchaudio `2.7.1+cu128`을 CUDA 12.8 wheel 저장소에서 먼저 설치한 뒤 해당 명세를 설치한다. 시스템 FFmpeg·libsndfile과 고정 backend 자산도 필요하다. 새 기기에서 처음부터 설치하는 전 과정은 이번에 재검증하지 않았다.

기존 `requirements-spec-stoi.txt`는 독립 CPU 테스트 환경용이며 NumPy 1.26.4를 지정한다. 이 GPU 환경의 NumPy 1.23.5와 섞어 설치하지 않는다.

새 전용 환경을 만드는 설치 명령 예시는 다음과 같다. 시스템 FFmpeg·libsndfile과 고정 자산은 별도로 준비한다. 아래 설치 절차 자체는 새 기기에서 재검증하지 않았다.

```bash
python3.10 -m venv /absolute/new-spec-venv
/absolute/new-spec-venv/bin/python -m pip install \
  torch==2.7.1+cu128 torchaudio==2.7.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
/absolute/new-spec-venv/bin/python -m pip install \
  -r docs/spec_app_environment_20261120.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128
/absolute/new-spec-venv/bin/python -m pip check
```

`ASSETS` 아래 `model/checkpoint.pth`, `model/config.json`, `effective_loss_config_verified.json`, 고정 MeloTTS 및 팀 backend 소스가 있어야 한다. 자산 내부 구조를 보존한다. 실행 시 고정 해시·revision을 검사하고 모델을 자동 다운로드하지 않는다. 초기 가중치 SHA256: `48e3ff3fd0b5348e095f0468e60ae727507564100f58142ef3a922ead6e0a4d0`. 데이터·가중치는 Git에 포함하지 않으며 로컬 자산과 최신 백업을 통해 인계한다. 자산 준비 과정은 아래 고정 자산·배치 실행·재현 절을 따른다.

## 실제 실행

다음 명령은 **WSL bash**에서 실행한다. 출력 디렉터리는 존재하지 않는 새 경로를 사용한다.

```bash
"$PY" scripts/protect_wav.py \
  --input runs/final_validation_20261120/app_handoff_examples/input.wav \
  --output-dir runs/spec-app-example-new \
  --assets "$ASSETS" --device cuda --iterations 10 --epsilon 8 --seed 52

# 같은 실행을 제공하는 쉘 예제
bash scripts/example_spec_protect.sh \
  "$PWD/runs/final_validation_20261120/app_handoff_examples/input.wav" \
  "$PWD/runs/spec-app-example-another-new"
```

실제 검증 입력·출력·진단 예제는 `runs/final_validation_20261120/app_handoff_examples/{input.wav,protected.wav,result.json}`에 있다. 이 음성 파일은 로컬 연구용이며 Git에 올리지 않았다. 최신 보완 백업에 포함한다. 임의의 정상 입력 WAV 절대 경로로 바꾸어도 된다. 연구에 사용한 원본 예제는 `runs/desktop_20261118/data/data_set/kss_part_200/train/wavs/kss_hf_00078.wav`다.

Python 앱에서 호출하는 예:

```python
import json
import subprocess

proc = subprocess.run([
    '/home/safespeech/.venvs/melo-training-20261118/bin/python',
    'scripts/protect_wav.py', '--input', '/absolute/input.wav',
    '--output-dir', '/absolute/new-output',
    '--assets', '/home/safespeech/models/melo-spec-pinned-20261118',
    '--device', 'cuda', '--iterations', '10', '--epsilon', '8', '--seed', '52',
], cwd='/mnt/c/Users/윤채영/SafeSpeech-desktop-integration-20261118',
   capture_output=True, text=True)
if proc.returncode:
    raise RuntimeError(proc.stderr)
result = json.loads(proc.stdout.strip().splitlines()[-1])
if result['status'] != 'PASS':
    raise RuntimeError(result)
protected_path = result['output']
```

Windows 앱은 `wsl.exe -d Ubuntu-22.04 -- <Python 경로> <스크립트 WSL 절대 경로> ...`로 호출하고 입력·출력도 WSL 경로로 전달한다. 앱에서 GPU 작업을 비동기 실행하고 취소·오류 표시·배포를 연결하는 작업은 앱팀 담당이다.

## 입출력 계약

| 항목 | 명세 |
|---|---|
| 입력 | mono, WAV/PCM16, 44,100Hz, 1~10초, 완전 무음 제외, 유한 샘플 |
| 전처리 | 호출자가 리샘플·다운믹스·분할. 자동 변환하지 않음 |
| 텍스트 | 대본 입력 불필요 |
| ε | `--epsilon 8`은 정규화 진폭 8/255. 최종 WAV는 정수 code 차이 1028 이하 |
| 반복·seed | iterations 1~200, seed 0~2³²−1 |
| 출력 | `protected.wav`: 입력과 같은 길이·44.1kHz mono PCM16 |
| 진단 | `result.json`, `generation*.json`, `validation.json`, `stoi.json`, backend·gradient 기록 |
| 중간 WAV | `optimized.wav`는 정수 투영 전 진단용. 앱 전달 파일은 `protected.wav` |
| 성공 | 종료 코드 0 **및** 마지막 stdout JSON의 `status=PASS` |
| 오류 | 종료 코드 1. 일부 생성 후 실패하면 `result.json`에 FAIL 기록. 사전 입력 검사 실패 시 결과 폴더가 없을 수 있음 |
| 덮어쓰기 | 기존 출력 디렉터리 거부. 실패 시 일부 결과가 남으면 새로운 경로로 재시도 |
| stdout | backend 로그가 앞에 올 수 있으므로 전체 stdout을 단일 JSON으로 파싱하지 않음 |

`result.json`에는 입출력 경로·SHA256, seed·반복·ε, 최종 `strict_linf`·`stoi`·`snr_db`·길이·형식을 기록한다. 진단 JSON의 세부 구조는 예제 파일을 따른다.

## 검증과 알려진 한계

```bash
"$PY" -m unittest discover -s scripts -p 'test_*.py' -v
"$PY" -m pip check
```

오늘 전체 자동 테스트 99개, pip check, 실제 WAV GPU 변환, 최종 ε·SNR 독립 검사, 기존 출력 재사용 거부와 해시 보존을 확인했다. 증거는 `runs/final_validation_20261120/`에 있다. 보호 입력 200개 보정본의 ε 초과는 0이며 별도 1,000-step 모델 학습·평가 결과를 최종 보고서에 수록했다.

GPU 기본 모드는 동일 입력·seed에서도 WAV 바이트 동일성을 보장하지 않는다. `--deterministic`은 현재 `reflection_pad1d_backward_out_cuda` 지원 부재로 실패하는 진단 옵션이다. 성공 보장 옵션으로 표시하지 않는다. CPU 경로의 완료 시간·성능은 미검증이다.

KSS 단일 화자 내부 비상업적 연구다. SPEC_10의 입력 품질 비용이 있으므로 무손실·불가청·복제 완전 차단으로 표시하지 않는다. 사람 청취·별도 화자 동의·외부 음성/가중치 공개 승인 상태는 기존 기록을 유지한다.

## 고정 자산과 모델 provenance

| 대상 | 고정 리비전·검증 내용 |
|---|---|
| 팀 Melo 코드 | `b6db484367d38909b4ffd74e06fcf1166cd1e0b9` |
| 공식 MeloTTS 소스 | `209145371cff8fc3bd60d7be902ea69cbdb7965a` |
| MeloTTS-Korean | `0207e5adfc90129a51b6b03d89be6d84360ed323` |
| 배포 config SHA256 | `74543376976dfadde45ba34336fa79c7e95509f43a7c2e701b22c0f71fd7695c` |
| 한국어 BERT | `kykim/bert-kor-base`, `1779cc0982ada0216dd6de0dd4e86fb78201926d`; 가중치 SHA256 `ae43a392e533ccb9fd38e5c65130aeee50381b87e81e795f9d90469accd78236` |
| ECAPA | `speechbrain/spkrec-ecapa-voxceleb`, `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` |
| ASR | openai-whisper `20240930`, multilingual small, language ko |

고정 모델은 strict state dict로 로딩한다. 배포 설정에 없는 `data.win_length=2048`, `data.n_mel_channels=128`, `data.mel_fmin=0`, `data.mel_fmax=null`, `train.c_mel=45`만 effective config로 보충하고 다른 설정 차이는 거부한다. 공식 소스 revision·clean 상태·모델/config 해시를 검사한다. 원본 config나 가중치를 보정하지 않는다. 모델 `KR:0`과 실제 사람의 가명 speaker ID는 별개다.

필요한 새 자산 경로에서만 다음 준비 명령을 실행한다. 기존 자산이 있으면 해시로 검증하며, 이 문서 정리 작업에서는 다운로드·설치·학습을 실행하지 않았다.

```bash
"$PY" scripts/prepare_melo_spec_assets.py --output /absolute/NEW-melo-assets
# 실제 학습용 특징·평가 모델 준비 스크립트의 인자를 먼저 확인
"$PY" scripts/download_korean_training_bert.py --help
"$PY" scripts/prepare_melo_evaluation_models.py --help
```

전체 다국어 melo.api 설치 검증과 한국어 전용 경로 검증은 구분한다. 현재 WSL 학습은 공식 한국어 G2P·실제 BERT 특징을 사용한다. 가짜 음소·무작위 생성기·영벡터 한국어 BERT를 사용한 것으로 바꾸지 않으며, 공식 모델에서 한국어가 사용하지 않는 중국어용 1024차원 입력만 영벡터다. 평가 모델의 파일 해시·로컬 경로는 `/home/safespeech/models/melo-evaluation-20261118/model_provenance.json`에 있다.

## 배치 입력 manifest와 내보내기

단일 WAV 앱 실행에는 대본이 필요 없지만 배치 연구용 `run_korean_melo_spec.py`는 입력 manifest를 검증한다. `docs/templates/korean_spec_input.template.json`은 기존 위치에 유지한다. 템플릿의 허가·검수 기본값을 실행 편의상 True로 바꾸지 않는다.

| 필드 | 의미·검사 |
|---|---|
| utterance_id / speaker_id | 유일한 안전 ID, 실명 대신 가명 speaker; 중복 ID·경로·해시의 split 중복 거부 |
| input_kind | human_recording / model_synthesized / mock를 사실대로 선언 |
| language / model_speaker | ko / KR |
| text / transcript_reference / transcript_verified | 대본과 근거·검수 상태. 일반 사람 음성 strict 경로는 검수 선언 필요 |
| source / research_use_permitted / permission_reference | 출처·사용 근거·확인 상태를 사실대로 기록 |
| permission_scope | 사람 음성 strict 경로는 spec_and_finetuning 범위 필요 |
| split | 미리 고정한 train / validation / test, ID·대본·특징을 조건 간 동일 유지 |
| original_path / original_sha256 / sample_rate | manifest 기준 상대 또는 절대 경로, 생성 전에 기록한 해시, 실제 Hz |

배치 runner의 mono·1~10초·유한값·비무음·진폭·선언 검사는 16000/22050/24000/44100/48000Hz를 지원한다. 이는 44.1kHz PCM16만 받는 앱 실행기 계약과 다르다. 파일 검사만으로 사람 녹음 여부·동의 진위·발음 일치를 증명하지 않는다.

```bash
"$PY" scripts/run_korean_melo_spec.py \
  --manifest /absolute/private-input.json --validate-only
"$PY" scripts/run_korean_melo_spec.py \
  --manifest /absolute/private-input.json --assets "$ASSETS" \
  --device cuda --seed 52 --epsilon 8 --iterations 1 10
```

합성/Mock 점검은 명시적 `--allow-nonhuman`을 사용한다. KSS 비상업 연구 정책 경로만 `--allow-licensed-research`를 명시하며, 고정 데이터·라이선스 근거·과거 선언을 재검증한다. 이 옵션은 임의의 사람 음성에 대한 허가 우회가 아니다. 현재 연구 사용 조건과 미검수 한계는 [최종 보고서](SPEC_FINAL_REPORT_20261120_KO.md)를 따른다.

새 `runs/korean_melo_spec_<고유값>/`에는 run.json, backend.json, 입력 manifest 사본, effective config, 고정 팀 소스, 발화별 clean/spec_1/spec_10의 보고서·gradient·validation·STOI와 handoff_manifest.json을 만든다. 실패 시 부분 WAV를 전달하지 않는다. 실제 사람 입력의 기존 내보내기는 `finetune_inputs/<조건>/<split>/<가명 speaker>/wavs/`와 transcripts.csv를 만들며 합성/Mock는 NONHUMAN_NOT_FOR_FINETUNING으로 구분한다.

레거시 배치 생성기의 PCM16 validator는 `epsilon + 1/32768 + 1e-7` 허용오차를 사용하므로 엄격한 ε와 구분한다. 앱에는 `protect_wav.py`의 최종 protected.wav만 전달한다. 배치 보정 사본은 `project_kss_strict_inputs.py`로 별도 생성하며 원본을 덮어쓰지 않는다.

학습 입력 준비는 `prepare_paired_melo_experiment.py --spec-run <새 run> --assets <assets>`가 조건별 dataset.json·config.json·metadata.list·cleaned 목록을 생성한다. 각 조건의 ID·대본·speaker·split·특징·초기 가중치를 동일하게 유지하고 서로 다른 반복 조건을 한 protected 세트로 섞지 않는다. 모델팀의 별도 melo_kr/prepare_finetune.py는 WAV·목록·transcripts까지만 만드는 경로이므로 학습 config 생성까지 완료됐다고 가정하지 않는다. 리샘플·재인코딩 후 파형은 별도 파생 데이터이며 원래 ε/STOI를 그대로 적용하지 않는다.

청취를 추가로 검수할 때에는 기존 `Review-KSS.cmd`를 사용한다. 가명 검수자, F5 재생, F8 일치/F9 불일치/F10 보류를 기록하며 재생 전·필요 재생 시간 전에는 판정 저장을 거부한다. 결과는 `runs/kss_listening_review_200/transcript_check_reviewed.csv`와 review_events.jsonl에 저장한다. 동시 창 편집은 지원하지 않는다. 기본 manifest·WAV·허가 플래그를 수정하거나 ASR 일치를 청취로 바꾸지 않는다. `build_authorized_kss_manifest.py`·`run_authorized_korean_training.py`는 별도 승인/검수 strict 경로이며 공개 라이선스 정책과 구분한다.

## 독립 WAV 검증·STOI 평가

생성 전에 기록한 원본 SHA256을 사용한다. 검증 시 처음 계산한 해시만으로 생성 이전의 원본 불변성을 증명할 수 없다. 공통 생성 보고서의 record-index·basename·Hz·길이·형식·유한값·손실·ε·seed·원본/보호/생성 보고서 해시를 검사한다.

```bash
"$PY" scripts/validate_melo_spec_pair.py \
  --generation-report /absolute/generation.json \
  --original /absolute/clean.wav --protected /absolute/protected.wav \
  --expected-original-sha256 ACTUAL_PREGEN_SHA256 \
  --execution-id new-validation --input-kind melo_artifact --record-index 0 \
  --output runs/NEW-validation.json
"$PY" scripts/evaluate_melo_spec_stoi.py --validation runs/NEW-validation.json
```

기존 evaluator는 매번 새 `runs/melo_stoi_*`에 JSON·CSV를 기록한다. mock_contract는 MOCK_PASS이며 실제 모델 실행 증거가 아니다. melo_artifact의 형식·해시 통과만으로 생산자의 가중치·revision 진위를 증명하지 않는다. 현재 앱 실행기는 별도로 고정 자산 provenance와 엄격한 ε를 검사한다. 경로를 다른 PC로 옮기면 절대 경로를 재매핑하고 모든 해시·검증을 다시 확인한다.

## 재검산·평가·보고서 재생성

다음 명령은 이미 완료된 실험을 읽고 새 결과 폴더에 검산·보고서를 생성한다. 기존 저장 경로를 재사용하지 않는다. 이 문서 통합 작업에서 완료된 학습이나 평가를 다시 실행하지 않았다.

```bash
"$PY" scripts/audit_kss_final.py \
  --study runs/kss_research_study_b0msiw2t/study.json --output runs/NEW-kss-audit
"$PY" scripts/recompute_kss_sim.py \
  --evaluation runs/melo_smoke_evaluation_5am0uct6/evaluation.json \
  --audit runs/final_validation_20261119/audit_strict/audit.json --output runs/NEW-seed53-sim
"$PY" scripts/build_spec_final_report.py \
  --validation-root runs/final_validation_20261119 --output-directory runs/NEW-historical-presentation
"$PY" scripts/build_spec_release_report.py --validation-root runs/NEW-release-validation
```

`build_spec_final_report.py`는 11월 19일 시점의 역사적 결과를 생성하는 도구이며 보정본 미측정 문구도 그 시점의 기록이다. 현재 결론은 [최종 보고서](SPEC_FINAL_REPORT_20261120_KO.md)를 따른다. `build_spec_release_report.py`는 validation root에 기존 seed53/strict 평가 위치, SIM 재검산, smoke, tests.log, pip_check.log 등의 실제 증거가 필요하고 presentation 폴더가 이미 있으면 중단한다. 기존 `runs/final_validation_20261120/`를 그대로 다시 지정하면 기존 presentation 때문에 실패한다. 재생성 시 필요한 증거를 새 root에 준비하고 입력 경로·해시를 검증한다. 생성된 Markdown을 docs에 배포할 때 그림도 현재 docs의 보존 자산에 맞게 배치한다.

추가 학습을 별도로 결정했을 때만 `run_kss_seed_repeat.py --training-summary <기존 summary> --assets <assets> --output-root <새 WSL 디렉터리> --seed <새 seed> --steps 1000` 또는 `run_strict_kss_validation.py --output <새 evidence> --checkpoint-root <새 WSL 디렉터리> --assets <assets>`를 사용한다. 이 둘은 실제 학습을 시작하므로 단순 검산·문서 정리에 사용하지 않는다.

재개 명령은 `train_melo_bounded.py --config <동일 config> --assets <assets> --resume <완전한 step 디렉터리> --steps <저장 step보다 큰 총 목표> --checkpoint-every <간격> --output-root <새 WSL 디렉터리>`다. config·dataset manifest·generator/state 해시가 일치해야 한다. generator·파형/길이 판별기·optimizer 3개·loss history·CPU/CUDA/NumPy/Python RNG를 복원한다. 3-step 당시 구형 checkpoint는 자동 복구 대상이 아니다. 완료 manifest 없는 저장은 거부하고 기존 checkpoint는 덮어쓰지 않는다.

## 증거·체크포인트·백업 인계

| 항목 | 보존 위치 |
|---|---|
| seed52 학습·실험 | `runs/paired_melo_training_rhp4a8t2/summary.json`, `runs/paired_melo_experiment_ong85l9i/` |
| seed52 평가 | `runs/melo_smoke_evaluation_qyh7qbgy/` |
| seed53 학습·평가 | `runs/kss_seed_53_8dvd53re/summary.json`, `runs/melo_smoke_evaluation_5am0uct6/` |
| 엄격한 ε 보정본 학습·평가 색인 | `runs/final_validation_20261120/strict_experiment/summary.json`, `evaluation_location.json` |
| seed53 checkpoint | `/home/safespeech/experiments/SafeSpeech-KSS-final-20261119/` |
| 보정본 checkpoint | `/home/safespeech/experiments/SafeSpeech-KSS-strict-20261120/` |
| 독립 검산·품질·decoder·GPU 실패 기록 | `runs/final_validation_20261119/`, `runs/final_validation_20261120/final_verification.json`, `checkpoint_state_verified.json`, `wav_smoke_checks.json` |
| 최신 표·그림·환경 | `docs/spec_final_results_20261120.csv`, `docs/spec_input_quality_20261120.csv`, `docs/figures/spec_release_results_20261120.svg`, `docs/spec_app_environment_20261120.txt` |

과거 합성 스모크의 상세 input quality·학습·평가 CSV는 `runs/training_20261118/presentation_artifacts/`와 `runs/benchmark_20261118/presentation/`에 보존했다. 실험별 손실·gradient·메모리·timing·bootstrap·강건성 품질 원본을 문서 축소를 이유로 삭제하지 않았다.

seed53 Clean generator SHA256은 `2b842c5963a2e94bd568f1c8b22e532ae9a2705a6a81bb2df447104511632f0b`, SPEC_10은 `064bdfdc0e1e0c3541b76ede2d034a6d16b8ccb1f12cf803913f89fe6b6041b1`이다. 마지막 저장 step=완료 step=1,000, 손실 step=0이며 optimizer 3개도 step1,000·LR0.0001이다. 전체 경로·generator/training_state 해시는 `checkpoint_state_verified.json`을 따른다.

기존 약 29GB ZIP `C:/Users/윤채영/SafeSpeech-backups/SafeSpeech_20261118_030155.zip`과 보완 ZIP `SafeSpeech_SPEC_final_20261120_215028.zip`을 함께 보관한다. 보완 ZIP은 3,738,788,779 bytes, 943개 파일의 전체 CRC·SHA256 검증을 통과했고 seed53 두 조건·보정본의 generator 3개와 optimizer/RNG state 3개를 포함한다. 보완 ZIP SHA256은 `677780cd5bfe1c1b0d985fbf0295c420cf5ec30c79ad777dba5434bf6043bd79`, 기존 ZIP은 `fdea386727762249d217b78320be64a1dee44f35475af670c19eacef606e071a`다. 이번 문서 정리로 백업을 갱신한 것은 아니다.

복원은 기존 ZIP의 payload/workspace를 프로젝트에, payload/native_experiments를 WSL experiments에 복원한 뒤 보완 ZIP project/와 wsl-home/를 대응 위치에 적용한다. 자산·환경·경로를 검증하고 기존 결과를 무조건 덮어쓰지 않는다. ZIP 내 BACKUP_MANIFEST.json과 옆 .sha256·.verification.json을 확인한다. 정확한 백업 상태는 `runs/final_validation_20261120/backup_release.json`, 원래 Git 배포 기록은 같은 폴더 git_release.json, 앱 연결 배포 기록은 `runs/app_integration_review_20261120/git_release.json`에 있다. 문서 축소 전 전체 본문은 커밋 `9882630dc3eabeeb801b1a3d1680e9e427067a58`에 보존됐다.

연구 검증 당시 자동 테스트 99개와 앱 브리지 추가 후 전체 113개 통과는 다른 시점의 기록이다. Qt까지 포함한 회귀는 기존 별도 `runs/app_integration_review_20261120/qt_test_venv/bin/python -B -m unittest discover -s scripts -p 'test_*.py' -q`로 수행한다. 이 환경은 PySide6 6.8.3과 기존 연구 site-packages를 읽기 전용으로 참조한다. Qt가 없는 환경의 skip을 전체 Qt 검증 성공으로 보고하지 않는다. 새 PC 설치·Windows 전체 GUI·오디오 장치 재생은 미검증이며 현재 앱 연결·미측정 confidence·파일 형식·작업 수명 정책은 [연동 점검](SPEC_APP_INTEGRATION_REVIEW_20261120_KO.md)을 따른다.

</details>
