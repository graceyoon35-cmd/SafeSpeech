# SafeSpeech 시연 재개 — 2026-11-24 종료 기록

오늘 예정한 ε=2/255·10회와 ε=4/255·10회 보호 데이터 생성, 각 1,000-step 공격 파인튜닝, 동일 30문장 SIM·CER 평가를 완료했다. 새 학습·실험을 자동으로 시작하지 않는다.

## 내일 먼저 실행할 시연

추천 설정은 **SPEC, ε=2/255, 10회, seed 52, 기본 step=(2/255)/10**이다. CLI 인자는 `--epsilon 2`이며 `--epsilon 2/255`가 아니다. 입력은 mono 44,100Hz PCM16 WAV, 1~10초다.

프로젝트 루트 `C:\Users\윤채영\SafeSpeech-spec-perturbation`의 PowerShell에서 실행한다.

```powershell
powershell -ExecutionPolicy Bypass -File ".\scripts\demo_noise_fix_20261124\Start-Demo.ps1"
```

이 실행기는 기존 WSL Qt 환경과 WSLg를 사용한다. 기존 backend 저장소는 `C:\Users\윤채영\SafeSpeech-desktop-integration-20261118`이다. 원래 시연 폴더의 `runs\demo_noise_fix_20261124\Start-Demo.ps1`도 보존했다.

1. `runs\demo_noise_fix_20261124\final_demo_eps2\AB_original_then_protected.wav`를 동일 음량으로 직접 듣는다. 순서는 원본 → 0.6초 무음 → 보호본이다. **사람 청취 검수는 아직 미완료**이며 사용자 답변도 ‘아직 듣지 못함’이었다.
2. 위 명령으로 앱 사본을 실행한다. 원본 입력은 `runs\demo_noise_fix_20261124\final_demo_eps2\original.wav`다. 발화는 “우리는 서로 답을 비교해 보았다.”이며 약 3.2초다.
3. WAV 추가 → 보호 시작 → 원본/보호 재생 → 새 이름으로 결과 저장을 확인한다. 새 실행은 `runs\demo_noise_fix_20261124\live_app_jobs\spec-<UUID>\result\protected.wav`를 생성한다.
4. `result.json`의 `epsilon_units_255=2`, `iterations=10`, `seed=52`, 입력·출력 SHA-256을 확인한다. 앱은 요청 설정·해시·샘플 길이·엄격한 ε를 검사한다.
5. 현장 스피커에서 직접 청취한다. GPU 재생성은 고정 시연 파일과 바이트 단위로 같다는 보장이 없다. 고정 파일을 사용할 때에는 아래 보존된 보호본을 선택한다.

시연용 고정 파일:

- 원본: `runs/demo_noise_fix_20261124/final_demo_eps2/original.wav`
- 추천 보호본: `runs/demo_noise_fix_20261124/final_demo_eps2/protected_eps2_iter10.wav`
- 청취 비교: `runs/demo_noise_fix_20261124/final_demo_eps2/AB_original_then_protected.wav`
- 실제 설정·해시: `runs/demo_noise_fix_20261124/final_demo_eps2/demo_settings.json`
- 최종 추천과 한계: `runs/demo_noise_fix_20261124/final_selection.json`

원본 SHA-256은 `1e6eafa93d90e2c132b03d14f76b7046f633ba77ff79b4de067fc7d53e8b2a8f`, 추천 보호본은 `3dbb2e7b9fc73c7dc9e56ef549337cddd3f224bbaada2450f4930e60c017d591`이다. 심한 잡음의 기존 ε=8 파일은 `demo_wavs/01_existing_profile_eps8_iter10.wav`에 비교용으로만 남아 있고 시연 전용 폴더에는 넣지 않았다. `optimized.wav`를 시연 결과로 선택하지 않는다.

## 오늘 확인한 결과

아래 SNR·STOI는 동일한 시연 입력 1개, SIM·CER는 동일한 평가 문장 30개다. CER는 평균 문자 오류율이며 복제 차단 확률이 아니다.

| 설정 | 시연 SNR | 시연 STOI | 복제 SIM | 복제 CER |
|---|---:|---:|---:|---:|
| Clean 기준 | — | 1.00000 | 0.5101 | 13.46% |
| **ε=2/255·10회** | **23.46dB** | **0.97955** | **0.4273** | **37.86%** |
| ε=4/255·10회 | 17.34dB | 0.94704 | 0.2998 | 92.86% |
| 기존 ε=8/255·10회 | 11.64dB | 0.88964 | 0.1627 | 173.62% |

ε=2의 시연 SNR은 기존 설정보다 11.82dB 높고 섭동 RMS는 74.35% 작다. 원본 RMS 0.08425879, 보호본 RMS 0.08435664이며 볼륨 감소·필터를 적용하지 않았다. 시연 보호본은 모두 클리핑 0, 엄격한 ε 통과다.

새 학습 입력 140개의 평균은 ε=2에서 SNR 23.22732dB·STOI 0.981106, ε=4에서 17.48083dB·0.948037이며 클리핑은 모두 0이다.

ε=2·4는 **각각 새 보호 데이터로 모델을 학습**했다. 기존 Clean·ε=8 체크포인트는 해당 조건의 비교 기준으로 재평가했으며 새 설정의 방어 입증에 재사용한 것으로 표시하지 않았다. 초기 가중치·트레이너 코드·학습 seed 52·입력 순서·학습 1,000-step을 일치시켰다. train 140, validation 30, test 30 분할을 유지했다.

ε=2는 Clean 대비 SIM 차이 -0.08279, CER 차이 +24.40%p다. 문장 단위 paired bootstrap 95% 구간은 SIM [-0.10140, -0.06474], CER [+15.47, +33.84]%p다. 단일 화자·단일 학습 seed·동일 Melo 공격에 한정한 관찰이며, 이 구간은 화자·학습 seed의 변동을 나타내지 않는다. 다른 복제 모델, 장기 학습, 보편적 방어 확률은 미검증이다. 기존 ε=8 모델은 보정 전 입력으로 학습됐고, 오늘 생성한 엄격한 ε=8 시연 WAV와는 구분한다.

주요 결과:

- `runs/demo_noise_fix_20261124/comparison.csv`, `comparison.json`
- `runs/demo_noise_fix_20261124/demo_quality.csv`
- `runs/demo_noise_fix_20261124/defense_attack_140/SPEC_eps2_iter10/quality.json`
- `runs/demo_noise_fix_20261124/defense_attack_140/SPEC_eps4_iter10/quality.json`
- `runs/demo_noise_fix_20261124/defense_attack_140/runs/melo_smoke_evaluation_h0eckqj8/`: 원문·ASR·편집 횟수·SIM·CER·합성 WAV
- `runs/demo_noise_fix_20261124/final_verification.json`: CER 150건 독립 검산, 기존 파일 415개와 평가 모델 8개 변경 0

## 체크포인트와 복구 상태

두 모델 모두 1,000-step 정상 완료했다. 각 폴더의 `checkpoint.json`을 기준으로 파일 해시를 검사해야 한다.

```text
runs/demo_noise_fix_20261124/defense_attack_140/checkpoints/
  melo_train_SPEC_eps2_iter10_6t75lk4c/checkpoints/step_001000/
  melo_train_SPEC_eps4_iter10_bp0o310j/checkpoints/step_001000/
```

각 폴더에는 `G_1000.pth`, `training_state.pth`, `checkpoint.json`이 있다. `training_state.pth`에는 discriminator·duration discriminator·optimizer 3개·Torch/CUDA/NumPy/Python RNG·학습 이력이 함께 들어 있다. 구성 파일과 학습 보고서는 각 모델 run 상위 폴더, 학습 로그는 `SPEC_eps2_iter10/training.log` 및 `SPEC_eps4_iter10/training.log`에 있다. 종료 요청 후에는 학습 재개·새 실험을 실행하지 않는다.

최초 분할 수 사전 점검 실패와 두 번째 backend 로드 시 모듈 경로 검사 중단 기록도 보존했다. 완료된 ε=2 학습을 재실행하지 않고 ε=4부터 이어서 정상 종료했다. 실패 기록을 삭제하거나 성공 기록으로 바꾸지 않았다.

## 앱팀 인계 상태

**패치는 아직 feature/ui에 반영하거나 병합하지 않았다.** 확인한 원격 커밋은 `d678c2051e336b2be2a595cc37562030c7df1954`이며 실제 원격 조회와 패치 기준이 일치했다.

- `patches/spec_demo_ui_20261124.patch`: 실제 처리기 연결, WAV 정책, 미측정 신뢰도, 작업 중 종료 방지.
- `scripts/spec_demo_profile.py`, `spec_demo_app.py`, `launch_spec_demo.py`: 명시적 ε=2·10회 호출과 앱 실행기.
- `config/spec_demo.template.json`: 설정 템플릿.
- `runs/demo_noise_fix_20261124/app_team_handoff.json`: 전달 대상과 실제 경로.
- `runs/demo_noise_fix_20261124/app_snapshot/ui/`: 이미 패치하고 검증한 앱 사본.

기존 backend `SpecConfig`의 기본 ε는 8이므로 명시적으로 2를 지정하는 새 factory/실행기를 사용해야 한다. 매번 새 UUID 작업을 생성하고 `result.output_path`만 재생·저장한다. 과거 결과 경로나 ‘최신 파일’ 검색을 이용하지 않는다.

WSLg 실제 창 기동과 offscreen 실제 앱의 파일 선택→GPU 처리→재생→저장을 2회 검증했다. Qt 재생기는 원본·보호본에서 PlayingState/NoError에 도달했고 저장본 해시가 직접 출력과 같았다. 재실행 시 이전 재생·결과 초기화와 새 작업 경로를 확인했다. **네이티브 Windows Qt 전체 GUI와 발표장 스피커는 미검증**이다.

## 소스 저장과 백업

종료용 작업 브랜치는 `research/demo-noise-closeout-20261124`다. master와 feature/ui를 수정하지 않으며 강제 Push를 하지 않는다. 실제 Commit·Push 영수증은 `backups/closeout_20261124/git_receipt.json`을 확인한다.

실험 원본 스크립트는 runs에 그대로 보존했다. Git용 사본은 `scripts/demo_noise_fix_20261124/`, `scripts/audio_quality_diagnosis_20261123/`에 저장했다. 사본은 출력 경로를 runs로 고정했으며 수동 GPU/GUI 점검은 `manual_*_check.py`로 분리했다. 이 파일들을 일반 자동 테스트로 실행하거나 오늘 실험을 재실행하지 않는다.

새 백업:

```text
backups/closeout_20261124/SafeSpeech_demo_noise_20261124.zip
backups/closeout_20261124/source_manifest.json
backups/closeout_20261124/verification.json
backups/closeout_20261124/SafeSpeech_demo_noise_20261124.zip.sha256
```

백업에는 두 진단·시연 run 전체, 새 모델 가중치와 optimizer/RNG 상태, 결과·로그·WAV·앱 사본·패치·선별 소스·이 문서를 포함한다. 모든 ZIP 항목 CRC, 전체 ZIP SHA-256, 원본 대 ZIP 항목별 SHA-256을 검사하고 결과를 verification.json에 남긴다. 오늘 산출물의 증분 백업이며 과거 환경·기준 데이터 복원에는 기존 백업도 필요하다.

기존 `SafeSpeech-backups/SafeSpeech_20261118_030155.zip`과 `SafeSpeech-backups/SafeSpeech_SPEC_final_20261120_215028.zip`은 읽기 전용으로 확인하고 변경·삭제하지 않는다. 체크포인트·음성·데이터·ZIP·실험 원시 결과는 Git에 추가하지 않는다.

종료 직전 프로세스 확인 결과는 `backups/closeout_20261124/safe_shutdown.json`에 기록한다. 백업·검증·Push가 끝나고 이 파일이 안전 상태이면 SafeSpeech 작업 기준으로 PC를 종료해도 된다. 내일 남은 우선 작업은 직접 청취, 현장 스피커와 실제 시연 흐름 확인, 앱팀의 별도 패치 반영이다.
