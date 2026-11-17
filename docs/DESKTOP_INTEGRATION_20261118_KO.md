# 데스크톱 환경 및 통합 검증 기록 — 2026-11-18 (KST)

기준은 `feature/spec-perturbation`의 `412c35887b64a828e3fc44f8b9f0e807b8f04af3`이다.
원래 작업 공간과 브랜치는 그대로 두고, Windows Git의 별도 worktree
`C:/Users/<USER>/SafeSpeech-desktop-integration-20261118`와
`integration/desktop-20261118`에서 작업했다. **커밋·푸시·팀원 브랜치 병합은 하지 않았다.**
새 파일은 아직 untracked 상태이며, 실제 실험 파일은 Git 제외 경로 `runs/`에 있다.

## 1. WSL2 GPU 환경 — 완료

- Ubuntu 22.04.5 LTS / WSL2, Python 3.10.12.
- 기존 `/home/safespeech/.venvs/safespeech`를 사용했다. 패키지 설치·업그레이드는 하지 않았다.
- RTX 5060 Ti 8GB, compute capability 12.0, torch/torchaudio `2.7.1+cu128`, CUDA runtime 12.8.
- 실제 CUDA Conv1d 역전파·행렬곱·STFT·오디오 리샘플링 PASS. GPU 인식만 확인한 결과가 아니다.
- 드라이버가 표시하는 CUDA UMD 13.3은 PyTorch runtime 12.8과 별개다.
- `nvcc`는 없다. 검증한 경로는 미리 빌드된 PyTorch로 실행되었다.
- Windows RAM 32GB와 별개로 WSL에서 사용 가능한 메모리는 약 15GiB, swap 4GiB다.
- ffmpeg 4.4.2 및 MP3 인코딩·디코딩 실행 확인.
- `melo` 배포 패키지, SpeechBrain, Whisper, jiwer는 기본 환경에 설치되어 있지 않다.
  SPEC는 고정 Melo 소스 경로를 사용하므로 실행 가능하다. 전체 다국어 `melo.api`는 검증하지 않았다.
- 로그: `runs/desktop_20261118/environment.json`.

[PyTorch Blackwell 지원](https://pytorch.org/blog/pytorch-2-7/)은 CUDA 12.8 wheel을 안내한다.
기존 환경이 해당 조건을 만족하고 실제 연산도 통과해 재설치하지 않았다.

## 2. 팀원 브랜치 — 조사 및 필요한 코드 연결 완료

`git fetch origin` 후 고정 Git archive로 필요한 경로만 별도 `runs/desktop_20261118/`에 추출했다.
전체 리비전은 `remote_branches.log`에 기록했다.

| 역할 | 조사한 원격 HEAD | 확인 사항 |
|---|---|---|
| 데이터 | `feature/data-preparation`, `8958402` | 실제 KSS WAV 200개와 manifest·split·대본 있음 |
| Melo | `feature/melo-kr`, `b6db484` | SPEC 음향 경로 및 prepare/evaluate/synth 도구 있음. 새 자체 학습 루프 없음 |
| 평가 | `feature/metrics`, `27ae06d` | 일반 평가 스모크 코드 있음. 한국어 SIM/CER는 Melo 브랜치 코드 조사 |
| 평가 별도 | `evaluation`, `3345f42` | 기존 upstream README 수준의 HEAD. 새 한국어 평가 구현으로 판단하지 않음 |
| 강건성 | `feature/robustness`, `c25e004` | paired 변환, 15조건 sweep, batch, 검증·내보내기 있음 |
| UI | `feature/ui`, `d678c20` | PySide6 UI. 더미 처리이며 실제 SPEC 연결 미구현 |

강건성 스크립트 6개와 해당 테스트 5개만 통합 작업 공간의 `scripts/`에 복사했다.
공통 `audio_metrics.py`는 기존 파일과 동일해 교체하지 않았다.
추가 bridge는 성공한 SPEC handoff의 상태·분류·WAV 해시·epsilon을 검사한 뒤
강건성 입력 manifest를 생성한다. 실패한 실행·변조 WAV·잘못된 사람 음성 주장은 거부한다.

Windows worktree의 `.git`에 기록된 `C:/...` 경로를 WSL Git이 읽지 못하는 문제를
`run_korean_melo_spec.py`에서 해당 Git 명령에만 경로를 변환해 해결했다.
공식 Melo 저장소의 Git 검사에는 이 환경을 전달하지 않는다.

## 3. KSS — 파일 확보·기술 검증 완료 / 사용·청취 승인 대기

로컬 사본: `runs/desktop_20261118/data/data_set/kss_part_200/`.
WAV 200개, train 140 / validation 30 / test 30, 모두 44.1kHz 모노 PCM16.
데이터 담당 검증기로 해시·길이·중복·분할·대본 목록을 검사해 PASS했다.
추가로 manifest.json·spec_input.json의 ID·대본·split·해시와 split 목록의 정확한 집합을 대조했다.
원격에서 추출한 사본이며 원본 또는 팀원 브랜치를 수정하지 않았다.

**사용 확인 0/200, 청취 검수 선언 0/200.** 이용 상태는 `pending_team_confirmation` 그대로다.
SPEC 실행기를 `--validate-only`로 호출했을 때 `Require reviewed transcript declaration`로
거부되었다. 이 예상된 거부는 모델 실행 성공이 아니다.

배포 문서의 라이선스 표기는 CC BY-NC-SA 4.0이며, 실제 받은 자료는 Hugging Face 미러다.
원본과의 동일성·개별 화자 동의·SPEC 및 음성 복제 학습 활용 권한을 이 작업에서 승인하지 않았다.
팀은 사용 범위에 대한 확인 기록을 남기고, 사람이 `transcript_check.csv`에 실제 청취 결과를 기록해야 한다.
확인한 발화만 별도 입력으로 구성해야 하며, 전체 일괄 승인 플래그를 임의로 켜면 안 된다.

## 4. SPEC — 실제 가중치 GPU 기술 스모크 완료

공식 소스 `209145371cff8fc3bd60d7be902ea69cbdb7965a`, 한국어 모델
`0207e5adfc90129a51b6b03d89be6d84360ed323`를 내려받았다.
위치: `/home/safespeech/models/melo-spec-pinned-20261118`.
checkpoint SHA-256 `48e3ff3fd0b5348e095f0468e60ae727507564100f58142ef3a922ead6e0a4d0` 및
배포 config SHA-256 `74543376976dfadde45ba34336fa79c7e95509f43a7c2e701b22c0f71fd7695c` 일치.
기존 runner의 누락 필드 overlay만 적용했다. 소스 clean 검사 및 strict state dict 로딩 PASS.

Mac의 합성 WAV·BERT·실행 harness는 이 PC에 없다. Git 제외된 다른 PC의 `runs/`는 fetch로 복원되지 않는다.
이번 입력은 직접 생성한 **2초 chirp+seeded noise 수학 신호**이며 한국어 합성 음성이 아니다.
manifest의 한국어 문장은 계약 검사 전용 자리표시자이며 실제 발화가 아니다.

- seed 52, epsilon 8/255, SPEC 1회·10회, CUDA.
- 모든 반복에서 유한·비영 gradient와 유한 loss 검사 PASS.
- 저장 PCM16 WAV의 길이·샘플레이트·epsilon(기존 1 LSB 허용 포함)·해시 검사 PASS.
- 최초 실행: `runs/korean_melo_spec_q4zkv8az/` (전체 약 8.91초).
- 메모리 측정 실행: `runs/korean_melo_spec_sef9_n92/`.
- 메모리 측정 실행의 PyTorch peak allocated 1,231,045,632 bytes (약 1.15GiB),
  reserved 1,413,480,448 bytes (약 1.32GiB). 데스크톱 전체 VRAM 사용량은 이 값에 포함되지 않는다.
- STOI 계산 경로는 실행했으나, 수학 신호의 STOI를 음성 명료도나 한국어 보호 효과로 해석하지 않는다.
- 실행 상태 `NONHUMAN_EXECUTION_PASS`, handoff `NONHUMAN_NOT_FOR_FINETUNING`.
  실제 사람 보호 실험·한국어 합성 재현·화자 복제 방어 효과는 미측정이다.

## 5. 파인튜닝 — 조사·설정안 준비 / 실행 안 함

팀원 `prepare_finetune.py`는 `file,text` CSV에서 WAV·4열 metadata.list를 생성한다.
**config.json은 만들지 않는다.** 공식 Melo `preprocess_text.py`가 7열 목록·BERT 특징·config를 만든다.
다만 공식 코드에는 `shuffle()`과 재분할이 있어 그대로 사용하면 140/30/30 고정 split이 깨진다.
전처리 후 ID 기준으로 고정 train/validation 목록을 다시 구성하고 test를 완전히 분리해야 한다.

`docs/templates/melo_paired_training_plan.json`에 batch 1, seed 52, FP16,
segment 8192, 단일 GPU, optimizer 1 step 스모크 제안을 기록했다.
이는 **실행용 trainer config가 아니라 검토용 설정안**이다. 세 조건은 같은 초기 가중치·전처리·대본·split·학습 step을 써야 한다.
Korean frontend/BERT·학습 의존성 구성, strict 초기화 호환성, 실제 trainer VRAM과 안정성은 아직 검증하지 않았다.
음성 복제 파인튜닝은 실행하지 않았고, SPEC의 1.15GiB를 학습 메모리로 사용하면 안 된다.

공식 [학습 안내](https://github.com/myshell-ai/MeloTTS/blob/209145371cff8fc3bd60d7be902ea69cbdb7965a/docs/training.md)도
metadata 준비와 전처리/config 생성, 학습을 별도 단계로 구분한다.

## 6. 강건성 및 평가 — 변환 연결 완료 / 방어 효과 평가 안 함

최초 GPU SPEC 1회·10회 쌍에 Gaussian 9조건, MP3 3조건, Resampling 3조건을 각각 2회 적용했다.
결과 `runs/desktop_20261118/robustness_results/robustness_batch_b_85of0rh6/`:
WAV 120개, MP3 24개. 모든 WAV/MP3 해시, 두 반복의 재현성, 입력 보존 확인 PASS.
Gaussian seed 조건과 두 실행 반복은 독립적인 사람 표본 수가 아니다.

SIM·CER 코드는 Melo 브랜치에서 확인했지만, 해당 모델 의존성과 파인튜닝 후 합성 결과가 없어 실행하지 않았다.
강건성 변환 뒤 STOI/SIM/CER 및 실제 방어 효과도 이번에는 측정하지 않았다.
평가 담당에게는 실제 합성 파일의 `reference,sentence_id,text,output` metadata.csv와
허가받은 화자의 기준 WAV, 고정 test 목록이 필요하다.
UI의 표시용 92%는 측정 성능이 아니며 발표 수치로 사용할 수 없다.

발표 표 준비 파일:
`runs/desktop_20261118/final_tables/summary.json`, `technical_measurements.csv`.
수학 신호라는 제한을 열과 JSON에 기록했고, SIM/CER/방어 성능 수치는 생성하지 않았다.
실제 사람 실험 결과가 확보되면 같은 조건별 CSV에서 발표용 품질·방어 성능 그래프를 생성한다.

## 7. 테스트·재실행·일정

- 기존 SPEC 테스트: 35 PASS.
- 초기 강건성 스냅샷 디렉터리 실행: 32개 중 24개 error (상대 기준 WAV가 해당 스냅샷에 없음).
  로그를 보존했으며 필요한 코드만 실제 통합 루트로 가져와 경로 문제를 해결했다.
- 코드 통합 후 58 PASS, 새 bridge의 4개 실패 방지 테스트를 포함한 최종 **62 PASS**.
  `runs/desktop_20261118/final_tests.log`.
- 패키지 설치 없음. 변경은 고정 소스·가중치 다운로드, 새 worktree의 코드/문서, 새 runs 산출물이다.
- 원래 worktree는 계속 clean이다. ignored 기존 runs에 대한 실행 전 전체 해시 목록은 없으므로,
  전체 기존 파일의 전후 해시 비교를 했다고 주장하지 않는다. 원래 runs에 쓰는 작업은 수행하지 않았다.

`scripts/Invoke-DesktopTechnicalValidation.ps1`는 WSL 환경 검사 → 새 수학 신호 → SPEC 1/10회 →
강건성 batch를 자동 실행한다. 원래 작업 공간에 이 파일이 없다는 점에 주의한다.
모델 자산이 이미 준비된 현재 PC 설정이 기본값이며, 사람 데이터나 학습은 실행하지 않는다.

마감까지의 우선순위 (계획이며 완료 예측이 아님):

1. **11/18~19:** 팀의 데이터 사용 범위 확인 기록·발화별 청취 검수 확보. Mac 산출물이 필요하면
   별도 파일 전송 필요. 권한이 확인되기 전에는 실제 사람 SPEC·학습을 시작하지 않는다.
2. **11/19~20:** 허가·검수된 소수 발화에서 GPU SPEC 1/10회, 이어 규모 확장.
   Korean 전처리와 고정 split/config 연결, 메모리 스모크 후 동일 조건 파인튜닝.
3. **11/21 핵심 결과 목표:** held-out 합성의 SIM/CER와 입력 품질 STOI, 강건성 조건 평가.
   학습/권한이 지연되면 기술 검증만 완료한 것으로 보고하고 실제 방어 성능은 미완료로 남긴다.
4. **11/22~23:** 결과 재검증·CSV/JSON 고정·PPT 표와 그래프·출처 및 제한 정리.
5. **11/24 PPT 마감:** 실제 측정 결과만 발표에 사용한다.
