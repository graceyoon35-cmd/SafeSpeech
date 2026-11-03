# QUICKSTART·README 보완 제안

작성: 2026-11-03 19:05 KST · 작성: 1학년 A(데이터·환경·재현성) · 상태: 제안(초안)

대상 파일은 `feature/metrics` 브랜치(`27ae06d`) 것이라 **1학년 A가 직접 고치지 않고, 팀장에게 제안으로 전달**합니다. 줄 번호는 위 커밋 기준입니다.

## 1. 한눈에 보기

| 번호 | 파일 | 줄 | 문제 | 우선순위 |
|---|---|---|---|---|
| 1 | `README.md` | 24 | `git clone --branch feature/cpu-experiment-pipeline`은 그 브랜치가 없어서 **실행하면 실패** | **높음** |
| 2 | `README.md` | 10~12 | 없어진 브랜치 이름 두 개(`cpu-experiment-pipeline`, `integration-setup`)와 존재하지 않는 `main` | 높음 |
| 3 | `docs/DEVELOPMENT_PROGRESS_KO.md` | 3, 180 (81은 이력) | 같은 옛 브랜치 이름. 180줄은 clone 명령이라 실패 | 높음 |
| 4 | `docs/QUICKSTART_KO.md` | 47, 88 | 팀원 이름이 든 개인 경로 | 중간 |
| 5 | `docs/QUICKSTART_KO.md` | 83 | 사용자 이름이 고정된 인터프리터 경로 | 중간 |
| 6 | `docs/QUICKSTART_KO.md` | 5 | Mac 안내가 한 줄뿐 | 낮음 |
| 7 | `docs/QUICKSTART_KO.md` | (없음) | 한국어 경로, 샘플레이트, 입출력 형식 안내가 없음 | 낮음 |

## 2. 항목별 제안

### 2-1. `README.md` 24줄: clone 명령 (우선순위 높음)

| | 내용 |
|---|---|
| 현재 | `git clone --branch feature/cpu-experiment-pipeline https://github.com/graceyoon35-cmd/SafeSpeech.git` |
| 제안 | `git clone --branch feature/metrics https://github.com/graceyoon35-cmd/SafeSpeech.git` |
| 이유 | 원격에 `feature/cpu-experiment-pipeline` 브랜치가 없습니다 (2026-11-03 확인). 새 팀원이 README 첫 명령에서 실패합니다 |

### 2-2. `README.md` 10~12줄: 공유 브랜치 안내

| | 내용 |
|---|---|
| 현재 | 현재 공유 브랜치는 `feature/cpu-experiment-pipeline`이며 `feature/integration-setup`의 WSL 작업을 포함합니다. **main** 병합 여부와 별개로 이 브랜치를 선택해야 아래 추가 기능을 사용할 수 있습니다. |
| 제안 | 현재 공유 브랜치는 `feature/metrics`이며 WSL 공통 환경 작업(`be448f2`)을 포함합니다. **master** 병합 여부와 별개로 이 브랜치를 선택해야 아래 추가 기능을 사용할 수 있습니다. |
| 이유 | 두 브랜치 이름이 모두 없어졌고(`feature/metrics`가 `be448f2`를 포함함은 확인), 이 저장소의 기본 브랜치는 `main`이 아니라 `master`입니다 |

### 2-3. `docs/DEVELOPMENT_PROGRESS_KO.md`

| 줄 | 현재 | 제안 |
|---|---|---|
| 3 | 공유 브랜치: `feature/cpu-experiment-pipeline` | 공유 브랜치: `feature/metrics` |
| 81 | 이력 표 (`6c38459`, `feature/cpu-experiment-pipeline`) | **이력이므로 유지**하고 "(현 `feature/metrics`)"만 덧붙임 |
| 180 | `git clone --branch feature/cpu-experiment-pipeline ...` | `git clone --branch feature/metrics ...` |

### 2-4. `docs/QUICKSTART_KO.md` 47, 88줄: 개인 경로

| | 내용 |
|---|---|
| 현재 | `cd /mnt/c/Users/<팀원 이름>/SafeSpeech` |
| 제안 | `cd <저장소 경로>` 와 함께 설명 한 줄: "WSL 홈에 clone했다면 `cd ~/SafeSpeech`, Windows 폴더에 clone했다면 `cd /mnt/c/Users/<Windows 사용자 이름>/…`" |
| 이유 | 공개 저장소에 팀원 이름이 노출되고, 다른 PC에는 없는 경로입니다 |

### 2-5. `docs/QUICKSTART_KO.md` 83줄: 인터프리터 경로

| | 내용 |
|---|---|
| 현재 | `/home/safespeech/.venvs/safespeech/bin/python` |
| 제안 | `$HOME/.venvs/safespeech/bin/python` (VS Code 선택 창에서는 `~/.venvs/safespeech/bin/python`) |
| 이유 | WSL 사용자 이름이 `safespeech`로 고정돼 있어, 다른 팀원 PC에서는 경로가 달라집니다 |

### 2-6. `docs/QUICKSTART_KO.md` 5줄: Mac

| | 내용 |
|---|---|
| 현재 | Mac CPU 담당자는 [CPU 실험 안내](EXPERIMENT_PIPELINE_KO.md)를 따릅니다. |
| 제안 | 위 문장 뒤에 "Windows WSL과 Mac의 설치·실행 순서는 `feature/data-preparation`의 `docs/data-preparation/실행_가이드_초안.md` 3, 4절을 참고합니다"를 추가 |
| 비고 | 가이드가 `feature/metrics`로 합쳐지기 전까지는 브랜치 이름과 함께 안내해야 링크가 열립니다 |

### 2-7. `docs/QUICKSTART_KO.md`: 추가하면 좋은 안내

실행 가이드의 다음 절을 가리키는 링크 한 줄씩을 추가합니다.

| 내용 | 가이드 절 |
|---|---|
| 언어·모델별 샘플레이트 (영어 24000, 한국어 MeloTTS 44100) | 2-1절 |
| 작업별 입출력 형식 | 2-2절 |
| 한국어(MeloTTS) 설치와 실행 | 7절 |
| 어떤 일에 어느 브랜치로 시작하는지 | 8-1절 |

## 3. 전달 방법

- 전달 대상: 팀장 (`feature/metrics`의 README·QUICKSTART를 고치는 사람)
- **가장 먼저 전달할 것: 2-1 (clone 명령 실패).** 단체방에는 한 줄이면 충분합니다: "README의 clone 명령이 없어진 브랜치(`feature/cpu-experiment-pipeline`)를 가리켜서 `feature/metrics`로 바꿔야 합니다."
- 1학년 A는 `feature/metrics`를 수정하지 않습니다.
