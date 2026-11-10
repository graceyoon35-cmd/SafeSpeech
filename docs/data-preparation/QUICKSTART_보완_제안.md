# QUICKSTART와 README 보완 제안

작성: 1학년 A (데이터, 환경, 재현성) / 작성 2026-11-03, 갱신 2026-11-10

대상 파일은 `feature/metrics` 브랜치(`27ae06d`)에 있습니다. 1학년 A가 직접 고치지 않고 팀장에게 제안으로 전달합니다. 줄 번호는 위 커밋 기준입니다.

## 1. 한눈에 보기

| 번호 | 파일 | 줄 | 문제 | 우선순위 |
|---|---|---|---|---|
| 1 | `README.md` | 24 | `git clone --branch feature/cpu-experiment-pipeline`은 그 브랜치가 없어서 실패함 | 높음 |
| 2 | `README.md` | 10부터 12 | 없어진 브랜치 이름 두 개와 존재하지 않는 `main` | 높음 |
| 3 | `docs/DEVELOPMENT_PROGRESS_KO.md` | 3, 180 | 같은 옛 브랜치 이름. 180줄은 clone 명령이라 실패함 | 높음 |
| 4 | `docs/QUICKSTART_KO.md` | 47, 88 | 팀원 이름이 든 개인 경로 | 중간 |
| 5 | `docs/QUICKSTART_KO.md` | 83 | 사용자 이름이 고정된 인터프리터 경로 | 중간 |
| 6 | `docs/QUICKSTART_KO.md` | 5 | Mac 안내가 한 줄뿐 | 낮음 |
| 7 | `docs/QUICKSTART_KO.md` | 없음 | 한국어 경로, 샘플레이트, 입출력 형식 안내가 없음 | 낮음 |
| 8 | (새로 발견) 브랜치 전체 | 없음 | 브랜치마다 가진 파일이 달라서 같은 명령이 다른 브랜치에서 실패함 (아래 3절) | 중간 |

## 2. 항목별 제안

### 2-1. README.md 24줄: clone 명령 (가장 먼저 전달)

- 현재: `git clone --branch feature/cpu-experiment-pipeline https://github.com/graceyoon35-cmd/SafeSpeech.git`
- 제안: `git clone --branch feature/metrics https://github.com/graceyoon35-cmd/SafeSpeech.git`
- 이유: 원격에 `feature/cpu-experiment-pipeline` 브랜치가 없습니다(11/3 확인). 새 팀원이 첫 명령에서 막힙니다.

### 2-2. README.md 10부터 12줄: 공유 브랜치 안내

- 현재: 현재 공유 브랜치는 `feature/cpu-experiment-pipeline`이며 `feature/integration-setup`의 WSL 작업을 포함합니다. main 병합 여부와 별개로 이 브랜치를 선택해야 합니다.
- 제안: 현재 공유 브랜치는 `feature/metrics`이며 WSL 공통 환경 작업(`be448f2`)을 포함합니다. master 병합 여부와 별개로 이 브랜치를 선택해야 합니다.
- 이유: 두 브랜치 이름이 모두 없어졌고, 이 저장소의 기본 브랜치는 `main`이 아니라 `master`입니다.

### 2-3. docs/DEVELOPMENT_PROGRESS_KO.md

- 3줄: 공유 브랜치를 `feature/metrics`로 바꿉니다.
- 81줄: 이력 표이므로 그대로 두고 "(현 `feature/metrics`)"만 덧붙입니다.
- 180줄: clone 명령의 브랜치를 `feature/metrics`로 바꿉니다.

### 2-4. docs/QUICKSTART_KO.md 47, 88줄: 개인 경로

- 현재: `cd /mnt/c/Users/<팀원 이름>/SafeSpeech`
- 제안: `cd <저장소 경로>` 로 바꾸고 설명 한 줄을 붙입니다. WSL 홈에 clone했다면 `cd ~/SafeSpeech`, Windows 폴더에 clone했다면 `cd /mnt/c/Users/<Windows 사용자 이름>/SafeSpeech` 입니다.
- 이유: 공개 저장소에 팀원 이름이 보이고, 다른 PC에는 없는 경로입니다.

### 2-5. docs/QUICKSTART_KO.md 83줄: 인터프리터 경로

- 현재: `/home/safespeech/.venvs/safespeech/bin/python`
- 제안: `$HOME/.venvs/safespeech/bin/python` (VS Code 선택 창에서는 `~/.venvs/safespeech/bin/python`)
- 이유: WSL 사용자 이름이 `safespeech`로 고정돼 있어 다른 PC에서는 경로가 다릅니다.

### 2-6. docs/QUICKSTART_KO.md 5줄: Mac

- 현재: Mac CPU 담당자는 CPU 실험 안내(`EXPERIMENT_PIPELINE_KO.md`)를 따릅니다.
- 제안: 위 문장 뒤에 "Windows WSL과 Mac의 설치와 실행 순서는 `feature/data-preparation`의 `docs/data-preparation/실행_가이드.md` 3절과 4절을 참고합니다"를 추가합니다.

### 2-7. docs/QUICKSTART_KO.md: 추가하면 좋은 링크

| 내용 | 가이드 위치 |
|---|---|
| 언어와 모델별 샘플레이트 | `데이터_명세.md` 2절 |
| 작업별 입력과 출력 | `데이터_명세.md` 6절 |
| 한국어 MeloTTS SPEC 실행 | `실행_가이드.md` 5절 |
| 어떤 일에 어느 브랜치로 시작하는지 | `실행_가이드.md` 1절 |

## 3. 새로 발견한 문제: 브랜치마다 파일이 다름 (2026-11-10 확인)

WSL에서 `feature/spec-perturbation`(`412c358`)을 새로 clone해 가이드의 CPU 평가 명령을 실행했더니 다음이 실패했습니다.

| 실패한 것 | 원인 |
|---|---|
| `pip install -r requirements-evaluation-smoke.txt` | 이 브랜치에는 그 파일이 없음. 같은 내용의 `requirements-spec-stoi.txt`가 있음 |
| `python scripts/smoke_evaluation.py` | 이 파일은 `feature/metrics`에만 있음 |
| `python scripts/run_experiment.py experiments/gaussian_smoke.json` | 이 파일과 `experiments/` 폴더도 `feature/metrics`에만 있음 |

`feature/metrics`에서는 같은 명령이 모두 통과했습니다(테스트 9개 통과). 반대로 `run_korean_melo_spec.py`와 테스트 35개는 `feature/spec-perturbation`에만 있습니다.

제안
- README에 "명령은 해당 브랜치에서만 동작한다"는 한 줄과 브랜치별 시작 표(`실행_가이드.md` 1절)를 링크합니다.
- 두 requirements 파일은 내용이 같으니 이름을 하나로 맞추는 것을 검토합니다.

## 4. 전달 방법

- 전달 대상: 팀장 (`feature/metrics`의 README와 QUICKSTART를 고치는 사람)
- 가장 먼저 전달할 것은 2-1입니다. 단체방에는 한 줄이면 충분합니다: "README의 clone 명령이 없어진 브랜치(`feature/cpu-experiment-pipeline`)를 가리켜서 `feature/metrics`로 바꿔야 합니다."
- 1학년 A는 `feature/metrics`를 수정하지 않습니다.
