# SPEC 앱 연동 계약과 현재 반영 상태

2026-11-26 문서 통합. 앱팀 브랜치나 인터페이스 구현을 변경하지 않았다. 원래11/20 계약은 아래에 보존하고11/24 시연 프로필 차이만 명시한다.

## 현재 앱팀에 전달할 사항

| 항목 | 적용할 계약 |
|---|---|
| backend | 별도 `SafeSpeech-desktop-integration-20261118`의 검증된 브리지/CLI |
| 프로필 | `scripts/spec_demo_profile.py:create_client`로 ε2/255·10회·seed52 명시. backend 기본 ε8과 구분 |
| 입력 | WAV/mono/44.1kHz/PCM16/1~10초. MP3/MP4 지원·자동 변환 완료로 표시하지 않음 |
| 성공 | 종료 코드·PASS JSON·저장 JSON·요청 설정·실제 WAV 해시/형식/길이/엄격 ε 확인 |
| 재생·저장 | 요청별 새 UUID 작업, 검증된 `result.output_path`만 사용. 이전 재생 상태 초기화 |
| 신뢰도 | `None`은 미측정. 더미92%·차단 확률로 변환하지 않음 |
| 작업 수명 | 이벤트 루프를 막지 않는 작업 스레드/별도 프로세스. 처리 중 강제 종료·재생 중 캐시 제거 금지 |
| 제외 연구 | silence_only_smooth·지각 가중·마스킹 후보의 앱 적용은 미완료 |

전달 대상은 `patches/spec_demo_ui_20261124.patch`, `scripts/spec_demo_profile.py`, `spec_demo_app.py`, `launch_spec_demo.py`, `config/spec_demo.template.json`이다. 로컬 `runs/demo_noise_fix_20261124/app_snapshot/ui`는 시연 사본이다. 앱팀의 feature/ui에 반영/병합된 것으로 표시하지 않는다. 인계 기준 커밋은d678c2051e336b2be2a595cc37562030c7df1954이며 현재 원격을 다시 조회하지 않았다.

11/20에는 Windows→WSL 실제 GPU 호출·파일 선택/저장 계약과 Qt 이벤트 루프를 확인했다.11/24에는 WSLg 창 기동과 offscreen 실제 앱 흐름2회를 기록했다. **네이티브 Windows 전체 GUI·물리적 스피커 청취·다른 PC 설치·패키징은 미검증**이다. 이 문서 작업에서 추가 GUI 테스트를 실행하지 않았다.

실행 명령·고정 시연 WAV·출력 캐시·청취 평가 재개는 [인계](README_SPEC_HANDOFF_20261120_KO.md), 지표와 한계는 [최종 보고서](SPEC_FINAL_REPORT_20261120_KO.md)를 따른다. 코드 생성기/테스트가 참조하는 기존 공식 파일명은 유지했다.11/20 원문 예제의 기본 ε8을 최신 시연 설정으로 복사하지 않는다.

## 보존한 최초 연결 계약

아래 원문의 파일·명령은 BackendRoot 기준이다. 당시 더미 UI 분석과 초기 ε8 브리지 예제이며 위 최신 프로필의 적용 완료 증거가 아니다.

<details>
<summary>11/20 앱 연동 원문 — 신호·처리기·Windows 경로·오류/저장 계약·테스트 범위</summary>

# 앱팀 실제 코드와 SPEC 연동 점검 — 2026-11-20

**판정: 현재 앱은 SPEC에 연결되지 않았다. SPEC 측 Windows→WSL 브리지와 Qt 작업 처리기는 준비·검증됐으며, 앱팀이 아래 연결·입력 정책·표시를 수정하면 제한된 WAV 범위로 연동할 수 있다. WAV·MP3·MP4 전체 지원이 완료된 상태는 아니다.**

## 확인한 원격 소스

`git ls-remote --heads origin`과 `git fetch origin`으로 실제 원격을 확인했다. 앱 브랜치는 [feature/ui](https://github.com/graceyoon35-cmd/SafeSpeech/tree/feature/ui), 최신 커밋은 `d678c2051e336b2be2a595cc37562030c7df1954`(2026-11-13 17:45:05 KST, Rename desktop UI directory to ui)다. 앱 관련 변경 20개 파일을 커밋 고정 ZIP으로 새 검증 폴더에 추출해 읽었다. `git fetch`로 남아 있는 과거 로컬 원격 참조를 실제 현재 브랜치로 잘못 세지 않았다. 점검 기준은 현재 원격 목록에 실제 존재하는 브랜치다.

앱팀 브랜치를 체크아웃·수정·병합하지 않았다. 개선 파일은 `research/spec-final-validation-20261119`에서만 추가했다. 기존 실험·체크포인트·백업은 쓰기 대상으로 사용하지 않았고 새 모델 학습은 하지 않았다.

## 기술 스택·파일 입력·API

| 항목 | 실제 구현 |
|---|---|
| 실행 형태 | Windows 로컬 Python 3.10+ 데스크톱 앱 |
| 프레임워크 | PySide6 `>=6.8,<7`, Qt Widgets·QtCore·QtMultimedia |
| 웹/API 서버 | HTTP·REST·FastAPI·Flask·웹 업로드 구현 없음 |
| 파일 선택 | `QFileDialog.getOpenFileNames` 또는 로컬 URL drag/drop |
| 파일 큐 | `FileSelection`, 최대 10개, WAV·MP3·MP4, 경로 중복 제거 |
| 사전 검사 | 확장자·파일 읽기 가능 여부. WAV 채널·샘플레이트·길이 검사 없음 |
| 보호 처리 | `DummyProtectionProcessor(QObject)` + `QTimer`; 파일당 약 1.5초 더미. 모델 호출·보호 파일 생성 없음 |
| 호출 API | `start(files) -> bool`, `running`, `results -> tuple`, `failures[index] -> str` |
| 신호 | `running_changed(bool)`, `file_started(int,int)`, `file_finished(int)`, `file_failed(int,str)`, `progress_changed(int)`, `finished()` |
| 결과 | `ProtectionResult(original_path, elapsed_seconds, confidence=92, output_path=None)` |
| 재생 | `QMediaPlayer`/`QAudioOutput`; 실제 `output_path`가 있어야 보호 재생 가능 |
| 저장 | `result_saver.py`: 실제 출력 복사, 임시 파일 후 원자 교체. 원본과 같은 확장자만 허용 |

근거: [앱 README](https://github.com/graceyoon35-cmd/SafeSpeech/blob/d678c2051e336b2be2a595cc37562030c7df1954/ui/README.md), [처리기·결과 타입](https://github.com/graceyoon35-cmd/SafeSpeech/blob/d678c2051e336b2be2a595cc37562030c7df1954/ui/protection_processor.py), [파일 선택](https://github.com/graceyoon35-cmd/SafeSpeech/blob/d678c2051e336b2be2a595cc37562030c7df1954/ui/file_selection.py), [저장](https://github.com/graceyoon35-cmd/SafeSpeech/blob/d678c2051e336b2be2a595cc37562030c7df1954/ui/result_saver.py).

## 기존 SPEC와의 차이 및 이번 개선

| 비교 항목 | 기존 차이 | SPEC 쪽 제공 상태 |
|---|---|---|
| 런타임 | UI는 Windows/PySide6, SPEC는 WSL Python/CUDA | `SpecClient`가 `wsl.exe --exec`로 호출. Windows에 torch 설치 불필요 |
| 경로 | SPEC JSON은 `/mnt/c/...` 반환 | 검증한 파일의 Windows `Path`를 `output_path`로 반환 |
| 비동기 | 더미 타이머를 GPU 실행으로 바꾸면 GUI 정지 위험 | `SpecProtectionProcessor`가 작업 스레드에서 순차 호출, 기존 신호·속성 유지 |
| 출력 폴더 | 기존 SPEC는 존재하지 않는 폴더 필요 | 지정 cache root 아래 UUID별 `spec-.../result/` 생성, 재사용·덮어쓰기 방지 |
| 형식 | UI는 WAV/MP3/MP4, SPEC는 mono 44.1kHz PCM16 WAV 1~10초 | WAV 사전 검사 및 명시적 실패. 자동 변환·분할 없음 |
| 성공 판정 | 더미는 파일 읽기만으로 완료 | 종료 코드 0, 마지막 JSON, 저장 JSON, 실제 WAV 경로·해시·형식·길이·ε 확인 |
| 신뢰도 | UI의 92%는 고정 테스트값 | 결과 `confidence=None`. 확률이나 차단율을 만들지 않음 |
| 배치 실패 | 한 파일 실패 후 다음 파일 처리 필요 | 실패 index 유지, 다음 파일 계속 처리. 진행률은 완료 파일 수 기준 |

이번 파일: `scripts/spec_desktop_bridge.py`, `scripts/spec_desktop_processor.py`, 두 `test_spec_desktop_*.py`, `scripts/smoke_spec_desktop_contract.py`. 모델 알고리즘·기존 실행기·학습 코드는 변경하지 않았다.

## 앱팀이 적용할 정확한 연결

UI 브랜치가 SPEC 코드 전체를 갖고 있지 않으므로, 먼저 SPEC 브랜치를 별도 경로에 체크아웃하거나 배포 사본으로 준비한다. 기존 UI 폴더를 이 연구 폴더로 옮기거나 임의 병합할 필요는 없다. `SpecConfig.repository`에는 `scripts/protect_wav.py`가 있는 **SPEC 체크아웃**을 지정한다.

앱팀의 `ui/ui/main_page.py:8`, `:21`에 있는 더미 import·생성 부분을 다음 방식으로 연결한다. 아래는 인계 예시이며 이번 작업에서 UI 파일에 적용하지 않았다.

```python
import sys
from pathlib import Path

SPEC_ROOT = Path(r'C:\Users\윤채영\SafeSpeech-desktop-integration-20261118')
sys.path.insert(0, str(SPEC_ROOT / 'scripts'))
from spec_desktop_bridge import SpecClient, SpecConfig
from spec_desktop_processor import SpecProtectionProcessor

# MainPage.__init__ 내부, self.processor 생성 위치
client = SpecClient(SpecConfig(
    repository=SPEC_ROOT,
    output_root=Path.home() / 'VoiceProtect' / 'spec-results',
    distribution='Ubuntu-22.04',
    python_executable='/home/safespeech/.venvs/melo-training-20261118/bin/python',
    assets='/home/safespeech/models/melo-spec-pinned-20261118',
))
self.processor = SpecProtectionProcessor(client, self)
```

기존 `start(self.files)`와 여섯 신호 연결은 유지한다. `SelectedFile.path`를 직접 받는다. Qt 어댑터는 기존 결과 타입과 같은 속성을 제공하며, 실제 앱의 `result_saver.save_result`가 이 결과를 정상 저장하는 것을 확인했다. 보호 이름은 항상 `<원본 stem>_protected.wav`다.

반드시 앱팀에서 적용할 나머지 수정은 다음과 같다.

1. **초기 연결 범위를 WAV로 제한:** 선택 필터·드롭 검사·도움말을 맞추고 1~10초 mono 44.1kHz PCM16 조건을 안내한다. 잘못된 WAV는 브리지에서 실패 처리한다. MP3/MP4까지 지원하려면 별도 추출·다운믹스·리샘플·분할·출력 정책이 필요하다. 압축 포맷 재인코딩 후에도 같은 ε가 보장된다고 표시하면 안 된다.
2. **신뢰도 표시 수정:** `main_page.py:298`의 `f'{result.confidence}%'`를 `None`일 때 `미측정`으로 표시하도록 변경한다. 92%를 실제 모델 신뢰도나 방어 성공률로 표시하지 않는다.
3. **실제 연결 상태 문구 수정:** `on_file_started`, `on_processing_finished`, 설정·도움말의 더미/미연결/1.5초 안내를 실제 처리 상태로 바꾼다. 처리 시간은 실제 경과 시간을 사용한다.
4. **작업 종료·수명 관리:** 처리 중 앱 종료를 막거나 완료 후 종료하게 한다. 이번 어댑터에는 강제 취소·WSL CUDA 자식 프로세스 정리 기능이 없다. `QThread.terminate()`로 취소하지 않는다. 재생·저장 전에 결과 cache를 삭제하지 않는다.
5. **다른 PC 배포:** 실제 WSL 배포 이름·Linux Python·고정 자산 경로를 설정한다. 로컬 드라이브 기본 `/mnt/<drive>` 매핑만 지원하며 UNC·네트워크 공유·커스텀 mount는 미지원이다.

WAV 입력은 기존 저장기의 원본 확장자 강제와 호환된다. 추후 MP3/MP4를 받아 WAV를 반환할 경우 `result_saver.py:17~22` 및 `main_page.py:318`의 저장 확장자·필터를 실제 출력에 맞게 바꿔야 한다. WAV 바이트를 `.mp3`·`.mp4` 이름으로 복사하면 안 된다.

## 독립 실행 명령과 반환 형식

Windows PowerShell에서 UI와 별개로 확인할 수 있다. Windows Python에는 표준 라이브러리만 필요하다. Qt 처리기는 앱팀의 기존 PySide6 환경에서 import한다.

```powershell
cd C:\Users\윤채영\SafeSpeech-desktop-integration-20261118
python scripts/spec_desktop_bridge.py `
  --input 'C:\Users\윤채영\SafeSpeech-desktop-integration-20261118\runs\final_validation_20261120\app_handoff_examples\input.wav' `
  --output-root "$env:LOCALAPPDATA\VoiceProtect\spec-results"
```

성공은 종료 코드 0과 stdout JSON `status=PASS`다. `output_path`는 Windows에서 존재하는 보호 WAV 절대 경로이며 `metadata_path`는 실제 SPEC `result.json`, `diagnostics_directory`는 요청·stdout·stderr·결과를 보관하는 새 작업 폴더다. `confidence`는 null이다. 실행 실패는 종료 코드 1의 FAIL JSON, CLI 인자 문법 오류는 argparse 종료 코드 2다. 잘못된 입력은 작업 폴더 생성 전에 거부될 수 있다.

실제 하위 명령은 `wsl.exe --distribution Ubuntu-22.04 --exec <Linux Python> <SPEC script> --input <WSL input> --output-dir <새 WSL directory> --assets <고정 자산> --device cuda --iterations 10 --epsilon 8.0 --seed 52`다. 쉘 문자열·bash 보간 없이 인자 배열로 호출한다. 한국어·공백·작은따옴표가 있는 Windows 경로로 실측 검증했다. 기존 [SPEC 환경·자산 명세](README_SPEC_HANDOFF_20261120_KO.md)를 따른다.

## 실제 검증과 한계

- Windows Python 3.15.0에서 표준 라이브러리 브리지 테스트 13개 PASS.
- Windows→WSL→실제 GPU SPEC 두 호출 PASS. 두 번째 호출은 한글·공백·작은따옴표 입력과 공백·한글 출력 root를 사용했다. 각 호출은 약 12초였으며, 일반적인 성능 보장 수치는 아니다.
- 별도 Python 3.10/PySide6 6.8.3 테스트 환경에서 전체 회귀 113개 PASS. 기존 연구 venv에 패키지를 설치하지 않았다. 처음 Qt 환경에는 NumPy가 없어 수집 오류가 났으며 기존 연구 site-packages를 읽기 전용으로 참조해 해결했다. 최초 실패 로그도 보존했다.
- 앱팀 커밋의 실제 `FileSelection`·`result_saver` + 신규 Qt 처리기 + 실제 GPU 보호를 연결했다. 유효한 MP3를 선택한 뒤 명시적 미지원 실패, 다음 WAV 보호 성공, 결과 복사 및 해시 일치, 입력 보존, 이벤트 루프 응답을 확인했다.
- Qt 테스트는 QCoreApplication을 사용했다. **Windows 전체 GUI 화면·오디오 장치 재생·다른 PC 신규 설치·패키징은 검증하지 않았다.** GUI를 실행해 완전 연동을 마쳤다고 보고하지 않는다.
- GPU 동일 seed의 바이트 결정론 한계와 기존 연구 결과는 그대로다. 신뢰도·차단율을 새로 평가하지 않았고 학습하지 않았다.

증거 폴더: `runs/app_integration_review_20261120/`. 주요 파일은 `remote_heads.txt`, `ui_commit.txt`, `ui_source_d678c20.zip`, `windows_jobs/`, `windows_path_smoke/`, `qt_contract_smoke.json`, `full_tests_final.log`다. 기존 최종 실험·체크포인트·백업은 변경하지 않았다.

문서 축소 후에도 이 연결 계약은 유지한다. 배치 입력·자산·검산·checkpoint·백업 명세는 [실행 가이드](README_SPEC_HANDOFF_20261120_KO.md), 라이선스·연구 프로토콜·이전 실험 결과·한계는 [최종 보고서](SPEC_FINAL_REPORT_20261120_KO.md)에 통합했다.

</details>
