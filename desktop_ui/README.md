# Voice Protect 데스크톱 UI

PySide6로 만든 Voice Protect 앱입니다. SafeSpeech 모델 코드와 독립적으로 실행하며, 현재 보호 처리는 파일당 약 1.5초의 더미 시뮬레이션입니다.

## 실행 (Windows, Python 3.10 이상)

저장소 루트에서 다음 명령을 실행합니다.

```powershell
cd desktop_ui
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py
```

이 UI만 실행할 때는 저장소 루트의 모델용 requirements.txt나 GPU 모델 다운로드가 필요하지 않습니다.

## 포함한 기능

- 시작 화면에서 메인 화면으로 전환, 실행 시 창 최대화 및 크기 대응
- WAV·MP3·MP4 다중 선택·드롭, 중복 방지, 최대 10개 제한
- 파일별 대기·처리 중·완료·실패 표시와 전체 진행률
- 처리 실패 시 다음 파일 계속 처리, 안내 메시지 및 콘솔 로그
- 결과별 원본명·보호 예정 파일명·처리 시간·테스트 신뢰도 92% 표시
- 원본/실제 보호 결과의 오디오 재생·일시정지 및 상호 정지
- 결과 저장 위치 선택, 실제 결과가 있으면 복사, 없으면 경로만 확인

파일 목록은 세 행을 표시하며, 파일이 많으면 마우스 휠로 이동합니다. 처리 완료 후 목록의 파일명을 클릭하면 해당 결과를 볼 수 있습니다. 보호 결과 파일이 없으면 보호 재생 버튼은 비활성화됩니다. MP4는 오디오 트랙만 재생합니다.

## 코드 구성

| 파일 | 역할 |
| --- | --- |
| main.py | 앱 초기화, 글꼴 등록, QStackedWidget |
| ui/start_page.py | 시작 화면 및 공통 UI·크기 대응 |
| ui/main_page.py | 메인 화면과 기능 연결 |
| styles.qss | 스타일 |
| file_selection.py | 파일 검증·중복·개수 제한 |
| protection_processor.py | 순차 더미 처리 및 결과 데이터 |
| result_saver.py | 저장 경로 검증과 결과 복사 |
| media_playback.py | QtMultimedia 오디오 재생 |
| error_handling.py | 사용자 오류 안내와 콘솔 진단 |
| assets/ | UI에서 사용하는 SVG와 나눔스퀘어 글꼴 |

## AI 모델 연결

`DummyProtectionProcessor`의 `start`, `running`, `results`, `failures`와 기존 신호 인터페이스를 유지하며 실제 처리기로 교체할 수 있습니다. 추론은 작업 스레드나 별도 프로세스에서 실행해야 합니다.

`ProtectionResult.output_path`에 실제 결과 파일 경로를 지정하면 보호 결과 재생과 저장에 연결됩니다. 현재 더미 처리는 원본을 수정하거나 보호 파일을 생성하지 않습니다.

나눔스퀘어 글꼴의 저작권 안내와 라이선스는 [assets/fonts/OFL.txt](assets/fonts/OFL.txt)에 포함되어 있습니다.
