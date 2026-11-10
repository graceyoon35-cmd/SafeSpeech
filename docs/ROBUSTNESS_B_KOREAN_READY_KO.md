# 한국어 Melo 입력 연결 준비 — 2026-11-10

작업 브랜치: `feature/robustness`. 다른 팀 브랜치의 코드와 모델은 변경하지 않는다.

이 문서는 수령 전 준비 기록이다. 이후 실제 자료 수령과 두 조건의 변환을 완료했다.
최신 상태는 [11/10 한국어 실행 결과](ROBUSTNESS_B_MELO_RESULTS_20261110_KO.md)를 따른다.

## 현재 상태

- 실제 영어 원본/SPEC 한 쌍의 15조건 × 2회 변환과 평가 전달 ZIP은 기존에 완료했다.
- 한국어 Melo 원본/보호 WAV는 아직 이 작업 폴더에 전달되지 않았다.
- 44,100Hz·모노·196,096 samples의 수학 신호로 추가 회귀 검사를 준비했다.
  이 길이는 담당자 문서의 합성 WAV와 같지만 내용은 사인파이며 한국어 발화가 아니다.
  보호 역할 입력도 수학적으로 만든 fixture이며 실제 SPEC 출력이 아니다.
- 이 검사는 파일 처리·조건 적용·반복 재현성·전달 무결성만 확인한다.
  한국어 음성 합성, STOI, 보호 효과, 실제 한국어 녹음 실험은 검증하지 않는다.

2026-11-10 Windows CPU 환경에서 기존 26개 + 신규 3개 = **29개 테스트 통과**,
실행 시간 18.729초, 건너뛴 테스트 0개. 실제 ffmpeg로 44.1kHz MP3 32/64/128kbps를
검사했다. 임시 테스트 출력은 검사 후 정리되며 실제 음성 결과 ZIP으로 배포하지 않는다.

## 추가 검사

`scripts/test_robustness_44100_b.py`에서 다음을 검사한다.

1. Gaussian std 0.001/0.005/0.01 × seed 42/43/44, MP3 32/64/128kbps,
   리샘플링 8000/12000/16000Hz의 15조건을 원본/보호 역할 입력에 동일 적용.
2. 2회 실행의 출력 해시 일치, WAV 60개·MP3 12개 및 입력 파일/설정 보존.
3. Gaussian·MP3의 44.1kHz/샘플 수 유지, 리샘플링의 목표 Hz/샘플 수/길이.
4. 전달 ZIP 이동·압축 해제 후 체크섬 검증과 평가 쌍 30행, 평가값 미측정 유지.
5. 스테레오와 1 sample 길이 차이 입력을 자동 모노 변환·잘라내기 없이 거부.

저장소 루트에서 현재 CPU 환경으로 실행한다. ffmpeg가 없으면 전체 MP3 검사 하나가
건너뛰어지므로 그 상태를 전체 경로 통과로 기록하면 안 된다.

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s scripts -p 'test_*.py' -v
```

## SPEC 담당자에게 받을 묶음

가능하면 11/10 실행 폴더에서 필요한 파일을 하위 폴더 구조 그대로 ZIP으로 받는다.

- 합성 한국어 원본 WAV와 동일 입력의 SPEC 1회·10회 보호 WAV.
- `handoff_manifest.json`, `run.json`, 입력 출처를 확인할 선언/보고서.
- 각 조건의 생성 `report.json`, 검증 JSON, STOI 결과 JSON.
- 발화 ID, 실제 파일 Hz·채널·샘플 수, seed·epsilon·반복 수, 사용 모델/코드 리비전과 해시.
- 합성 입력임을 명시. 개인 경로가 든 보고서는 필요한 범위에서 경로를 정리해 공유한다.

필수 실행 입력은 동일 Hz·동일 샘플 수의 모노 WAV 한 쌍이다.
실험 출처와 조건을 확인하려면 생성 보고서도 함께 받아야 한다.
1회/10회는 서로 다른 실험으로 실행하며 하나의 보호 조건으로 섞지 않는다.
모델 가중치, BERT 파일, CUDA 환경은 강건성 변환 실행에 필요하지 않다.

## 데이터 담당자에게 확인할 내용

실제 한국어 녹음 준비 시 녹음자 가명 ID·발화 ID, 검수 대본, 고정 train/validation/test
분할(사용할 split은 팀에서 확정), 사용 허가 상태/범위를 함께 전달받는다.
동의서 원문을 강건성 코드 저장소에 넣지 않는다. 실제 녹음 확보와 허가 판단은
데이터 담당자가 진행하며, 2B는 전달받은 파일의 조건·출처 기록과 실험 연결을 담당한다.

## 평가 담당자에게 확인할 내용

- 이전에 전달한 영어 3종 결과가 읽히고 STOI 계산까지 가능한지.
- 평가 reference를 변환 전 원본으로 할지, 동일조건으로 변환한 원본으로 할지.
- 목표 Hz가 달라지는 리샘플링 조건의 명시적 정렬 방법과 점수/오류 결과 형식.
- 명료도 평가와 모델 파인튜닝/복제 공격 후 평가를 별개로 기록할지 확인.

## WAV 수령 뒤 실행

1. 새 비공개 `runs/` 폴더에 압축을 풀고 파일 해시·출처·모노·Hz·샘플 수를 확인한다.
   원본은 보존하고 샘플레이트/길이가 다르면 담당자에게 확인한다.
2. `experiments/robustness_b_pair.example.json`을 별도 `*.local.json`으로 복사하여
   원본/보호 경로와 실험 ID를 입력한다. 3종 변환부터 실행한다.
3. `experiments/robustness_b_sweep.example.json`도 별도 복사해 동일 쌍을 지정하고
   15조건 2회 실행 후 전달 ZIP을 만든다. 두 manifest는 각각 자기 파일 위치 기준 경로다.

```powershell
.\.venv\Scripts\python.exe scripts/run_robustness_b.py experiments/robustness_b_korean_pair.local.json
.\.venv\Scripts\python.exe scripts/run_robustness_sweep_b.py experiments/robustness_b_korean_sweep.local.json --repeats 2
.\.venv\Scripts\python.exe scripts/export_robustness_handoff_b.py runs/<새_실험폴더>/summary.json
```

위 local JSON 두 개와 `<새_실험폴더>`는 수령 후 실제 값으로 만든다. 지금은 실행용 파일이 아니다.
한국어 합성 입력의 결과에는 `model_synthesized` 출처를 별도 보고서로 명시한다.
현재 강건성 manifest는 경로/조건만 읽고 입력의 실제 출처나 SPEC 생성 여부를 인증하지 않는다.
자동 생성 전달 README는 입력 출처를 인증하지 않으므로 생성 보고서를 함께 전달한다.
원본/생성 보고서의 로컬 절대 경로는 이동 후 경로에 맞게 확인하고 WAV 해시로 대조한다.

음성·모델·실행 결과 ZIP은 Git Push 대상이 아니다. 공유 대상은 코드·테스트·문서이며,
실제 음성 결과는 팀에서 합의한 Drive 경로로 전달한다.
