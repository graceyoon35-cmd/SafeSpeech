# 합성 한국어 Melo SPEC 1/10회 강건성 변환 결과 — 2026-11-10

브랜치: `feature/robustness`. 받은 합성 한국어 음성에서 기존 15조건 변환을 실행했다.
실제 사람의 녹음, 파인튜닝 데이터 또는 복제 방어 성능 결과가 아니다.

## 수령 자료 검증

- 수령 ZIP: `SafeSpeech_MeloTTS_SPEC_Handoff_20261110.zip`
- ZIP SHA-256: `2c95be085c2e75a6745cd692524d18d49373fa1b56eeac1042dd7a9c83463ed2`
- 포함된 체크섬 목록 9개 모두 일치, ZIP CRC 통과, 원본 ZIP 변경 없음.
- 입력: MeloTTS-Korean 합성 문장 "안녕하세요. 음성 보호를 연구합니다."
- 원본/보호본 모두 44,100Hz·모노·196,096 samples·약 4.4466초.
- 원본 FLOAT, 보호본 PCM_16. 전달 상태는 `NONHUMAN_NOT_FOR_FINETUNING`.
- 모델 파일은 수령하지 않았다. 실제 모델 실행·gradient는 전달 보고서의 기록이며
  이번 2B 검사는 WAV·보고서 해시/조건·저장 후 delta·STOI를 독립적으로 대조한 것이다.

| 입력 | SHA-256 | 변환 전 STOI 재계산 |
|---|---|---:|
| clean.wav | ca64cfae3ff5c82c8e6080535b593bf58afc95bf191f60c6d63ff1d914d7041f | 기준 |
| spec_1.wav | 502f79a3cd02bc2491b62dc99e168f49d07616cdc08ebff3b482fb69d4623844 | 0.9961603424775992 |
| spec_10.wav | 197ca74402dc425dcf614e6b6519ab677632067b8c1f297d0a161eb85065767b | 0.8757433629288035 |

위 STOI는 변환 전 원본/보호 비교이며 변환 후 점수가 아니다.
저장 후 최대 변화량은 1회 0.00316776438648958, 10회 0.0314030647277832로
epsilon 8/255 + PCM16 1 LSB + 1e-7 허용 범위를 통과했다.

## 실험 조건과 결과

Gaussian std 0.001/0.005/0.01 × seed 42/43/44 = 9조건,
MP3 32/64/128kbps = 3조건, 리샘플링 8000/12000/16000Hz = 3조건.
각 SPEC 조건은 원본/보호 입력에 같은 조건을 적용하고 2회 반복했다.
SPEC 생성 seed 52와 Gaussian 변환 seed 42/43/44는 서로 다른 설정이다.

| 보호 입력 | 변환 조건 | 반복 | 변환 WAV | MP3 | 평가 쌍 행 | 상태 |
|---|---:|---:|---:|---:|---:|---|
| SPEC 1회 | 15 | 2 | 60 | 12 | 30 | PASS |
| SPEC 10회 | 15 | 2 | 60 | 12 | 30 | PASS |
| 합계 | 각 15 | 각 2 | 120 | 24 | 60 | PASS |

- 두 조건 모두 입력 파일 해시 유지, 동일조건 반복 WAV/MP3 해시 일치.
- 출력 WAV 저장 후 파형 재읽기 일치. MP3는 입력 Hz/샘플 수 유지 여부를 검사한다.
- 리샘플링 출력은 목표 Hz 그대로 저장하며 평가용 자동 잘라내기/정렬은 하지 않는다.
- JSON/CSV에는 조건·역할·회차·전후 Hz/샘플 수·길이·클리핑 수·해시를 기록했다.
- 전달 ZIP CRC 및 파일 172개 체크섬 검증 통과. 체크섬 목록 파일 자체는 172개에서 제외한다.
- ZIP 압축 해제 후 두 하위 평가 패키지의 체크섬 검증도 통과했다.
- 120개는 변환 WAV 수다. 전달본에 함께 포함된 원본/보호 입력 복사본은 별도다.

합성 원본은 한 발화다. 1회/10회 보호본은 같은 발화의 서로 다른 최적화 조건이고,
2회 반복은 동일조건 재현성 검사이며 독립 발화 표본이 아니다.

## 실행 기록과 전달

기존 Windows CPU Python 3.10 환경과 ffmpeg를 사용했다. 모델/GPU 설치는 변경하지 않았다.
실행별 실제 버전은 각 패키지 `verification.json`, `artifacts.csv`와 로컬 report에 기록된다.

- 수령 검사: `runs/melo_handoff_check_wulzy_pj/receipt_validation.json`
- 전체 실행: `runs/korean_robustness_b_hqr40y1c/execution_summary.json`
- 조건별 manifest/summary/report: 위 실행 폴더의 `spec_1/`, `spec_10/`.
- 공유용 ZIP은 저장소 바깥 `outputs/SafeSpeech_2B_한국어Melo_SPEC1_10_15조건_20261110.zip`.
- 전달 ZIP SHA-256: `ccd72ab4c647ae9356fffbbe8a7d644ae187cebe29c857f2b21f0428c78250a8`.
- ZIP 안 `source/`는 수령 자료, `spec_1/`·`spec_10/`는 상대 경로 평가 목록과 결과다.
  로컬 절대 경로가 포함된 수령 검사 보고서는 공유 ZIP에 포함하지 않았다.
- ZIP은 준비 완료이며 평가 담당자에게 전송하거나 평가 성공 회신을 받은 상태는 아니다.

공개 결과 요약은 `experiments/results/robustness_b_melo_20261110.json`이다.
WAV·실행 폴더·모델·ZIP은 Git에 포함하지 않는다.
재실행은 기존 sweep example을 복사해 경로와 실험 ID를 지정한 뒤
`run_robustness_sweep_b.py <manifest> --repeats 2`,
`export_robustness_handoff_b.py <새 summary.json>` 순서로 한다.

## 검증과 남은 범위

전체 자동 테스트 29개 통과(18.133초, skip 0개). 기존 26개와 44.1kHz 추가 3개다.
일반 전달 README는 모델 출처를 단정하지 않도록 수정하고 이번 합성 출처를 별도로 명시했다.
입력 수령 이전 수학 신호 fixture 검사와 이번 실제 SPEC 산출물 변환을 구분한다.

변환 후 STOI/SIM/CER/MCD/WER, 청취, 파인튜닝/복제 공격 및 방어 효과는 미측정이다.
실제 사람 한국어 녹음·다발화/다화자 실험도 미완료다.
본 PASS는 변환·저장·보존·재현성·전달 무결성을 뜻한다.
