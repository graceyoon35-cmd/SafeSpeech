# 11/8~11/10 평가 연결 항목 확인 — 2026-11-10

브랜치: `feature/robustness`. 기존 공유 `audio_metrics.load_audio/score_pair`를 사용해
한국어 변환 패키지의 STOI 로컬 실행을 확인했다. 평가 담당자 코드/브랜치는 수정하지 않았다.

| 항목 | 상태 |
|---|---|
| 변환 파일·설정·전달 ZIP 준비 | 완료 |
| 변환 후 STOI 로컬 실행 | SPEC 1/10회 각각 30행, 총 60행 PASS |
| 잘못된 파일·옵션 거부 테스트 | 기존 자동 테스트에 구현, 전체 29개 PASS 확인 완료 |
| 평가 담당자에게 파일 전달 | 미확인 |
| 평가 담당자의 환경에서 실행 및 최종 reference 정책 합의 | 미확인 |

이 문서 작성 시 추가 테스트 재실행은 하지 않았다. 29개 PASS는 직전 11/10 최종 실행
(18.133초, skip 0개)의 결과다. 잘못된 경로/동일 파일/동일 파형/다른 Hz·길이,
스테레오/무음/NaN/Inf/범위 밖 진폭/짧은 입력, 잘못된 변환 옵션, 중복 조건,
파일 해시 변조/위험 경로 등을 기존 테스트가 검사한다.

## 이번 점수의 비교 정책

- reference: 동일한 조건으로 변환한 원본(`transformed_original`).
- candidate: 동일한 조건으로 변환한 SPEC 보호본(`transformed_protected`).
- 각 쌍의 Hz·샘플 수·패키지 체크섬을 확인하고 자동 리샘플링/잘라내기는 하지 않았다.
- `pystoi.stoi(..., extended=False)`를 공통 함수로 호출했다.
- 조건마다 같은 두 번 실행의 STOI가 정확히 일치했다.
- 실행 전후 입력 패키지 해시/체크섬 검증을 통과했다.

최종 reference 정책은 평가 담당자와 합의하지 않았다. 이 점수는 **로컬 실행 확인**이며
변환 전 clean을 기준으로 한 최종 명료도 점수가 아니다.
Gaussian은 동일 seed 잡음을 양쪽에 더하므로 상관 잡음에 의해 점수가 높게 나타날 수 있다.
STOI 상승을 보호 섭동 제거, 음질 개선 또는 복제 방어 효과로 해석하지 않는다.
합성 한국어 한 발화의 두 SPEC 조건이며 다화자·다발화 통계가 아니다.

## 전달 파일

1. 기존 `outputs/SafeSpeech_2B_한국어Melo_SPEC1_10_15조건_20261110.zip` (WAV/MP3/설정/출처).
2. 추가 `outputs/SafeSpeech_2B_한국어_평가연결_STOI_20261110.zip` (점수 CSV/실행 보고서/README).

추가 ZIP SHA-256: `d7e2d7e6d0d6e8227d799b5645b00b44d872d01b1fa12e9cf56117b2d3039afc`.
추가 ZIP은 기존 결과를 덮어쓰지 않고 별도로 작성했으며 CRC와 내용 일치를 확인했다.
CSV의 `source_reference/source_candidate` 경로는 첫 번째 ZIP 압축 해제 루트 기준이다.
원래 전달 패키지의 `evaluation_status=not_measured`는 당시 기록으로 그대로 보존했다.
이번 로컬 점수/비교 정책은 새 CSV와 보고서에서 확인한다.

로컬 실행 기록: `runs/korean_stoi_execution_b_gcl9xo1s/report.json`, `stoi_execution.csv`.
음성·실행 결과 ZIP은 Git 제외다. 이 문서는 로컬 추가 기록이며 실제 전달/회신을 대신하지 않는다.
