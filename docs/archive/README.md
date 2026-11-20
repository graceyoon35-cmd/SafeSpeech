# 보관 문서

최신 결론·실행·앱 연결은 [최종 보고서](../SPEC_FINAL_REPORT_20261120_KO.md), [실행 가이드](../README_SPEC_HANDOFF_20261120_KO.md), [앱 연동 점검](../SPEC_APP_INTEGRATION_REVIEW_20261120_KO.md)를 확인하세요.

이 폴더에는 중복·중간 보고서를 삭제하지 않고 보관했습니다. 본문 수치·당시 상태는 현재 상태와 다를 수 있습니다. 문서 이동에 따른 링크와 경로만 갱신했으며 역사적 실험 명령은 변경하지 않았습니다. 원문은 정리 전 커밋 `71077e8a2080f20fa6f0a2f2f7deea6d60ff3ace`에서도 확인할 수 있습니다.

| 이전 위치 | 보관 문서 | 보관 이유 |
|---|---|---|
| `docs/DESKTOP_100STEP_RESULTS_20261118_KO.md` | [DESKTOP_100STEP_RESULTS_20261118_KO.md](DESKTOP_100STEP_RESULTS_20261118_KO.md) | 100-step 중간 측정·발표 준비 |
| `docs/DESKTOP_INTEGRATION_20261118_KO.md` | [DESKTOP_INTEGRATION_20261118_KO.md](DESKTOP_INTEGRATION_20261118_KO.md) | 초기 데스크톱 기술 검증 |
| `docs/MELO_DESKTOP_TRAINING_20261118_KO.md` | [MELO_DESKTOP_TRAINING_20261118_KO.md](MELO_DESKTOP_TRAINING_20261118_KO.md) | 초기 3-step 합성 입력 학습·평가 |
| `docs/MELO_MAC_REAL_MODEL_20261105_KO.md` | [MELO_MAC_REAL_MODEL_20261105_KO.md](MELO_MAC_REAL_MODEL_20261105_KO.md) | Mac 모델 로딩·gradient 초기 검증 |
| `docs/MELO_SPEC_ITERATIONS_20261105_KO.md` | [MELO_SPEC_ITERATIONS_20261105_KO.md](MELO_SPEC_ITERATIONS_20261105_KO.md) | 초기 1/10회 비교 |
| `docs/MELO_SPEC_VALIDATION_20261105_KO.md` | [MELO_SPEC_VALIDATION_20261105_KO.md](MELO_SPEC_VALIDATION_20261105_KO.md) | Melo 산출물 계약·STOI 초기 검증 |
| `docs/README_SPEC_HANDOFF_20261119_KO.md` | [README_SPEC_HANDOFF_20261119_KO.md](README_SPEC_HANDOFF_20261119_KO.md) | 최신 11월 20일 실행 가이드로 대체 |
| `docs/SPEC_APP_HANDOFF_REPORT_20261119_KO.md` | [SPEC_APP_HANDOFF_REPORT_20261119_KO.md](SPEC_APP_HANDOFF_REPORT_20261119_KO.md) | 최신 앱 실행·연동 문서로 대체 |
| `docs/SPEC_FINAL_REPORT_20261119_KO.md` | [SPEC_FINAL_REPORT_20261119_KO.md](SPEC_FINAL_REPORT_20261119_KO.md) | ε 보정본 평가 전의 결과 기록 |
| `docs/SPEC_RESUME_VALIDATION_20261030_KO.md` | [SPEC_RESUME_VALIDATION_20261030_KO.md](SPEC_RESUME_VALIDATION_20261030_KO.md) | 10월 재개·환경 검증 이력 |
| `docs/SPEC_STOI_HANDOFF_KO.md` | [SPEC_STOI_HANDOFF_KO.md](SPEC_STOI_HANDOFF_KO.md) | 초기 BERT-VITS2 STOI 연결 이력 |
| `docs/SPEC_STOI_RESULTS_KO.md` | [SPEC_STOI_RESULTS_KO.md](SPEC_STOI_RESULTS_KO.md) | 영어 단일 발화 BERT-VITS2 STOI 결과 |
| `docs/SPEC_VALIDATION_20261030_KO.md` | [SPEC_VALIDATION_20261030_KO.md](SPEC_VALIDATION_20261030_KO.md) | 초기 CUDA·SPEC 검증 이력 |
| `RESUME_20261119_KO.md` | [RESUME_20261119_KO.md](RESUME_20261119_KO.md) | 당시 중단·재개 기록 |
| `docs/figures/spec_final_results_20261119.svg` | [spec_final_results_20261119.svg](figures/spec_final_results_20261119.svg) | ε 보정본 모델 점수가 없는 이전 그래프 |

## 유지한 실행·재현 자료

- 저장소 최상위 README와 최신 핵심 문서 3개.
- 배치 입력·manifest 가이드, KSS 연구 정책·사용 근거, 인계·Git·백업 상태 기록.
- 최신 환경 freeze·최종 CSV 2개·발표 SVG, 기존 requirements·templates·QuickStart.ipynb.
- 모델 카드·모델별 README·라이선스는 원래 위치에 유지.
- `docs/README_SPEC_HANDOFF_20261119_KO.md`에는 기존 생성 코드 참조를 위한 안내만 유지.

## 추후 삭제 검토 후보

이번 정리에서 영구 삭제한 파일은 없습니다. 즉시 삭제를 권장하는 문서는 없습니다. 최신 문서·백업·참조 보존을 확인하고 사용자가 승인한 뒤에만 아래 중복 본문을 검토할 수 있습니다.

- `README_SPEC_HANDOFF_20261119_KO.md`: 최신 실행 가이드가 대체. 최신 문서의 보관 링크도 먼저 조정해야 함.
- `SPEC_APP_HANDOFF_REPORT_20261119_KO.md`: 최신 실행·연동 문서가 대체.
- `SPEC_FINAL_REPORT_20261119_KO.md`: 최신 최종 보고서가 대체. 당시 측정 기록 보존 여부 확인 필요.
- `RESUME_20261119_KO.md`: 중단 상태 기록. 체크포인트·재개 근거를 다른 기록으로 확보했는지 확인 필요.

나머지 초기 검증 기록은 서로 다른 모델·환경·계약 또는 실패 이력을 담고 있으므로 단순 중복으로 삭제하지 않습니다. 코드가 참조하는 원래 경로의 안내 파일도 그대로 삭제하면 안 됩니다.

Git에서 관리하는 PDF·DOCX·PPTX·HTML 보고서는 없었습니다. 실험 폴더의 발표 자료·보고서·로그, 체크포인트, 데이터, 백업 ZIP은 정리 대상에서 제외했습니다. Git Commit·Push는 수행하지 않았습니다.
