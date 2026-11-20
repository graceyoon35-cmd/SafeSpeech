# SPEC 최종 인계 상태 — 2026-11-20

생성·학습·방어 평가 및 오프라인 WAV 실행기 인계 준비를 완료했다. 엄격한 ε 보정본의 별도 1,000-step 실험과 독립 검산 결과는 [최종 보고서](SPEC_FINAL_REPORT_20261120_KO.md)에 있다. 11월 19일 문서는 당시 상태를 보존하는 기록이며 오늘 보고서가 최신이다.

- 실행 파일: `scripts/protect_wav.py`, `scripts/example_spec_protect.sh`.
- 앱팀 계약: [README](README_SPEC_HANDOFF_20261120_KO.md), `requirements-spec-app-wsl.txt`, `docs/spec_app_environment_20261120.txt`.
- 로컬 실제 예제: `runs/final_validation_20261120/app_handoff_examples/`의 input.wav, protected.wav, result.json.
- 연구 코드: `scripts/run_strict_kss_validation.py`, `scripts/build_spec_release_report.py`, 기존 audit·SIM 재검산·ε 투영·seed 반복 스크립트.
- 표·그림: `docs/spec_final_results_20261120.csv`, `docs/spec_input_quality_20261120.csv`, `docs/figures/spec_release_results_20261120.svg`; PNG는 로컬 presentation 폴더.
- 오늘 검증: 자동 테스트 99 PASS, pip check PASS, 실제 WAV GPU 실행·엄격한 ε·SNR 독립 재계산·입력 및 기존 출력 보존 PASS.
- 증거: `runs/final_validation_20261120/final_verification.json`, `wav_smoke_checks.json`, 각 SIM 재인코딩 summary 및 실행 로그.

Git 인계 브랜치는 `research/spec-final-validation-20261119`이며 코드·문서만 명시적으로 선택한다. master·팀원 브랜치를 수정·병합하지 않는다. 커밋 SHA·원격 일치 확인은 로컬 `runs/final_validation_20261120/git_release.json`과 최종 사용자 보고에 기록하며 새 보완 백업에도 포함한다.

기존 29GB ZIP은 보존하고 SHA256을 재검산한다. `C:\Users\윤채영\SafeSpeech-backups`에 별도 보완 ZIP을 생성한다. 기존 ZIP 복원 후 보완 ZIP의 project/와 wsl-home/를 해당 위치에 덧붙이는 방식이다. 보완 ZIP은 오늘·전날 검증 결과, seed53·보정본 체크포인트, 해당 평가, 최신 코드·문서를 포함한다. ZIP 내부 BACKUP_MANIFEST.json과 외부 .sha256·.verification.json으로 전체 항목의 SHA256 및 CRC를 확인한다. 실제 새 ZIP 경로·검증 결과는 `runs/final_validation_20261120/backup_release.json`과 최종 사용자 보고에 기록한다.

남은 연구 문제는 GPU 바이트 비결정성, 보정본 추가 seed·독립 섭동 seed·다화자·장기 학습 검증, 사람 청취 검수다. 앱 UI·비동기 실행·배포 연결은 앱팀 담당이다. 결과를 무손실·불가청·복제 차단 성공률로 발표하지 않는다.
