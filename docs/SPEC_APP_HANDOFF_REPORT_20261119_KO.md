> 11월 19일 시점의 기록입니다. ε 보정본 추가 학습·평가를 포함한 최신 결과는 [2026-11-20 최종 보고서](SPEC_FINAL_REPORT_20261120_KO.md)와 [앱팀 명세](README_SPEC_HANDOFF_20261120_KO.md)를 확인하세요.

# 앱팀 인계 완료 보고서 — 2026-11-19

SPEC 연구 코드·오프라인 보호 실행기·문서만 인계한다. UI·서버·프론트엔드는 변경하지 않았다.

실행 명세와 설치·입출력 예제는 [README](README_SPEC_HANDOFF_20261119_KO.md), 연구 검증·추가 seed·발표 표는 [최종 결과 보고서](SPEC_FINAL_REPORT_20261119_KO.md)에 있다.

## 인계 파일

- `scripts/protect_wav.py`: 실제 WAV → 엄격한 ε를 지키는 보호 PCM16 WAV, 원본 및 기존 출력 덮어쓰기 거부.
- `scripts/example_spec_protect.sh`: WSL에서 검증한 호출 예제.
- `scripts/audit_kss_final.py`, `scripts/recompute_kss_sim.py`: 원본 실험·파형·CER·SIM 독립 검증.
- `scripts/project_kss_strict_inputs.py`: 기존 SPEC_10을 보존하는 ε 보정 사본 생성.
- `scripts/run_kss_seed_repeat.py`: Clean·SPEC_10의 추가 학습 seed 반복.
- `scripts/build_spec_final_report.py`, `docs/figures/spec_final_results_20261119.svg`: 실측 표·발표 그래프 재생성.
- `scripts/test_final_spec_validation.py`: CER 100% 초과, PCM16 저장 경계 및 입력 거부 검사.

## 실제 검증

최종 예제 GPU 실행 PASS: 입력 `/mnt/c/Users/윤채영/SafeSpeech-desktop-integration-20261118/runs/desktop_20261118/data/data_set/kss_part_200/train/wavs/kss_hf_00078.wav` → `runs/final_validation_20261119/final_cli_smoke/protected.wav`.

44,100Hz mono PCM16, 141,123샘플, 최대 진폭 차이 0.0313720703125, STOI 0.8874569446, SNR 11.639769dB. 출력 SHA256 `40e1c0f12e98b706fecd24c84f03009d98ab4f1cbdbf4d7578fdcb05e4743d86`.

동일 출력 폴더 재사용은 예상대로 종료 코드 1에 거부됐으며 기존 파일 해시는 전부 유지됐다. 전체 자동 테스트 99개 PASS, pip check PASS. 엄격한 ε 보정 사본 200개는 버전 관리할 스크립트로 재생성해 동일 WAV 해시를 확인했다.

## 실패·제한과 앱팀에 남는 작업

기본 GPU 실행의 동일 seed 바이트 재현 검사는 FAIL이다. 결정론 강제 옵션도 `reflection_pad1d_backward_out_cuda` 지원 부재로 FAIL이다. 실패를 숨기거나 성공 처리하지 않았으며 `handoff_repeat_mismatch.json`, `deterministic_repeat.json` 및 관련 로그에 보존했다. 기본 GPU 모드에서 같은 입력이 같은 해시를 낸다고 앱에 표시하지 않는다. ε 제한과 출력 형식 검사는 별도로 통과했다.

엄격한 ε 보정 입력으로의 모델 재학습·SIM/CER는 미실행이다. 추가 seed53은 원래 연구 조건과 같은 보정 전 WAV를 사용한다. KSS는 단일 화자이며 사람 청취 검수·별도 화자 동의·외부 음성/가중치 공개 승인으로 간주하지 않는다.

앱 UI 연결·비동기 호출·오류 표시·배포는 앱팀 담당이다. 이번 작업에서 UI·서버·프론트엔드 코드에 손대지 않았다. 가중치·데이터·백업 ZIP은 Git에 포함하지 않으며, 필요한 로컬 모델 경로는 README에 기록했다.

## 저장·인계 상태

새 증거: `runs/final_validation_20261119/`. 새 seed 체크포인트: `/home/safespeech/experiments/SafeSpeech-KSS-final-20261119/`. 기존 증거 2140개 파일의 종료 시점 해시는 모두 유지됐다. 기존 백업 파일은 이 작업에서 쓰기 대상으로 사용하지 않았다.

코드·문서 Git 인계 대상 브랜치: `research/spec-final-validation-20261119`. commit·push 결과는 별도 `git_release.json`과 최종 사용자 보고로 확인한다. master 및 팀원 브랜치는 수정·병합하지 않는다.
