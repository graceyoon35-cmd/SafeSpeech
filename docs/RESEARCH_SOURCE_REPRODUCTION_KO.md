# 연구 보존 소스와 재검증 방법

이 문서는 2026-11-26 소스 보존 및 향후 통합을 위한 실행 안내다. 공식 성능 보고서는 `SPEC_FINAL_REPORT_20261120_KO.md`, 알고리즘 설명은 `SPEC_ALGORITHM_KO.md`의 문서 브랜치 버전을 사용한다. 여기서 새 방어 성능을 주장하지 않는다.

## 소스 및 입력·출력

기존 `runs/`의 원본 Python 파일은 삭제·이동하지 않았다. 다음 복사본은 원본 계산 AST를 대조하고 경로·import·실행 가드만 정리했다. 특히 `analyse_stages.py`의 import 시 자동 분석을 `main()` 안으로 옮겼다. 읽는 CSV·JSON·manifest와 WAV는 Git 업로드 대상이 아니다.

|공식 소스 (`scripts/research_analysis_20261126/`)|기존 입력|새 출력 / 실행 범위|
|---|---|---|
|`analyze_existing_only.py`|`runs/segment_noise_balance_20261124/analysis/per_file.csv`, 140발화×6조건의 기존 WAV, 기존 생성 로그, 지각 목적 코드, backend 학습 소스|명시한 새 폴더에 진폭·구간·포화도·강한 발화 동일성 CSV, timing·입력 해시 JSON. CPU 읽기 분석만|
|`package_existing.py`|backend의 Clean dataset, KSS handoff manifest, strict ε8 quality CSV와 기존 ε2/10·ε1/5·지각가중치0.5 음성|새 폴더에 3발화×5조건의 **복사** WAV, 안내문 ZIP, 별도 비공개 매핑표·검증 JSON. 정규화·새 SPEC 생성 없음|
|`analyse_stages.py`|`runs/psychoacoustic_spec_refinement_20261126/stage_diagnosis/stage_measurements.csv`|새 폴더에 `stage_summary.json/csv`. 5발화×10업데이트의 동일 제안 50개 비교|
|`report_and_verify.py`|기존 pilot의 complete·quality·summary·threshold·WAV·블라인드 ZIP·당시 Git 상태|역사적 보고서 작성기 원형 보존. CLI는 입력 존재 확인만 지원. 현재 refs에서 옛 보고서를 재발행하지 않음|
|`report_results.py`|A/B의 preservation·단계별 complete·quality·summary·WAV·백업·당시 source/hash/refs|역사적 결과·음성 검증기 원형 보존. CLI는 입력 존재 확인만 지원. 당시 HEAD·해시 검사를 우회하지 않음|
|`verify_saved_pcm.py` (보완 도구)|pilot의 5개 원본, 고정 threshold/radius, 마스킹 10개·A/B 10개·진단 재생 5개 WAV|새 JSON에 원본 SHA·PCM16 ε257code·radius·clipping·2개 FFT 마스킹·FFmpeg 전체 디코딩 재검사. 기존 보고서를 바꾸지 않음|

`--check-inputs`는 필수 파일의 **존재만** 확인한다. 실제 음성·설정·해시 검증을 모두 통과했다는 의미가 아니다. 분석 실행과 저장 PCM 검증은 별도 명령이다. 두 역사적 보고서 `main()` 함수의 계산·보존 검사는 그대로 남아 있다. 당시 소스 변경과 현재 refs 때문에 현재 HEAD에서 완전 재실행을 보장하지 않는다. 원본 결과·실행환경·원래 소스는 로컬 추가 백업으로 보존한다.

## 기존 환경에서 실행

WSL/Linux의 Python 3.10 이상을 사용한다. 현재 검증 환경은 기존 MeloTTS venv이며 추가 설치하지 않았다. NumPy·SciPy·soundfile·pystoi·Torch와 FFmpeg가 필요하다. 세 분석/복사 실행에는 모델 가중치나 GPU가 필요 없다. 마스킹 수학 모듈이 Torch를 import하지만 아래 PCM 검증은 CPU에서 실행한다. Python 패키지의 정확한 설치 목록은 비공개 백업의 환경 freeze와 기존 backend의 requirements를 참고한다. 새 환경 설치까지 검증한 것은 아니다.

저장소 루트에서 다음처럼 명시한다. `WORKSPACE`는 비공개 실험 자료가 존재하는 루트, `BACKEND`는 기존 backend checkout이다. `OUT`은 아직 존재하지 않고 역사적 실험 폴더 밖인 출력 경로여야 한다.

```bash
WORKSPACE=/path/to/existing-research-workspace
BACKEND=/path/to/existing-backend-checkout
OUT=/path/to/new-output
PY=/path/to/existing-venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

$PY -B scripts/research_analysis_20261126/analyze_existing_only.py \
  --workspace "$WORKSPACE" --backend-workspace "$BACKEND" --output-dir "$OUT"
# 다른 작업은 매번 다른 새 OUT 사용
$PY -B scripts/research_analysis_20261126/analyse_stages.py \
  --workspace "$WORKSPACE" --output-dir /path/to/new-stage-summary
$PY -B scripts/research_analysis_20261126/package_existing.py \
  --workspace "$WORKSPACE" --backend-workspace "$BACKEND" --output-dir /path/to/new-listening-copy
$PY -B scripts/research_analysis_20261126/report_and_verify.py --workspace "$WORKSPACE" --check-inputs
$PY -B scripts/research_analysis_20261126/report_results.py --workspace "$WORKSPACE" --check-inputs
$PY -B scripts/research_analysis_20261126/verify_saved_pcm.py \
  --workspace "$WORKSPACE" --output /path/to/new-pcm-verification.json
```

입력 manifest의 절대 경로는 자동 변경하지 않는다. 다른 PC에서는 원본 자료의 배치를 확인한 비공개 manifest 사본이 필요하다. 원본 manifest는 덮어쓰지 않는다. 패키지의 `organizer_private`, 해시 원본, API 키, 음성, 모델은 Git에 추가하지 않는다.

新 clone でデータなしに実行できる検査:

```bash
$PY -B -m unittest discover -s scripts/psychoacoustic_spec_pilot -p 'test_*.py' -v
$PY -B -m unittest discover -s scripts/research_analysis_20261126 -p 'test_*.py' -v
$PY -B scripts/psychoacoustic_spec_pilot/run_pilot.py --help
$PY -B scripts/psychoacoustic_spec_pilot/run_refinement.py --help
```

## 통합 계획 (이번 작업에서는 실행하지 않음)

1. 저잡음 브랜치 `6205fbb`와 마스킹 보존 브랜치를 향후 별도 통합 브랜치에서 결합한다. 두 브랜치의 `segment_balance.py`·`workspace_paths.py`는 같은 소스다. 분석 브랜치는 마스킹 브랜치를 기반으로 하므로 알고리즘 보존 커밋을 포함하지만, 저잡음 브랜치의 나머지 도구 8개를 포함했다고 간주하면 안 된다.
2. 기존 backend `research/spec-final-validation-20261119`와의 결합 전에 `scripts/protect_wav.py`, `protect_melo.py`, `run_korean_melo_spec.py`, 학습·평가 코드와 고정 모델 provenance를 검토한다. 연구 소스 clone만으로 이 backend와 대용량 고정 모델이 확보되지는 않는다. 새 clone CPU 검사는 가능해도 GPU 생성/학습/평가의 처음부터 재현은 별도 자산이 필요하다.
3. 공식 문서 브랜치와 backend는 `.gitignore`, `README.md`, 공식 문서 세 개 (`README_SPEC_HANDOFF_20261120_KO.md`, `SPEC_APP_INTEGRATION_REVIEW_20261120_KO.md`, `SPEC_FINAL_REPORT_20261120_KO.md`)에 중복 변경이 있다. 향후 최신 검증·한계 설명을 유지하며 수동 비교하고, 모델·WAV·private mapping 제외 규칙을 보존한다. 문서 브랜치와 저잡음 커밋 사이에는 같은 변경 경로가 없다. 이번에는 Merge/PR을 만들지 않는다.
4. Windows의 `feature/windows-spec-integration`은 기존 `feature/ui`에서 파생된 실제 Qt 보호 파이프라인을 가진다. `SpecConfig`의 기본은 ε8·10회이고, 연구 시연 ε2·10회 및 `silence_only_smooth`/마스킹 코드가 적용됐다는 의미가 아니다. 설정은 backend 경계에서 명시적으로 전달하고 검증된 기존 결과와 혼동하지 않게 표시한다. ε2 선택만으로 silence 후처리나 마스킹이 활성화되지 않는다.
5. Windows 파이프라인의 8초 청크·0.5초 overlap·mono 44.1kHz 변환·짧은 청크 padding·재인코딩을 별도 검토한다. 청크 내부 strict PCM ε 통과는 원래 stereo/다른 sample rate의 전체 입력 또는 MP3/MP4 저장 결과에 대한 ε 보증이 아니다. 동일 기준 PCM으로 upload→protect→재결합→재생→저장까지 재검사해야 한다. GUI 테스트의 mock backend 통과와 실제 모델 처리의 끝까지 검증을 구분한다.
6. 이후 새로운 실행 manifest에 모델 해시·seed·설정·청크/codec 조건을 고정해 통합 검증한다. 기존 SIM/CER는 해당 원본 보호 데이터와 공격 조건에 한정된다. 마스킹 3/6dB·A/B는 surrogate와 음질 검사만 있으며 신규 MeloTTS 공격 방어 성능은 미검증이다. 새 코드 게시, PCM 검사 또는 청취용 복사 패키지 생성은 방어 효과의 추가 측정이 아니다.

이번 보존 작업은 기존 알고리즘·측정 결과를 바꾸지 않고 경로와 실행 안전성만 정리한다. 실제 통합, 앱팀 브랜치 수정, 새 보호음 생성과 대규모 학습·공격 평가에는 별도 단계가 필요하다.
