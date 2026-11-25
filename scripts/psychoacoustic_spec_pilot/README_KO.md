# 청각 마스킹 SPEC 연구 소스

5개 KSS 발화의 마스킹 3/6dB와 투영 개선안 A/B를 보존한 연구 코드다. 계산·PCM 투영·fallback은 기존 원본과 같다. 두 실행기의 import 경로만 변경했고, 저잡음 도구 커밋 `6205fbb`의 `segment_balance.py`, `workspace_paths.py`를 같은 공식 경로에 그대로 포함했다. `runs/`의 ignored Python 파일에 의존하지 않는다.

WSL/Linux의 기존 환경(Python, NumPy, SciPy, soundfile, pystoi, Torch)에서 CPU 검사:

```bash
cd scripts/psychoacoustic_spec_pilot
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -B -m unittest -v test_auditory_masking test_projection_refinement test_research_paths
python -B run_pilot.py --help
python -B run_refinement.py --help
```

실험 실행기는 `--workspace`로 비공개 연구 자료의 루트를 받고, backend는 `SAFESPEECH_LEGACY_WORKSPACE` 또는 해당 루트의 형제 `SafeSpeech-desktop-integration-20261118`에서 찾는다. 예전 결과 폴더가 비어 있는 별도 연구 사본에서만 생성 실행을 검토해야 한다. 모델 자산은 기존 `/home/safespeech/models/melo-spec-pinned-20261118`, backend의 `scripts/run_korean_melo_spec.py`·`protect_melo.py`, CUDA, ffmpeg와 원본 manifest/WAV가 별도로 필요하다. 소스 clone만으로 모델·음성·비공개 정답표가 제공되지는 않는다.

`run_refinement.py`는 당시 HEAD·브랜치·refs·원본 해시를 검사하는 역사적 실행기다. 이 검사를 약화하거나 현재 refs에 맞춰 원본 보존 JSON을 고치지 않았다. 게시 후 HEAD에서 예전 결과 폴더를 사용한 그대로의 재실행은 의도적으로 거부된다. 저장된 결과의 재검증과 CPU 단위 테스트는 새 clone에서 사용할 수 있으나, 옛 실행의 완전 재현에는 백업의 당시 소스/메타데이터와 고정 모델·환경을 별도 준비해야 한다.

마스킹 모델은 상대적 스펙트럼 정규화와 all-tonal masker 근사를 사용한다. 실제 SPL 보정·시간 마스킹·사람 대상 불가청 검증을 대신하지 않는다. 3/6dB와 A/B의 SPEC 목적 변화는 surrogate 측정이며 MeloTTS 공격 SIM/CER로 검증된 방어 성능이 아니다. 이 소스 보존 과정에서는 새 보호 WAV·학습·공격 평가를 실행하지 않는다.
