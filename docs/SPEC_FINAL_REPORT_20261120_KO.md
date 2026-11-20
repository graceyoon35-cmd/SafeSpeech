# SPEC 연구 개발 최종 보고서 — 2026-11-20

생성·학습·방어 성능 평가 및 앱팀 오프라인 인계 준비를 완료했다. 11월 19일 보고서의 보정본 SIM·CER 미측정 항목은 아래 별도 실험으로 갱신한다. 기존 실험·체크포인트·백업은 보존했다.

## 검증된 성과

- KSS 200개, 분할 140/30/30, 단일 화자, 동일 초기 생성기, batch 1, BF16, segment 8192, AdamW 0.0001, 1,000-step 제한 실험이다.
- seed53 기존 평가 90건과 보정본 평가 90건의 WAV 해시·문장·합성 seed·독립 편집거리·macro 및 corpus CER를 검산했다. 두 평가의 SIM도 ECAPA 재인코딩으로 검산했다.
- 기존 증거 2140개 파일의 SHA256이 유지됐다. 신규 보정본만 별도 학습했고 Clean은 기존 seed52 체크포인트를 재사용했다.

## 최종 복제 합성 평가

CER 전사는 Whisper small, language ko, temperature 0으로 수행하고 정규화는 `[^0-9A-Za-z가-힣]` 제거를 사용한다. ECAPA 입력은 16kHz로 리샘플링한다. SIM은 고정된 clean 학습 음성 5개에 대한 ECAPA cosine 평균이다. CER는 30개 테스트 발화별 비율의 평균(macro), corpus CER는 문자 수 가중 전체 비율이다. 낮은 SIM·높은 CER는 본 복제 실험에서 방어 방향이지만 차단 성공률은 아니다.

| 학습 seed | 입력 조건 | SIM | macro CER | corpus CER |
|---:|---|---:|---:|---:|
| 52 | clean | 0.5099738 | 13.4606% | 12.1581% |
| 52 | SPEC_1 | 0.4744760 | 15.7133% | 16.7173% |
| 52 | SPEC_10 | 0.1627789 | 174.0339% | 181.7629% |
| 53 | clean | 0.5134293 | 15.7430% | 14.8936% |
| 53 | SPEC_10 | 0.1328125 | 446.3187% | 444.3769% |
| 52 | SPEC_10_strict | 0.1451893 | 286.7974% | 310.9422% |

## ε 보정 전후의 차이

기존 SPEC_10은 PCM16 저장 반올림으로 200개 모두 엄격한 ε=8/255를 최대 1 LSB 이내 초과했다. ε+1/32768를 허용하던 기존 검증에서는 통과했다. 원본 파일은 수정하지 않았다. 정수 투영 보정본은 원본 PCM16 대비 floor(ε×32768)=1028 code 이내로 제한한다.

| 보호 입력 | 개수 | 평균 STOI | 평균 SNR (dB) | 최대 진폭 차이 | 엄격한 ε 초과 |
|---|---:|---:|---:|---:|---:|
| SPEC_1 | 200 | 0.9973078 | 28.6835017 | 0.003143310546875 | 0 |
| SPEC_10 | 200 | 0.9016711 | 11.7870250 | 0.031402587890625 | 200 |
| SPEC_10_strict | 200 | 0.9017007 | 11.7890859 | 0.031372070312500 | 0 |

보정본은 섭동을 새로 최적화한 결과가 아니라 기존 SPEC_10의 저장 단계 정수 투영 사본이다. seed52·분할·텍스트 특징·초기 가중치·모델·학습 설정을 유지하고 WAV 경로·해시, manifest/list 경로, 조건 이름만 바꾸었다. 같은 30문장과 합성 seed 2026~2055, Whisper small 기본 한국어 decoder를 사용했다.

보정 후 Clean 대비 방어 방향(낮은 SIM·높은 CER) 동시 충족: **관측됨**. 보정본 SIM 0.1451893, macro CER 286.7974%. 한 번의 추가 학습이므로 기존 대비 차이를 저장 보정만의 인과 효과로 해석할 수 없다. CUDA 학습 변동이 함께 포함된다.

## ASR 민감도와 GPU 재현 범위

CER 100% 초과는 반복 전사의 삽입 오류를 포함한 유효한 편집거리 비율이다. 주 결과에서 잘라내거나 상한 처리하지 않았다. 기존 seed52 SPEC_10의 짧은 발화 decoder 민감도 결과는 174.0339% → 115.3416%다. 이는 사후 별도 분석이며 주 결과를 대체하지 않는다. seed53 및 보정본에는 이 추가 decoder 분석을 수행하지 않았다. 발화별 삽입 수·최대 CER 전사 미리보기는 final_verification.json에 보존했다. 별도 decoder 민감도는 원래 120개 WAV 전체에 `sample_len=64`, `condition_on_previous_text=False`를 동일 적용했고 Whisper small·한국어·temperature 0은 유지했다. Clean·SPEC_1 값은 그대로이며 표본 제거·CER 상한 처리·원본 덮어쓰기는 하지 않았다.

기존 동일 seed GPU 보호 두 실행은 76,205샘플이 달랐고 출력 간 최대 차이는 0.050201416015625였다. 각 출력은 입력 대비 엄격한 ε를 만족했다. 강제 결정론은 reflection_pad1d_backward_out_cuda의 결정론 구현 부재로 실패했다. 해당 연산을 유일 원인으로 단정하지 않는다. 실패 로그를 보존했고 추가 장시간 해결을 시도하지 않았다. 같은 seed는 바이트 동일성 보장이 아니다. 재현 범위는 이 고정 환경·동일 화자·짧은 제한 학습에서 방어 지표 방향을 확인한 수준이다.

## 테스트와 앱팀 전달

- 2026-11-20 전체 자동 테스트 99개 PASS, pip check PASS.
- 실제 WAV→보호 WAV GPU 실행 PASS: 141,123샘플, mono 44,100Hz PCM16, 최대 차이 0.0313720703125, STOI 0.8888067, SNR 11.494176dB.
- 최종 저장 WAV를 독립 재계산했고 입력 해시 보존, 기존 출력 폴더 재사용 거부(exit 1), 기존 결과 해시 보존을 확인했다.
- 실행·의존성·입출력·예제 경로: [앱팀 README](README_SPEC_HANDOFF_20261120_KO.md). UI·프론트엔드는 앱팀 범위다.

## 한계

- 단일 화자, 두 원래 학습 seed, 보정본 학습 seed 하나, 단일 대리·복제 모델, 1,000-step 제한이다. 장기 수렴·다화자·다른 공격 모델로 일반화하지 않는다.
- 사람 청취 검수 미완료, 사전학습 코퍼스 비중복 미확인이다. STOI/SNR은 보호 입력, SIM/CER는 복제 합성이므로 같은 품질 지표로 섞지 않는다.
- SPEC_10의 보호 입력 STOI 약 0.902와 SNR 약 11.79dB는 품질 비용을 보인다. 무손실·불가청·복제 불가능으로 발표하지 않는다.

## 미해결 사항

- GPU 바이트 동일성 및 강제 결정론 실행 실패.
- 독립 섭동 seed 반복, SPEC_1 추가 학습 seed, 보정본 두 번째 학습 seed, 보정본 decoder 민감도는 미실행.
- CPU 실행 성능 및 새 기기의 전체 신규 설치는 미검증. 앱 UI·비동기 실행·배포 연결은 앱팀 담당.
- 별도 화자 동의·음성/가중치 외부 공개 승인을 확보한 것으로 기록하지 않는다. 기존 KSS 내부 비상업적 연구 범위를 유지한다.

## 발표 그래프

![최종 SPEC 지표](figures/spec_release_results_20261120.svg)

CSV·PNG·SVG 및 재검산 색인은 runs/final_validation_20261120/presentation/에 있다. Git·백업·체크포인트 인계 상태는 [실행 가이드](README_SPEC_HANDOFF_20261120_KO.md)에 통합했다.

원래 평가 120개 WAV와 seed53·보정본 각 90개 WAV의 독립 ECAPA 재인코딩은 기존 SIM과 최대 절대 차이 0이었다. 보호 입력 기존 400개와 정수 보정 사본 200개는 mono 44.1kHz PCM16이며 clipping rail 샘플이 0개였다. 상세 해시·발화별 검산·품질 CSV는 원래 실험 폴더에 유지한다.

## 실험 프로토콜과 원본 데이터 품질

이 절은 기존 KSS 실행 프로토콜과 사용 근거를 통합한 것으로, 새로운 사용 승인이나 법률 판단을 생성하지 않는다.

- KSS 원본 200개, train 140 / validation 30 / test 30을 유지했다. 디코딩·mono/PCM16·길이·유한 값·샘플레이트·비무음·대본·해시·split을 검사했고, 한국어 BERT 특징은 해시 검증 후 재사용했다.
- 원본 대본과 Whisper ASR의 정규화 일치는 142/200, 불일치는 58/200이었다. 원본 녹음 CER 단순 평균 약 4.918%는 대본 점검용 수치이며 복제 합성 CER가 아니다. ASR 일치는 청취 완료가 아니고, 불일치는 대본 오류의 증명도 아니다. 권장 검수 목록은 제외 목록과 다르며 구조 검사상 제외는 0개였다.
- 최초 실행 순서는 clean → SPEC_10 → SPEC_1, 동일 초기 생성기·seed52·batch1·BF16·segment8192·AdamW 0.0001이다. 파형·길이 판별기는 동일 seed의 무작위 초기화이며 사전학습 판별기로 표시하지 않는다.
- `train_melo_bounded.py`는 공식 Melo 생성기·MPD·duration discriminator·mel/KL/duration/feature/adversarial 손실을 쓰는 제한 학습기다. 공식 train.py 전체·DDP·epoch scheduler를 그대로 실행한 결과가 아니다. optimizer 3개와 고정 LR을 사용하므로 scheduler 상태는 없다.
- 최초 seed52는 250-step마다 완전 checkpoint를 별도 디렉터리에 저장했다. seed53과 엄격한 ε 보정본은 최종 1,000-step을 저장했다. 같은 발화 ID·대본·split·특징을 조건 간 유지했다.
- 평가는 같은 test 30문장과 합성 seed 2026~2055, clean train 기준 화자 WAV 5개, pretrained baseline, 고정 ECAPA·Whisper small 한국어 decoder로 수행했다. SIM은 성공 threshold 없는 cosine 평균이다. 발화별 값·분산·문장 bootstrap 및 원본 ASR 일치/불일치 strata를 CSV에 남겼다.
- bootstrap 구간은 문장 내 변동만 나타내며 화자·학습 seed 간 불확실성을 포함하지 않는다. test는 이번 미세조정에서만 held-out이다. 사전학습 checkpoint의 전체 코퍼스와 KSS 비중복을 확인하지 못했다.
- seed53은 학습 seed만 52→53으로 바꾸고 기존 섭동 seed52·보정 전 WAV를 재사용했다. 독립 섭동 seed 재현이 아니다. pretrained 재합성의 기존 대비 해시는 30/30 일치했다. 보정본은 seed52의 별도 학습이며 기존 Clean checkpoint를 재사용했다.

## KSS 출처·라이선스·데이터 사용 조건

출처 표기는 **Kyubyong Park, KSS Dataset: Korean Single Speaker Speech Dataset, 2018**, 원 배포 URL, CC BY-NC-SA 4.0, 선택·파일명 정리·SPEC 변형 등의 실제 변경 내역을 함께 기록한다.

| 근거 | 고정 내용·확인 한계 |
|---|---|
| [원 배포처](https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset) | 한국어 TTS·비상업 용도 및 CC BY-NC-SA 4.0 표기. 당시 직접 본문은 반환되지 않았고 Kaggle API 2개는 404였으므로 검색 색인으로 확인했다. 현재 로그인 화면의 계약을 확인했다고 주장하지 않는다. |
| [실제 사용한 미러](https://huggingface.co/datasets/Bingsu/KSS_Dataset/tree/48fdfd7ab1dbc1a62e4e8a8b9f4c360259d51d3c) | 리비전 `48fdfd7ab1dbc1a62e4e8a8b9f4c360259d51d3c` 카드·WAV/manifest 해시를 보존했다. 미러를 원 저자의 추가 승인으로 해석하지 않으며 원 배포 파일과 모든 바이트가 같다고 주장하지 않는다. |
| [CC BY-NC-SA 4.0 약관](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en) | 기존 검토는 Section 2(a), 2(b), 3을 근거로 기록했다. 로컬 HTTP 403도 보존했다. 출처·변경 표시, 비상업 조건, 적용되는 동일조건 의무를 유지한다. |
| [저자의 예제 코드](https://github.com/Kyubyong/kss) | 코드 저장소의 Apache-2.0을 음성 데이터의 라이선스로 적용하지 않는다. |

현재 내부 대학 연구는 명시적 `licensed_noncommercial_kss_research_v1` 정책으로 실행했다. 비상업적 내부 SPEC·clean/protected 학습·평가는 해당 공개 조건을 근거로 진행했으며 상업적 사용은 수행하지 않는다. 기존 strict validator를 완화하거나 모든 입력에 이 정책을 자동 적용하지 않는다.

원본의 `research_use_permitted=false`, `transcript_verified=false`, `pending_team_confirmation`은 과거 선언 그대로 유지했다. config의 `permission_declarations_validated=false`와 `research_policy_validated=true`를 구분한다. 고정 원본 ID·대본·split·WAV 해시·라이선스 카드·과거 선언이 달라지면 정책 경로도 거부한다. 초기 strict 경로의 미검수 거부 기록은 현재 모든 내부 연구가 금지됐다는 뜻이 아니다.

별도 화자 동의는 `not_verified`, 음성·모델 외부 공개는 `not_approved_by_this_record`다. 저작권상 조건과 개인정보·인격권을 동일시하거나 약관이 모든 권리를 부여한다고 해석하지 않는다. 반대로 확인되지 않은 별도 동의나 전수 청취를 모든 내부 TTS 연구의 필수 조건으로 추가하지 않는다. 외부 WAV 공유는 출처·변경·비상업·해당 동일조건을 검토해야 하며, 가중치·합성물이 언제 변형물인지 일괄 확정하지 않았다. 외부 공개나 화자의 지지·승인을 확보한 것으로 기록하지 않는다.

기존 근거·HTTP 실패 기록은 `runs/training_20261118/license_evidence/sources.json`·`license_fetch.log`에 있고, 사람 청취·외부 공개는 여전히 미완료 범위다. 이 문서 통합 과정에서 원본 manifest·정책 JSON·승인·검수 플래그는 수정하지 않았다.

## 이전 실험에서 이어받은 결과와 한계

아래는 삭제된 중간 문서의 기술적 결론을 통합한 것이다. 서로 다른 모델·화자·입력·학습량을 같은 방어 실험으로 합치지 않는다. 상세 손실·해시·시간은 기존 CSV/JSON과 Git 원문에 남아 있으며 이 문서 정리를 위해 재실행하지 않았다.

| 단계 | 기존 실측 | 해석 범위·보존 위치 |
|---|---|---|
| 2026-10-30 BERT-VITS2 CUDA | 영어 1.02004초, 24kHz, FLOAT 보호 WAV. SPEC 1/10회 STOI 0.9612467971012578 / 0.7754191151068554, 최대 차이 0.003137260675430298 / 0.0313725508749485 | 영어 단일 발화 명료도 점검이며 복제 방어 성능 아님. `runs/spec_stoi_20261030_011000_995566/`, `runs/smoke_20261030_005658_023201/`, `runs/smoke_20261030_005719_318250/` |
| 2026-11-05 Melo Mac CPU | 영어 SPEC 1/10회 STOI 0.9610571370019663 / 0.7854723272822395. 합성 한국어 1회 STOI 0.9958038334655199, 최대 차이 0.0031677665901952423 | 실제 사전학습 가중치 gradient·산출물 확인. 사람 녹음 보호·GPU·다화자 효과 아님. `runs/melo_mac_actual_20261105_z1muic00/`, `runs/melo_iterations_20261105_ipmj1fpe/` |
| 2026-11-10 한국어 합성 입력 | SPEC 1/10회 STOI 0.9961603424775992 / 0.8757433629288034 | seed 초기화 위치 등이 앞 실험과 달라 11/5 숫자의 정확한 재현으로 주장하지 않음. `runs/korean_melo_spec_85hbcsho/` |
| 합성 3-step 스모크 | pretrained / clean / SPEC_1 / SPEC_10 SIM 0.804136 / 0.780215 / 0.788476 / 0.796549, CER 모두 15.3846% | 합성 train2·validation1·test1, 기준 train WAV2개. 학습 경로 점검이고 KSS 방어 근거 아님. `runs/paired_melo_training_063mlxyb/`, `runs/melo_smoke_evaluation_29va9n1c/` |
| 합성 100-step·12문장 | pretrained / clean / SPEC_1 / SPEC_10 SIM 0.7877 / 0.7867 / 0.7258 / 0.3916; macro CER 4.72 / 5.21 / 8.07 / 42.06%; corpus CER 4.43 / 4.93 / 7.88 / 41.87% | 합성 train2·단일 seed의 48개 평가 WAV. KSS나 사람 전체 방어 효과로 일반화하지 않음. `runs/melo_timing_100_4n35l8ff/`, `runs/melo_smoke_evaluation_yktb2105/` |

100-step 실험은 clean step50 저장 후 의도적 중단·별도 프로세스 재개를 확인했다. 세 optimizer step50, RNG·데이터 순번·MAS schedule·loss history를 복원했고 재개 분기 step51/52 손실·RMS 최대 차이는 0.0이었다. 이는 무중단 실행이나 다른 GPU/OS의 바이트 재현 검증이 아니다. 완전 manifest가 없는 저장 중 디렉터리는 복구 대상으로 거부한다. 전원 장애 자체는 시험하지 않았으며 마지막 완전 checkpoint 이후 step은 유실될 수 있다.

합성 100-step의 clean/SPEC_1/SPEC_10 평균 초/step은 0.5793/0.5685/0.6102, warm-up 제외는 0.5345/0.5297/0.5729, peak allocated는 약 2.052GiB였다. Windows 마운트 저장 17.26초와 WSL 내부 2.37초 비교는 별도 clean 50→52 분기의 저장 1회 표본이다. 저장 7.3배 차이를 전체 학습 속도 향상으로 해석하지 않는다. 이를 사용한 1k/5k/10k 일정 추정은 합성 입력의 조건부 계획이었으며 현재 KSS 실측이나 수렴 보장이 아니다. 강건성 변환 산출물도 원음 품질 검사와 변환 후 재학습 방어 평가를 구분하며 후자를 수행한 것으로 표시하지 않는다.

초기 Mac의 전체 `melo.api`는 MeCab 충돌로 실패했고, 한국어 전용 harness·공식 모델 infer로 검증했다. CPU 결과를 MPS/CUDA 성공으로 표시하지 않으며 Mac 패키지 조합을 현재 WSL 환경에 섞지 않는다. weight_norm probe 오류·무음 crop·초기 선언 gate 거부·결정론 실패 등의 로그는 원본 실험 폴더에 보존했다. 과거 mock 계약 검사는 `MOCK_PASS`, 합성 실행은 `NONHUMAN_EXECUTION_PASS`로 구분하며 사람 음성 방어 성공으로 바꾸지 않는다.

## 문서 통합과 원문 보존

2026-11-20 문서 축소 전 추적 Markdown은 저장소 전체 39개, docs 내부 23개였다. 중간·중복 문서 20개를 삭제하고 docs에는 최종 보고서·실행 가이드·앱 연동 점검 3개만 남겼다. 최상위 README와 모델/하위 프로젝트 문서 15개는 유지하므로 저장소 전체 Markdown은 19개다.

삭제 전 현재 본문이 [정리 전 커밋 9882630](https://github.com/graceyoon35-cmd/SafeSpeech/tree/9882630dc3eabeeb801b1a3d1680e9e427067a58/docs)에 모두 저장돼 있음을 파일별로 확인했다. 과거 전체 명령·수치·개발 일지는 `git show 9882630:<기존 경로>`로 조회한다. 새 archive 문서를 만들지 않았다. 기존 archive 아래 그래프는 원본 자산이므로 삭제·이동하지 않았다. CSV·JSON·설정·스크립트·모델·학습 로그·백업도 보존했다. 기존 보고서 생성기의 삭제 문서 참조 2곳만 최신 가이드 경로로 바꾸었다.
