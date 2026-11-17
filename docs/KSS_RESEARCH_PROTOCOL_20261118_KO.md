# KSS 비상업적 연구 실행 범위와 품질 정책 — 2026-11-18

사용자는 이 프로젝트가 비상업적 대학 연구이며, 전수 청취를 개발/학습의 필수 조건으로 삼지 말라고 명시했다.
이에 따라 이전의 전수 청취 및 별도 팀 승인 bool에 의존하던 실행 정책과 별도로,
공개 라이선스에 근거한 `licensed_noncommercial_kss_research_v1` 경로를 만들었다.
기존 승인 대기·미청취 기록을 승인/검수 완료로 덮어쓰지 않는다.

## 근거와 실제 범위

출처 표기: Kyubyong Park, KSS Dataset: Korean Single Speaker Speech Dataset, 2018,
원 배포처 URL 및 CC BY-NC-SA 4.0 링크. 200개 선택·파일명 정리·SPEC 섭동 생성 등 변경 사항을 함께 기록한다.

- [원 배포처](https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset),
  [배포 설명의 검색 색인](https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset/kernels):
  한국어 TTS 목적과 비상업적 사용을 명시하고 라이선스를 CC BY-NC-SA 4.0으로 표시한다.
- [CC BY-NC-SA 4.0 약관](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en)의
  비상업적 복제·변형 범위와 원 배포 목적을 이번 내부 SPEC·TTS 학습·평가 합성의 근거로 적용한다.
  이는 용도를 한정한 공개 조건 해석이며 별도 화자 동의를 확보했다는 기록이 아니다.
- 실제 데이터는 [Bingsu/KSS_Dataset](https://huggingface.co/datasets/Bingsu/KSS_Dataset)의
  고정 리비전 `48fdfd7ab1dbc1a62e4e8a8b9f4c360259d51d3c`에서 팀이 확보한 200개다.
  고정 카드와 WAV/manifest 해시를 검사한다. 원본 Kaggle 사본과 모든 바이트가 같다고 추가 주장하지 않는다.
- 비상업적 내부 연구용 SPEC·clean/protected 파인튜닝·내부 평가 합성은 진행한다.
  상업적 사용은 해당 공개 조건에 맞지 않아 수행하지 않는다.
  별도 화자 동의는 `not_verified`, 음성/가중치 외부 공개는 `not_approved_by_this_record`로 남긴다.
  이 작업에서 외부 업로드나 화자의 지지/승인을 암시하는 공개를 하지 않는다.
- 공개 조건이 자동으로 모든 인격권을 부여한다고 해석하지 않는다. 다만 확인하지 못한 별도 동의나
  전수 청취를 근거 없이 모든 내부 TTS 연구의 선행 조건으로 추가하지 않는다.

## 품질 관리와 데이터 보존

원본 KSS 200개와 train 140 / validation 30 / test 30을 그대로 유지한다.
디코딩·모노/PCM16·길이·유한 값·샘플레이트·비무음·해시·대본·split 검증을 적용한다.
기존 한국어 BERT 특징은 해시로 검증 후 재사용한다.

ASR 불일치 58개는 별도 `review_recommended.csv`에 기록한다.
142개의 ASR 일치도 사람 청취 완료로 바꾸지 않는다.
추가 자동 음질 지표는 RMS, peak, 낮은 진폭 비율이다. 검수 권장 목록은 제외 목록과 다르다.
현재 구조 검사에 실패한 파일이 없어 제외는 0개이며, `excluded_utterances.csv`는 header만 가진다.
모든 조건은 같은 전체 발화 ID·원본 split·대본·특징으로 학습/평가한다.

원래 `transcript_verified=false`, `research_use_permitted=false`, `pending_team_confirmation`은 과거 선언 그대로다.
실행 근거는 이 legacy bool을 true로 만드는 대신 별도의 public-license policy에 기록한다.
config에도 `permission_declarations_validated=false`, `research_policy_validated=true`를 구분한다.
이 차이는 "별도 승인 선언 검증"과 "공개 조건에 근거한 연구 범위 검증"을 혼동하지 않기 위한 것이다.

기본 strict validator는 기존처럼 동작한다. 새 정책은 명시적 opt-in일 때만 허용되며
원본 ID·대본·split·WAV 해시·과거 선언 및 고정 라이선스 카드가 바뀌면 거부한다.
새 policy를 이유로 미검수 상태, 별도 동의 또는 외부 공개 승인을 허위로 변경할 수 없도록 검사한다.

## 실험 사전 설정

- 순서: clean → SPEC_10 → SPEC_1, 각 1,000-step.
- 공통 초기 생성기 checkpoint, seed 52, batch 1, BF16, segment 8192, AdamW LR 0.0001.
- 파형·길이 판별기는 동일 seed의 무작위 초기화. 사전학습 판별기를 사용한 것으로 기록하지 않는다.
- 매 250 step에 새 checkpoint를 WSL 내부 디스크에 저장한다.
- 평가는 동일한 held-out test 30문장과 동일 합성 seed, 같은 clean train 기준 화자 WAV 5개를 사용한다.
- 사전학습 모델 baseline도 함께 평가한다. SIM·CER의 발화별 값·평균·분산·문장 bootstrap을 기록한다.
- 원본 녹음의 ASR 일치/불일치 strata를 평가 CSV에 함께 기록한다. 사람 청취 검증으로 표시하지 않는다.
- 기존 합성 100-step 실험과 실제 KSS 실험은 별도 study·CSV·그래프로 구분한다.

이 문서는 실행 프로토콜이다. 실제 완료 여부·손실·SIM/CER는 study.json 및 이후 실측 보고서를 기준으로 판단한다.

추가 해석 제한: [공식 MeloTTS-Korean 모델 카드](https://huggingface.co/myshell-ai/MeloTTS-Korean)와
[공식 학습 안내](https://github.com/myshell-ai/MeloTTS/blob/main/docs/training.md)를 확인했으나,
현재 고정 사전학습 checkpoint의 전체 코퍼스 목록과 KSS 비중복을 확인하지 못했다.
따라서 test split은 이번 미세조정에서만 held-out이며, 사전학습 단계에서도 미노출이었다고 주장하지 않는다.
