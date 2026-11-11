# KSS 부분 데이터 전달 폴더

만든 도구: `build_kss_manifest.py` (1학년 A, 데이터 담당)

## 출처와 이용 조건

- 데이터: KSS Dataset (Korean Single Speaker Speech Dataset, Kyubyong Park)
- 원본 배포처: https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset
- 실제로 받은 곳: Hugging Face 미러 Bingsu/KSS_Dataset (revision 48fdfd7ab1dbc1a62e4e8a8b9f4c360259d51d3c), parquet 7개 SHA-256 확인함. 미러에는 원본 파일 이름이 없어 발화 ID는 미러 안의 순번(kss_hf_00000부터)임
- 라이선스: CC BY-NC-SA 4.0 (비상업 목적만, 출처 표시, 변경본은 같은 조건으로 공유)
- 이용 상태: `pending_team_confirmation`
  - `pending_team_confirmation`은 팀이 음성 복제 학습에 쓰는 것을 확인하기 전이라는 뜻입니다.
  - 확인이 끝나면 `--usage-confirmed`로 다시 만들거나 manifest의 값을 팀 기록과 함께 바꿉니다.
- 화자는 전문 성우 1명입니다. 화자 ID는 `kss_f1`로 통일했습니다.
- 원본 KSS 폴더는 읽기만 했고 수정하지 않았습니다.

## 고른 방법

- 대본 파일: `transcript.hf.txt` (SHA-256은 manifest.json에 기록)
- 무작위 시드 2026로 섞은 순서대로 검사해서 조건에 맞는 200개를 골랐습니다. 같은 시드와 같은 KSS 파일이면 같은 결과가 나옵니다.
- 길이 2.0초에서 4.0초 사이, 모노, 16비트 PCM만 받았습니다.

## 전처리와 검증 (자동)

- WAV가 열리는지, 프레임 수가 헤더와 맞는지
- 모노, 16비트 PCM인지
- 무음이거나 너무 작은 소리인지, 깨질 만큼 큰 소리(클리핑)인지
- 대본이 비어 있거나 한글이 없는지
- 같은 대본이나 같은 음성 파일(SHA-256이 같음)이 겹치는지
- 복사한 파일의 SHA-256이 원본과 같은지

검사한 파일 252개 중 제외된 파일 사유 (rejects.csv):
- duration_out_of_range: 52개

## 분할

| 분할 | 발화 수 |
|---|---:|
| train | 140 |
| validation | 30 |
| test | 30 |

- 발화 단위로 나눴고 한 발화는 한 분할에만 있습니다. 같은 대본은 걸러서 분할 사이에 겹치지 않습니다.
- 화자가 1명이라 화자 단위로는 나눌 수 없습니다. 같은 책에서 나온 문장이라는 점은 같습니다.
- clean, SPEC 1회, SPEC 10회가 같은 발화 ID와 분할을 쓰도록 이 분할 파일을 기준으로 삼습니다.
- 샘플레이트: 44100 Hz (MeloTTS-Korean은 44100 Hz)

## 폴더 구조

```text
manifest.csv          발화별 정보 (필수 11개 열 + 참고 열)
manifest.xlsx         manifest.csv와 같은 내용 (엑셀용, openpyxl이 있을 때만 생성)
manifest.json         같은 내용과 선택 조건, 대본 파일 해시
transcripts_all.csv   발화별 대본 전체 (발화 ID, 분할, 대본, 길이, 파일 경로)
spec_input.json       한국어 SPEC 실행 도구(run_korean_melo_spec.py)용 입력
splits/               train.txt, validation.txt, test.txt (발화 ID 목록)
train/ validation/ test/
  wavs/               원본과 바이트가 같은 WAV (파일 이름은 발화 ID)
  transcripts.csv     file,text 형식 (MeloTTS prepare_finetune용)
transcript_check.csv  사람이 듣고 대본을 대조해서 채우는 표
rejects.csv           제외한 파일과 사유
```

## 아직 사람이 해야 하는 일

1. `transcript_check.csv`에서 발화마다 듣고 대본이 맞는지 확인하고 표시합니다. 자동 검사는 대본이 실제 발음과 맞는지 알지 못합니다.
2. 전부 맞으면 `--transcripts-verified`로 `spec_input.json`을 다시 만들거나 값을 직접 바꿉니다.
3. 팀이 이용 조건을 확인하면 `--usage-confirmed`를 사용합니다.

## 다른 팀과 연동

- 섭동 담당: `spec_input.json`과 `train/validation/test/wavs`를 전달합니다. 실행 도구는 `transcript_verified`와 `research_use_permitted`가 true여야 실행합니다.
- MeloTTS 담당: 분할별 `transcripts.csv`(file,text)를 쓰면 됩니다. 파일 열의 이름 형식은 `prepare_finetune.py`와 맞는지 확인이 필요합니다.
- 평가 담당: `splits/test.txt`가 별도 test 발화 목록입니다.
- 실제 음성 복제 학습은 이용 조건 확인 후에 진행합니다.

## 모양 확인

`python build_kss_manifest.py --verify <이 폴더>` 로 파일, 해시, 길이, 분할 겹침을 다시 확인할 수 있습니다.
