# data_set

한국어 음성 데이터 후보와 이를 정리한 목록, 도구를 모아 둔 폴더입니다.
작성: 1학년 A (데이터, 환경, 재현성) / 2026-11-11

## 들어 있는 것

| 위치 | 내용 |
|---|---|
| `kss_part_200/` | KSS Dataset에서 고른 200발화의 WAV, 대본, manifest, 분할 목록, 입력 JSON |
| `tools/build_kss_manifest.py` | 발화를 고르고 manifest, 분할, 검증을 만드는 도구 (표준 라이브러리만 사용) |
| `tools/hf_parquet_to_kss_layout.py` | Hugging Face 사본(parquet)을 KSS 원본과 비슷한 폴더로 푸는 도구 (pyarrow 필요) |

## WAV 위치

WAV 200개(약 58MB)는 `kss_part_200/train/wavs/`(140개), `validation/wavs/`(30개), `test/wavs/`(30개)에 있습니다. 파일 이름은 발화 ID이고, `manifest.csv`의 `wav_path`와 `sha256`이 이 파일들과 맞습니다.
같은 파일이 팀 Google Drive의 "We-Meet 데이터 처리" 폴더 안 `KSS_부분데이터_200발화`에도 zip 조각으로 있습니다.

이 저장소는 공개 저장소라서 WAV를 올리면 기록에 남아 되돌릴 수 없습니다. KSS의 라이선스(CC BY-NC-SA 4.0)가 출처 표시와 같은 조건 공유를 지키면 재배포를 허용하므로 올렸습니다. 이 음성을 쓰거나 다시 공유할 때는 아래 인용과 라이선스를 함께 적어야 하고, 상업 목적으로는 쓸 수 없습니다.

## 데이터 출처와 이용 조건

- 데이터: KSS Dataset (Korean Single Speaker Speech Dataset), 전문 여성 성우 1명의 한국어 음성
- 인용: Kyubyong Park, KSS Dataset: Korean Single speaker Speech Dataset, https://kaggle.com/bryanpark/korean-single-speaker-speech-dataset, 2018
- 라이선스: CC BY-NC-SA 4.0 (비상업 목적만, 출처 표시, 바꾼 결과물은 같은 조건으로 공유)
- 실제로 받은 곳: Hugging Face `Bingsu/KSS_Dataset` (revision 48fdfd7ab1dbc1a62e4e8a8b9f4c360259d51d3c). 제3자가 올린 사본이며 parquet 7개의 SHA-256은 공개된 값과 일치함을 확인했습니다. 원본 Kaggle 사본과 같은지는 확인하지 못했습니다.
- 변경 사항: 고른 WAV는 바이트 그대로 복사했고 파일 이름만 발화 ID로 바꿨습니다. 대본은 원작자가 배포한 원문 열을 그대로 썼습니다.

## 현재 상태

- 대본이 음성과 맞는지 사람이 듣고 확인하는 일은 아직 하지 않았습니다 (`kss_part_200/transcript_check.csv`).
- 팀이 이용 조건을 확인하기 전이라 이용 상태는 `pending_team_confirmation`입니다.
- 발화 ID(`kss_hf_00078` 등)는 받은 사본 안의 순번입니다. 사본에 원본 파일 이름이 없어 원본 번호와의 대응은 확인할 수 없습니다.
- 받은 사본은 12,854행이며 원작자가 밝힌 12,853개보다 1개 많습니다. 모노가 12,853개, 스테레오가 1개이고 스테레오는 도구가 제외합니다.

## 다시 만드는 방법

```bash
python tools/hf_parquet_to_kss_layout.py --parquet-dir <parquet 폴더> --out-dir <새 폴더>
python tools/build_kss_manifest.py --kss-root <위에서 만든 폴더> --out-dir <새 폴더> --n 200 --seed 2026
python tools/build_kss_manifest.py --verify <만든 폴더>
```

같은 시드와 같은 사본이면 같은 200발화가 선택됩니다.
