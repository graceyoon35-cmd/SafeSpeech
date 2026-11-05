# Mac 실제 MeloTTS SPEC 1회·10회 비교와 합성 한국어 입력 검증

실행일: 2026-11-05 · 브랜치: `feature/spec-perturbation` · HEAD: `7c2ec7e`

## 실험 범위

기존 실제 모델 실행 폴더 `runs/melo_mac_actual_20261105_z1muic00/`를 읽기 전용으로
재사용했습니다. 새 결과는 고유 폴더 `runs/melo_iterations_20261105_ipmj1fpe/`에
저장했습니다. 원본 알고리즘·담당자 소스·체크포인트를 수정하지 않았으며 Commit·Push도
하지 않았습니다.

이번 결과는 실제 사전학습 Melo 가중치의 SPEC 실행·미분 가능성·STOI 확인입니다.
한국어 입력은 이전에 모델이 합성한 WAV이며 사람의 한국어 녹음이 아닙니다.
Mock은 사용하지 않았지만 복제 공격 방어·파인튜닝·통계적 보호 성능은 평가하지 않았습니다.

## 재확인한 조건

| 항목 | 값 |
| --- | --- |
| 공식 Melo 소스 | `209145371cff8fc3bd60d7be902ea69cbdb7965a` |
| 담당자 소스 | `b6db484367d38909b4ffd74e06fcf1166cd1e0b9` |
| 공식 한국어 모델 | `myshell-ai/MeloTTS-Korean`, `0207e5adfc90129a51b6b03d89be6d84360ed323` |
| 체크포인트 SHA-256 | `48e3ff3fd0b5348e095f0468e60ae727507564100f58142ef3a922ead6e0a4d0` |
| Python·장치 | 기존 전용 Python 3.10.11, Mac CPU, 스레드 4개 |
| torch / torchaudio | 2.7.1 / 2.7.1 |
| 화자 | 모델 config의 `KR: 0` |
| seed / epsilon / step | 52 / `8/255` / `epsilon/10` |
| mode / beta / alpha | SPEC / 10 / 0.05 (alpha는 SPEC 손실에 사용하지 않음) |

배포 config 누락 항목 5개는 이전과 같은 공식 학습 기본값을 메모리에 보충했습니다.
`win_length=2048`, `n_mel_channels=128`, `mel_fmin=0`, `mel_fmax=null`, `c_mel=45`입니다.
원본 config는 수정하지 않았으며 이전 `effective_loss_config_verified.json`을 사용했습니다.

새 실행 드라이버는 담당자의 원본 `perturb()`를 호출합니다. `torch.autograd.grad()`의
실제 반환값을 관찰하여 매 반복의 loss·gradient를 검사하고, 같은 값을 그대로 반환합니다.
기울기·손실·갱신 알고리즘을 대체하지 않았습니다. 수학적 연산은 모델 가중치를 고정한
train 모드에서 수행합니다.

## 영어 1회·10회 비교

입력은 기존 FLOAT 복사본 `gradient_verified_input/short_input.wav`를 바이트 그대로
새 폴더에 복사했습니다. 24kHz·모노·24,481 samples, 약 1.02004초입니다.
원본 파일 SHA-256은 두 조건 모두
`2c288737af92fc76668be54137e050217d5499d7cc3ae236fc2ad4f32e4eda80`입니다.
모델 내부에서는 담당자 코드가 미분 가능한 44.1kHz 리샘플링을 수행합니다.

| 항목 | 기존 1회 | 새 10회 |
| --- | ---: | ---: |
| STOI | 0.9610571370019663 | 0.7854723272822395 |
| 저장 PCM16 WAV 최대 변화량 | 0.003143310546875 | 0.031402587890625 |
| 첫 반복 loss | 3969.97705078125 | 3969.97705078125 |
| 마지막 반복 loss | 3969.97705078125 | 2043.914306640625 |
| 전체 실행 시간 (모델 로딩 포함) | 약 2.52초 | 약 9.74초 |
| 섭동 함수 시간 | 별도 기록 없음 | 약 8.37초 |
| 프로세스 최대 RSS | 약 1.37GiB | 약 1.59GiB |

10회 minus 1회 STOI는 **-0.1755848097197268**입니다.
10회 첫 반복의 loss와 gradient 최대 절댓값은 이전 1회 기록과 정확히 일치했습니다.
기존 1회 validation을 다시 읽어 재평가한 STOI도 이전 기록과 정확히 같습니다.
이는 첫 단계와 비교 조건의 일치 확인이며 전체 실험의 통계적 재현성을 증명하지는 않습니다.

loss는 각 **갱신 직전** 값입니다. 10번째 갱신 후 최종 loss를 다시 계산한 것으로
해석하면 안 됩니다. 메모리는 프로세스 최대 RSS로, GPU VRAM이나 순수 증분 메모리가
아닙니다. 과거 BERT-VITS2 SPEC 점수와 이번 Melo 점수도 별도 실험입니다.

### 매 반복 점검

| 반복 | total loss | gradient 최대 절댓값 | 유한성·비영 gradient |
| --- | ---: | ---: | --- |
| 1 | 3969.9771 | 2062.2166 | PASS |
| 2 | 2952.8193 | 90.5574 | PASS |
| 3 | 2627.8196 | 114.6555 | PASS |
| 4 | 2463.2976 | 20.5040 | PASS |
| 5 | 2320.1174 | 17.8334 | PASS |
| 6 | 2242.8301 | 17.0967 | PASS |
| 7 | 2178.0417 | 31.8051 | PASS |
| 8 | 2127.0654 | 8.8673 | PASS |
| 9 | 2078.2429 | 63.9528 | PASS |
| 10 | 2043.9143 | 44.2387 | PASS |

각 loss 성분도 모두 유한했습니다. 3회에서는 gradient 원소 1개가 0이었지만
나머지는 0이 아니며 전체 gradient 단절은 없었습니다. 각 반복의 성분 손실·gradient
L2·비영 원소 수·시간은 `english10/result.json`과 생성 보고서의 history에 있습니다.
명료도 감소나 목적 손실 감소만으로 음성 복제 방어 성공을 결론 내리지 않습니다.

## 한국어 합성 WAV 입력 검증

기존 `korean_synthesis.wav`를 변경 없이 새 입력 경로에 복사했습니다.
SHA-256은 `ca64cfae3ff5c82c8e6080535b593bf58afc95bf191f60c6d63ff1d914d7041f`입니다.
이는 “안녕하세요. 음성 보호를 연구합니다.”의 이전 실제 Melo 합성 결과입니다.

입력 점검에서 44.1kHz·모노·196,096 samples·약 4.4466초,
유한값·비무음·진폭 [-1, 1]을 확인했습니다. 모델 기준 Hz와 같으므로 이 경로에서는
리샘플링하지 않습니다. 음성 인코더·디코더 기반 손실에는 텍스트/G2P/BERT 전처리가
필요하지 않고, 설정에 존재하는 화자 `KR: 0`을 사용했습니다.
따라서 음향 입력 조건에 맞는 것으로 판단해 전체 길이에 SPEC 1회를 실행했습니다.

| 항목 | 실제 결과 |
| --- | --- |
| 입력 종류 | 모델 합성 한국어, 사람 녹음 아님 |
| SPEC 반복 | 1회, random_init=False |
| total loss (갱신 전) | 6014.8330078125 |
| gradient | None 아님, 전체 유한, 196,096개 원소 모두 비영 |
| gradient 최대 절댓값 / L2 | 약 416.2489 / 8970.9854 |
| 저장 후 최대 변화량 | 0.0031677665901952423 |
| 저장 형식 | 44.1kHz·모노·PCM16, 입력 길이 유지 |
| STOI | 0.9958038334655199 |
| 섭동 계산 / 전체 시간 | 약 6.00초 / 7.51초 |
| 프로세스 최대 RSS | 약 3.07GiB |
| 검증기·STOI 평가 | 둘 다 PASS |

이 결과는 **합성 한국어 파형을 SPEC 입력으로 처리할 수 있음**을 보여줍니다.
원본 사람 목소리·새 화자·실제 한국어 녹음에 대한 보호 효과로 일반화하지 않습니다.

## 저장 검증·테스트·보존

두 새 산출물에 기존 `validate_melo_spec_pair.py`와 `evaluate_melo_spec_stoi.py`를
실제 CLI로 실행했습니다. 모두 종료 코드 0이며 JSON·CSV 결과는 `PASS`입니다.
원본 해시, 길이·Hz·채널, 유한값·무음·진폭, 보고 delta와 저장 delta를 확인했습니다.
PCM16 양자화로 저장 차이가 epsilon을 조금 넘을 수 있으므로 기존 정책인
`epsilon + 1/32768 + 1e-7`을 적용했습니다. 영어 10회도 이 상한 안에 있습니다.

기존 단위 테스트 **21개를 실행해 전부 통과**했습니다. 새 실험 드라이버 자체의
별도 단위 테스트 수를 추가한 것은 아니며 실제 실행·기존 검증기 통과로 확인했습니다.

실험 전 기존 모델·한국어 BERT·실행 WAV·보고서·드라이버·공식 소스 등
144개 파일의 SHA-256을 기록했고, 실험 후 변경 0개를 확인했습니다.
기존 미커밋 보고서 `MELO_MAC_REAL_MODEL_20261105_KO.md`도 보존 대상입니다.
전용 가상환경 `pip freeze`는 전후 같고 공식 소스 checkout도 깨끗합니다.
체크포인트 해시와 입력·출처 해시도 각각 전후 일치했습니다.

## 새 파일과 실행 방법

새 실행 폴더의 구조:

```text
runs/melo_iterations_20261105_ipmj1fpe/
  baseline.json
  preservation_before.json
  environment_freeze_before.txt
  run_iteration_case.py
  comparison.json
  comparison.csv
  english10/
    input/short_input.wav
    protected/short_input.wav
    protected/report.json
    result.json
    validation.json
    cli_commands.json
    validation_cli.log
    evaluation_cli.log
    evaluation/melo_stoi_dmwzer25/report.json
    evaluation/melo_stoi_dmwzer25/comparison.csv
  korean1/
    input/korean_synthesis.wav
    protected/korean_synthesis.wav
    protected/report.json
    result.json
    validation.json
    cli_commands.json
    validation_cli.log
    evaluation_cli.log
    evaluation/melo_stoi_cjrvvjmi/report.json
    evaluation/melo_stoi_cjrvvjmi/comparison.csv
```

실제로 실행한 주요 명령은 다음과 같습니다. 기존 출력 폴더에는 재실행하지 않습니다.
재실험은 새 고유 폴더에서 같은 고정 소스·모델을 읽도록 준비합니다.

```bash
runs/melo_mac_actual_20261105_z1muic00/venv/bin/python -B \
  runs/melo_iterations_20261105_ipmj1fpe/run_iteration_case.py --case english10
runs/melo_mac_actual_20261105_z1muic00/venv/bin/python -B \
  runs/melo_iterations_20261105_ipmj1fpe/run_iteration_case.py --case korean1
```

검증·평가의 정확한 argv·종료 코드·stdout/stderr는 각 case의 `cli_commands.json`에
있습니다. 새 코드는 이 실행용 드라이버뿐이며 Git 제외인 `runs/` 안에 보관했습니다.
공유용 신규 문서는 이 파일입니다. 기존 미커밋 실제 모델 보고서도 그대로 남아 있습니다.
Commit·Push는 수행하지 않았습니다.

## Windows CUDA 후속 작업

1. 동일 리비전·checkpoint·5개 누락 설정 보충 정책으로 GPU 결과와 VRAM을 검증합니다.
2. 실제 허락받은 한국어 녹음에서 1회/10회·긴 발화·다화자 실험을 수행합니다.
3. clean/protected 동일 split·대본·학습 step의 Melo 파인튜닝 공격을 비교합니다.
4. 화자 SIM·CER·청취 품질·강건성 변환을 평가하고 반복 seed·발화별 통계를 기록합니다.
5. PCM16 저장 오차, 모델·생성 보고서 provenance, 정식 환경 구성은 모델 담당자와
   합의합니다. Mac에서의 단일 사례를 전체 보호 효과로 보고하지 않습니다.
