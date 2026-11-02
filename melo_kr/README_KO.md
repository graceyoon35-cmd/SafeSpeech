# melo_kr: MeloTTS 한국어 음성 보호 + XTTS 공격 평가

SafeSpeech(SPEC) 방식을 한국어로 옮긴 코드다. 섭동은 **MeloTTS-Korean**으로 만들고, 보호 효과는 두 가지 공격으로 잰다.

| 공격 | 방법 | 의미 |
|---|---|---|
| 파인튜닝 공격 | 보호 음성으로 MeloTTS를 학습한 뒤 합성 | 섭동을 만든 모델에 대한 방어 (논문 Table 1에 해당) |
| 제로샷 공격 | 보호 음성 1개를 참조로 XTTS-v2가 바로 합성 | 다른 모델에도 통하는지 (전이성) |

원본 SafeSpeech 코드(`protect.py`, `train.py`, `bert_vits2/` 등)는 수정하지 않는다. 이 폴더만 추가된다.

> **상태:** 무작위 가중치로 코드 경로만 검증했다. 실제 MeloTTS·XTTS 가중치로는 아직 돌리지 않았고, 보호 효과는 미검증이다.

## 파일

| 파일 | 하는 일 | 실행 환경 |
|---|---|---|
| `download_models.py` | MeloTTS-Korean, XTTS-v2 가중치 받기 | 기본 |
| `protect_melo.py` | **섭동 생성** (원본 `protect.py`의 MeloTTS판) | 기본 |
| `prepare_finetune.py` | 파인튜닝 공격용 44.1kHz 데이터·목록 만들기 | 기본 |
| `synth_melo.py` | 파인튜닝된 MeloTTS로 테스트 문장 합성 | MeloTTS |
| `attack_xtts.py` | XTTS 제로샷 복제 | XTTS |
| `evaluate_kr.py` | SIM(ECAPA), 공격 성공률, CER(Whisper) | 기본 |
| `common.py` | 공통 함수 (손실은 원본 `protect.py`와 같은 식) | - |
| `test_sentences_ko.txt` | 합성에 쓸 한국어 문장 10개 | - |

## 원본 protect.py와 같은 점, 다른 점

- **같음:** 손실(pivotal 멜 L1 × c_mel + β·(잡음 KL + 잡음 멜 L1), SafeSpeech 모드는 + α·(STOI + STFT)), ε = 8/255, 200회, 한 걸음 ε/10 부호 기울기, β = 10, α = 0.05, 매 반복 같은 시드의 잡음 z.
- **다름:** 대리 모델이 MeloTTS-Korean(44.1kHz). 손실이 실제로 쓰는 경로인 음성 인코더(`enc_q`) → 디코더(`dec`)와 화자 임베딩만 계산한다. 원본에서도 텍스트·BERT 경로는 손실에 들어가지 않으므로 계산 결과는 같고, **대본과 BERT 없이** 섭동을 만들 수 있다. 파일을 하나씩 처리하고, 섭동은 입력 샘플레이트 그대로 만든다.

## 1. 준비 (처음 한 번)

```bash
# 저장소 루트에서
git clone https://github.com/myshell-ai/MeloTTS third_party/MeloTTS

# 기본 환경 (기존 safespeech 가상환경)
source ~/.venvs/safespeech/bin/activate
pip install openai-whisper speechbrain jiwer       # 평가용
python melo_kr/download_models.py --melo --xtts    # checkpoints/melo_kr, checkpoints/xtts_v2
```

**XTTS 환경** (패키지 충돌을 피하려고 따로 만든다)

```bash
python3 -m venv ~/.venvs/xtts && source ~/.venvs/xtts/bin/activate
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install "coqui-tts[ko]" soundfile
```

**MeloTTS 환경** (MeloTTS는 transformers 4.27.4 등 옛 버전을 고정해서 따로 만든다)

```bash
python3 -m venv ~/.venvs/melo && source ~/.venvs/melo/bin/activate
pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install -e third_party/MeloTTS
python -m unidic download
```

한국어 G2P(g2pkk)는 MeCab-ko가 필요하다. Python 3.10에서는 `pip install python-mecab-ko`가 미리 빌드된 패키지로 설치된다(3.13에서는 빌드 실패를 확인함).

## 2. 데이터

보호할 화자의 한국어 녹음을 `data/korean/clean/`에 WAV로 둔다. 한 파일 2~10초, 모노 권장. 섭동 생성에는 대본이 필요 없다.

**주의:** 실제 사람의 목소리라 GitHub에 올리면 안 된다. `.gitignore`에 `/data/korean/`, `/data/melo_ft/`이 들어 있다. 녹음한 사람의 동의를 받는다.

## 3. 섭동 생성 (기본 환경, GPU)

```bash
python melo_kr/protect_melo.py --input-dir data/korean/clean --out-dir data/korean/protected/SPEC
python melo_kr/protect_melo.py --input-dir data/korean/clean --out-dir data/korean/protected/SafeSpeech --mode SafeSpeech
```

처음에는 `--limit 1 --log-every 20`으로 한 개만 돌려 `report.json`의 손실 변화와 SNR을 보고, 직접 들어 본다. CPU로는 3회에 약 13초(3초 음성)가 걸려 200회는 GPU가 필요하다. 8GB GPU에서 몇 초 길이까지 되는지는 아직 재지 않았다.

## 4. 제로샷 공격 (XTTS 환경)

```bash
source ~/.venvs/xtts/bin/activate
python melo_kr/attack_xtts.py --refs data/korean/clean --label clean --limit-refs 5
python melo_kr/attack_xtts.py --refs data/korean/protected/SPEC --label SPEC --limit-refs 5
```

같은 문장·같은 시드로 합성해 조건끼리 공정하게 비교한다. 결과: `runs/xtts_attack/<label>/metadata.csv`

## 5. 파인튜닝 공격 (MeloTTS 환경)

```bash
# (기본 환경) 학습 데이터 만들기: 원본은 Whisper로 대본을 만들고, 보호본은 그 대본을 재사용
python melo_kr/prepare_finetune.py --audio-dir data/korean/clean --name clean --asr
python melo_kr/prepare_finetune.py --audio-dir data/korean/protected/SPEC --name SPEC \
    --transcripts data/melo_ft/clean/transcripts.csv

# (MeloTTS 환경) 전처리 → 학습. 한국어 체크포인트에서 시작한다
source ~/.venvs/melo/bin/activate
cd third_party/MeloTTS/melo
python preprocess_text.py --metadata ../../../data/melo_ft/SPEC/metadata.list
torchrun --nproc_per_node=1 train.py --c ../../../data/melo_ft/SPEC/config.json \
    --model melo_ft_SPEC --pretrain_G ../../../checkpoints/melo_kr/checkpoint.pth
cd ../../..
```

- 학습 전 `data/melo_ft/SPEC/config.json`에서 `train.batch_size`(기본 6, 메모리 부족하면 낮춤)와 `train.epochs`(기본 10000)를 정한다. clean과 SPEC은 **같은 설정·같은 스텝 수**로 맞춘다.
- 시작할 때 `emb_g.weight` 크기가 다르다는 로그가 나오는데, 화자 수가 256 → 1로 바뀌어서 그 부분만 새로 학습하는 정상 동작이다.
- 체크포인트는 `third_party/MeloTTS/melo/logs/melo_ft_SPEC/G_<step>.pth`에 쌓인다. `--pretrain_D`, `--pretrain_dur`를 안 주면 MeloTTS가 기본 사전학습 파일을 내려받는다.

```bash
python melo_kr/synth_melo.py --ckpt third_party/MeloTTS/melo/logs/melo_ft_SPEC/G_2000.pth \
    --config data/melo_ft/SPEC/config.json --label SPEC
```

## 6. 채점 (기본 환경)

```bash
python melo_kr/evaluate_kr.py \
    --meta runs/xtts_attack/clean/metadata.csv runs/xtts_attack/SPEC/metadata.csv \
    --speaker-refs data/korean/clean --out runs/eval_xtts
```

| 지표 | 보호가 잘 됐으면 |
|---|---|
| SIM (ECAPA 코사인, 원본 음성 최대 5개와 평균) | 낮아짐 |
| ASR (SIM > 0.25인 비율, 공격 성공률) | 낮아짐 |
| CER (Whisper 한국어, 문장 부호·공백 제외) | 높아짐 |

원본 `evaluate.py`와 달리 테스트 문장을 화자가 직접 읽은 녹음이 없으므로, SIM은 화자의 원본 녹음 여러 개와 비교한 평균이다. 결과는 `summary.csv`를 실험 로그 시트에 옮긴다.

## 라이선스

- MeloTTS 코드·가중치: MIT
- XTTS-v2 가중치: Coqui Public Model License, 모델과 **출력물 모두 비상업 용도만**. 평가용으로만 쓰고, 앱 서비스에는 넣지 않는다.
- `bert_vits2/`(원본 저장소): AGPL-3.0. `common.py`가 그 안의 `mel_processing.py`를 가져다 쓴다.
