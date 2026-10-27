# SafeSpeech

## 팀 개발 현황 및 시작 안내 (2026-10-27)

SafeSpeech는 음성에 작은 섭동을 적용하여 원치 않는 음성 합성 학습으로부터 보호하는
연구 프로젝트입니다. 이 저장소에서는 원본 연구 코드 위에 공통 환경과 실험 기반을
추가했습니다. **전체 보호 성능·논문 재현·모델 학습은 아직 검증하지 않았습니다.**

전체 이력과 완료/미완료 범위는 [통합 개발 보고서](docs/DEVELOPMENT_PROGRESS_KO.md)를
먼저 읽어 주세요. 현재 공유 브랜치는 `feature/cpu-experiment-pipeline`이며
`feature/integration-setup`의 WSL 작업을 포함합니다. main 병합 여부와 별개로 이 브랜치를
선택해야 아래 추가 기능을 사용할 수 있습니다.

| 환경 | 실행 확인한 범위 | 시작 문서 |
| --- | --- | --- |
| Windows + WSL2 Ubuntu 22.04 / NVIDIA CUDA | GPU·오디오 검증, 단일 음성 SPEC 1회·10회 | [WSL 시작 안내](docs/QUICKSTART_KO.md) |
| Mac / Python 3.10 / CPU | 독립 STOI, Gaussian 변환, manifest runner, JSON·해시, 테스트 9개 | [CPU 실험 안내](docs/EXPERIMENT_PIPELINE_KO.md) |

### 새 팀원: Mac CPU에서 공통 실험 시작

Python 3.10이 설치된 환경의 명령입니다. 기존 가상환경이 있으면 재생성하지 않습니다.

```bash
git clone --branch feature/cpu-experiment-pipeline https://github.com/graceyoon35-cmd/SafeSpeech.git
cd SafeSpeech
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-evaluation-smoke.txt
.venv/bin/python -m pip check
.venv/bin/python scripts/smoke_evaluation.py
.venv/bin/python scripts/run_experiment.py experiments/gaussian_smoke.json
.venv/bin/python -m unittest discover -s scripts -p 'test_*.py' -v
```

CPU 경로에는 PyTorch·CUDA·모델 가중치가 필요하지 않습니다. 원본과 Gaussian 대조군의
STOI는 각각 약 1.000000 / 0.970284입니다(seed 42, std 0.005, 예제 WAV).
결과 JSON과 변환 WAV는 Git에서 제외된 `runs/experiment_*`에 저장됩니다.
manifest는 원본·보호·합성 역할을 구분하며, 현재 예제는 원본 대조군만 실행합니다.
같은 sample rate·샘플 수의 1초 이상 mono 음성을 요구하고 자동 정렬은 하지 않습니다.

### Windows CUDA 개발과 실제 SPEC 연결

WSL에서 섭동 생성을 개발할 팀원은 [설치·실행 순서](docs/QUICKSTART_KO.md)와
[기존 GPU·SPEC 검증 결과](docs/SETUP_VALIDATION_SUMMARY_KO.md)를 확인합니다.
CPU 의존성 목록과 WSL 목록은 별도 가상환경에서 사용합니다. 아래 원본 연구용 전체
설치 명령을 CPU 환경에 추가 설치하지 마세요. Mac에서 CUDA smoke를 실행하지 않습니다.

실제 WSL SPEC 출력은 Mac에 없으므로 보호 음성 STOI는 미측정입니다.
[파일 가져오기 안내](docs/SPEC_PAIR_IMPORT_KO.md)를 따라 원본·보호 음성·생성 보고서를
확보한 후 `experiments/spec_gaussian_pending.json`을 실행할 수 있습니다.
현재는 해당 manifest가 파일 없음 오류를 내는 것이 정상입니다.

### 문서 읽는 순서와 현재 제약

1. [통합 개발 보고서](docs/DEVELOPMENT_PROGRESS_KO.md): 전체 구조·이력·문제·담당자 연동
2. 환경 선택: [Mac CPU 실험](docs/EXPERIMENT_PIPELINE_KO.md) 또는 [WSL CUDA 설정](docs/QUICKSTART_KO.md)
3. [WSL 검증 총정리](docs/SETUP_VALIDATION_SUMMARY_KO.md)와 [상세 이력](docs/WSL_SETUP_KO.md)
4. [초기 코드 점검](docs/COMMON_FOUNDATION_REVIEW_KO.md)과 [실제 SPEC 연결 준비](docs/SPEC_PAIR_IMPORT_KO.md)

기존 코드에는 화자 임베딩 누락, WER 원본 행 건너뛰기, 학습 모델 파일 수 검사 문제가
확인되어 있습니다. 담당자 수정·검증이 필요하며 이번 공통 기반에서 해결하지 않았습니다.
SIM/MCD/WER 전체 실행, 추가 강건성 실험, 모델 학습·합성은 후속 작업입니다.
모델·가상환경·로컬 결과는 clone으로 전달되지 않습니다.

## 원본 연구 프로젝트 안내

아래는 기존 연구 소개·설치·데이터·보호·학습·평가·인용 안내를 보존한 내용입니다.
원본 예제 명령이 팀의 실행 검증 완료를 뜻하지 않으며, 새 팀원은 위 환경별 안내부터
따릅니다.


This is the source code of our paper "SafeSpeech: Robust and Universal Voice Protection Against Malicious Speech Synthesis" in the USENIX Security 2025. We propose a proactive framework named SafeSpeech that utilizes the pivotal objective optimization and Speech PErturbative Concealment (SPEC) techniques to prevent publicly uploaded speeches from malicious speech synthesis.

\[[Paper](https://arxiv.org/pdf/2504.09839)\] \[[Demo Page](https://wxzyd123.github.io/safespeech)\]




## Setup
We test our experiments on Ubuntu 20.04. At least one GPU is needed.

The required dependencies can be installed by running the following:

```bash
conda create --name safespeech python=3.10
conda activate safespeech
pip install -r requirements.txt
sudo apt install ffmpeg

# For CUDA 11.8
pip install torch==2.0.0 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.4
pip install torch==2.3.1 torchaudio==2.3.1
```



## Pre-trained Models

Before fine-tuning BERT-VITS2, you should download the pre-trained checkpoints. Assuming the checkpoint folder is `checkpoints`.

- BERT-VITS2: You can download checkpoints [here](https://huggingface.co/OedoSoldier/Bert-VITS2-2.3/tree/main) to `checkpoints/base_models`;

- DeBERTa: You can download pre-trained BERT models to `bert/deberta-v3-large`. You can download it [here](https://huggingface.co/microsoft/deberta-v3-large).

- WavLM: BERT-VITS2 employs the pre-trained WavLM to enhance the timbre similarity. You can download it [here](https://huggingface.co/microsoft/wavlm-base-plus) to `bert_vits2/slm/wavlm-base-plus`.
- ECAPA-TDNN: We utilize the ECAPA-TDNN encoder as the timbre extractor. You can download it [here](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) to `encoders/spkrec-ecapa-voxceleb`.

Alternatively, you can download models by this command:
```
python download_models.py
```



## 1. Datasets

In our paper, we have conducted our experiments on two datasets.

For [LibriTTS](http://www.openslr.org/60/), we download the train-clean-100.tar.gz subset and select speaker 5339. For [CMU ARCTIC](http://festvox.org/cmu_arctic/packed/), we select 100 sentences from each speaker. You can use your customized voices to achieve protection as follows, and we use the LibriTTS dataset as an example:

1. Move the dataset to `data/{dataset_name}`, the structure of the dataset can be `data/{dataset_name}/{speaker-id}/{name}.wav`.
  
2. The training dataset is indexed by a file list. The initial file list is like `{path}|{speaker-id}|{language}|{text}`, such as the provided`filelists/libritts_train_text.txt`. Then convert the file list to the correct form that BERT-VITS2 can accept by:
   ```bash
   python preprocess_text.py --file-path filelists/libritts_train_text.txt
   ```
   Then the processed and cleaned file list can be found at `filelists/libritts_train_text.txt.cleaned`, which can index the dataset.


**Remark**: We provide the LibriTTS in `data/LibriTTS` and its corresponding file lists in `filelists`, you can use them directly without preprocessing.



## 2. Protect

After obtaining the dataset and successfully running the model, you can protect the dataset by SafeSpeech.

1. Get BERT files from DeBERTa-V3:
   ```bash
   python bert_gen.py --dataset LibriTTS --mode clean
   ```

2. Generate perturbation:
   ```
   python protect.py --dataset LibriTTS \
                     --model BERT_VITS2 \
                     --batch-size 27 \
                     --gpu 0 \
                     --mode SPEC \
                     --checkpoint-path checkpoints \
                     --epsilon 8 \
                     --perturbation-epochs 200
   ```

    Basic arguments:

   - `--dataset`: which dataset to protect. Default: LibriTTS
   - `--model`: the surrogate model. Default: BERT_VITS2
   - `--batch-size`: the batch size of training and perturbation generation. Default: 27
   - `--gpu`: use which GPU. Default:0
   - `--mode`: the protection mode of the SafeSpeech. Default: SPEC
   - `--checkpoints-path`: the storing dir of the checkpoints. Default: checkpoints 
   - `--epsilon`: the perturbation radius boundary. Default:8
   - `--perturbation-epochs`: the optimization iterations of perturbation. Default: 200

    For data protection, we provide two protective modes: [`SPEC` and `SafeSpeech`]. The mode of `SPEC` implements the proposed method in Section 4.1, while `SafeSpeech` combing the introduced perceptual loss. For more protective methods, please refer to their open-source repositories: [AdvPoison](https://arxiv.org/abs/2106.10807), [SEP](https://github.com/Sizhe-Chen/SEP), [Unlearnable Examples/PTA](https://github.com/HanxunH/Unlearnable-Examples), [AttackVC](https://github.com/cyhuang-tw/attack-vc), and [AntiFake](https://github.com/WUSTL-CSPL/AntiFake).

    In this experiment, large GPU memories are needed. We set the batch size as 27 on an A800 GPU with 80GB memory. 

3. After generating the perturbation, you can save the generated audio by:

    ```bash
    python save_audio.py --mode clean --batch-size 27
    ```

    or

    ```bash
    python save_audio.py --mode SPEC --batch-size 27
    ```

    The saved dataset can be found at `data/{dataset}/protected/{mode}`.



## 3. Fine-tuning

You can fine-tune the model on the original dataset or protected dataset.

1. Before training, the BERT file should be generated by:
    ```
    python bert_gen.py --dataset LibriTTS --mode SPEC
    ```

2. Fine-tuning on the original dataset without perturbation:
    ```bash
    python train.py --mode clean --batch-size 64
    ```

3. Fine-tuning on the protected dataset by SafeSpeech:
    ```bash
    python train.py --mode SPEC --batch-size 64
    ```

After fine-tuning, the code will generate the checkpoint at `checkpoints/{dataset}`.



## 4. Evaluation

You can evaluate the synthetic quality by this command:
```bash
python evaluate.py --mode SPEC
```



## **Acknowledgment**

- [BERT-VITS2](https://github.com/fishaudio/Bert-VITS2)
- [Unlearnable Examples](https://github.com/HanxunH/Unlearnable-Examples)



## Citation

If you find our repository helpful, please consider citing our work in your research or project.

```
@inproceedings{zhang2025safespeech,
  author = {Zhang, Zhisheng and Wang, Derui and Yang, Qianyi and Huang, Pengyang and Pu, Junhan and Cao, Yuxin and Ye, Kai and Hao, Jie and Yang, Yixian},
  title = {SafeSpeech: Robust and Universal Voice Protection Against Malicious Speech Synthesis},
  booktitle = {34th USENIX Security Symposium (USENIX Security 25)},
  year = {2025},
  address = {Seattle, WA, USA}
}
```




## Disclaimer
SafeSpeech is utilized for personal sensitive information protection. If users use this tool to disrupt legitimate and beneficial speech synthesis, all the resulting consequences shall have nothing to do with the publishers and designers of SafeSpeech!
