# SafeSpeech

## SPEC 핵심 문서

| 목적 | 문서 |
|---|---|
| 최종 연구 결과·SIM/CER·품질·ε·한계 | [최종 보고서](docs/SPEC_FINAL_REPORT_20261120_KO.md) |
| 설치·실행·입출력·실제 WAV 예제 | [앱팀 실행 가이드](docs/README_SPEC_HANDOFF_20261120_KO.md) |
| 실제 앱 코드와 연결 방법·남은 작업 | [앱 연동 점검](docs/SPEC_APP_INTEGRATION_REVIEW_20261120_KO.md) |

SPEC 실행에는 검증된 WSL/CUDA 환경과 고정 Melo 자산이 필요합니다. 앱은 아직 더미 처리기를 사용하므로 연동 점검 문서의 연결 작업을 반영해야 합니다.

재현·라이선스·데이터 사용 조건·실험 프로토콜은 위 핵심 문서에 통합했습니다. docs의 Markdown은 이 3개만 유지하고 중복 문서 20개를 실제 삭제했습니다. 삭제 전 전체 본문은 [Git 기록 9882630](https://github.com/graceyoon35-cmd/SafeSpeech/tree/9882630dc3eabeeb801b1a3d1680e9e427067a58/docs)에 남아 있습니다. 원본 CSV·JSON·설정·그래프·실행 스크립트·가중치·로그·백업은 보존했습니다. 모델/하위 프로젝트 README·모델 카드는 출처·라이선스·의존성 안내를 위해 원래 위치에 유지합니다.

아래는 원본 논문의 BERT-VITS2 실행 안내입니다. 한국어 Melo SPEC 실행은 위 가이드를 따릅니다.

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
