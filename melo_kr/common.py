"""melo_kr 스크립트 공통 함수.

- 저장소 루트와 MeloTTS 코드 위치를 sys.path에 넣는다.
- MeloTTS 설정/가중치를 읽는다 (melo.utils는 모든 언어 모듈을 import해서 쓰지 않는다).
- 손실 계산은 SafeSpeech 원본(protect.py, bert_vits2/mel_processing.py)과 같은 함수를 쓴다.
"""
import json
import os
import sys
import types

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MELO_ROOT = os.path.join(REPO_ROOT, "third_party", "MeloTTS")

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# SafeSpeech 원본과 같은 스펙트로그램·멜 함수 (librosa 0.10 키워드 인자 사용)
from bert_vits2.mel_processing import mel_spectrogram_torch, spectrogram_torch  # noqa: E402


def add_melo_to_path(melo_root):
    melo_root = os.path.abspath(melo_root)
    if not os.path.isfile(os.path.join(melo_root, "melo", "models.py")):
        sys.exit(f"MeloTTS 코드를 찾지 못했습니다: {melo_root}\n"
                 "  git clone https://github.com/myshell-ai/MeloTTS third_party/MeloTTS")
    if melo_root not in sys.path:
        sys.path.insert(0, melo_root)


def to_ns(obj):
    """dict → 속성으로 읽는 객체 (Melo의 HParams와 같은 용도). model 항목은 **kwargs로도 풀 수 있게 dict를 보관."""
    if isinstance(obj, dict):
        ns = types.SimpleNamespace(**{k: to_ns(v) for k, v in obj.items()})
        ns._dict = obj
        return ns
    return obj


def load_hparams(path):
    with open(path, encoding="utf-8") as f:
        return to_ns(json.load(f))


def build_melo(hps, device):
    """Melo api.py와 같은 방식으로 SynthesizerTrn을 만든다."""
    from melo.models import SynthesizerTrn
    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        num_tones=hps.num_tones,
        num_languages=hps.num_languages,
        **hps.model._dict,
    ).to(device)
    return model


def load_melo(model_dir, device, random_init=False, seed=52):
    """model_dir: MeloTTS-Korean의 config.json, checkpoint.pth가 있는 폴더."""
    if random_init:
        # 가중치 없이 코드 경로만 점검: Melo 학습 기본 설정 + 전체 기호표
        import importlib
        sym = importlib.import_module("melo.text.symbols")  # melo.text.symbols는 같은 이름의 리스트에 가려지므로 모듈로 직접 가져온다
        melo_pkg = os.path.dirname(os.path.dirname(sym.__file__))
        with open(os.path.join(melo_pkg, "configs", "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        cfg.update(symbols=sym.symbols, num_tones=sym.num_tones, num_languages=sym.num_languages)
        cfg["data"]["spk2id"] = {"KR": 0}
        hps = to_ns(cfg)
        torch.manual_seed(seed)
        model = build_melo(hps, device)
        print("[주의] 무작위 가중치 모델: 코드 경로 점검용이며 보호 효과와 무관합니다.")
        return model, hps

    cfg_path = os.path.join(model_dir, "config.json")
    ckpt_path = os.path.join(model_dir, "checkpoint.pth")
    for p in (cfg_path, ckpt_path):
        if not os.path.isfile(p):
            sys.exit(f"{p} 가 없습니다. 먼저 python melo_kr/download_models.py --melo 를 실행하세요.")
    hps = load_hparams(cfg_path)
    model = build_melo(hps, device)
    state = torch.load(ckpt_path, map_location="cpu")
    state = state.get("model", state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    need = [k for k in missing if k.startswith(("enc_q.", "dec.", "emb_g."))]
    if need:
        sys.exit(f"체크포인트에 섭동 계산에 필요한 가중치가 없습니다: {need[:5]} ...")
    if missing or unexpected:
        print(f"[알림] 사용하지 않는 부분의 키 차이: 없음 {len(missing)}개, 남음 {len(unexpected)}개")
    return model, hps


# ---- SafeSpeech protect.py와 같은 손실 ----

def mel(hps, wav):
    d = hps.data
    return mel_spectrogram_torch(wav.squeeze(1).float(), d.filter_length, d.n_mel_channels, d.sampling_rate,
                                 d.hop_length, d.win_length, d.mel_fmin, d.mel_fmax)


def reconstruction_loss(hps, wav, wav_hat):
    """pivotal objective: 멜 L1 × c_mel"""
    return F.l1_loss(mel(hps, wav), mel(hps, wav_hat)) * hps.train.c_mel


def kl_divergence(hps, x_hat, z):
    return F.kl_div(F.log_softmax(mel(hps, x_hat), dim=-1), F.softmax(mel(hps, z), dim=-1), reduction="batchmean")


def stft_loss(wav, p_wav):
    """원본 compute_stft와 같은 값 (창 없음, n_fft 2048, hop 512, L2)."""
    s = lambda x: torch.view_as_real(torch.stft(x, n_fft=2048, win_length=2048, hop_length=512, return_complex=True))
    return torch.norm(s(p_wav) - s(wav), p=2)


def linear_spec(hps, wav):
    d = hps.data
    return spectrogram_torch(wav, d.filter_length, d.sampling_rate, d.hop_length, d.win_length, center=False)
