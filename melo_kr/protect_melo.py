#!/usr/bin/env python3
"""MeloTTS 한국어 모델을 대리 모델로 써서 SafeSpeech(SPEC) 보호 섭동을 만든다.

원본 protect.py와 같은 것
    - 손실: pivotal objective(멜 L1 × c_mel) + β·(잡음 KL + 잡음 멜 L1), SafeSpeech 모드는 + α·(STOI + STFT)
    - 기본값: ε = 8/255, 반복 200회, 한 걸음 ε/10 (부호 기울기), β = 10, α = 0.05
    - 매 반복 같은 시드의 가우시안 잡음 z를 목표로 쓰고, 모델은 train() 상태에서 가중치를 고정
원본과 다른 것
    - 대리 모델: BERT-VITS2(영어) → MeloTTS-Korean
    - 손실이 쓰는 경로(음성 인코더 enc_q → 디코더 dec, 화자 임베딩 emb_g)만 계산한다.
      원본에서도 텍스트·BERT 경로는 손실에 들어가지 않으므로 결과는 같고, 대본·BERT 파일이 필요 없다.
    - 파일 하나씩 처리하고(패딩 없음), 섭동은 입력 샘플레이트 그대로 만든다.
      모델 입력은 미분 가능한 리샘플링으로 44.1kHz로 바꾼다.

사용 예 (저장소 루트):
    python melo_kr/protect_melo.py --input-dir data/korean/clean --out-dir data/korean/protected/SPEC
    python melo_kr/protect_melo.py --input-dir data/korean/clean --out-dir data/korean/protected/SafeSpeech --mode SafeSpeech
    python melo_kr/protect_melo.py --random-init --inputs a.wav --iters 3 --out-dir /tmp/t   # 가중치 없이 점검

출력: <out-dir>/<원본 파일명>.wav (PCM16, 원본 샘플레이트), <out-dir>/report.json
기존 폴더는 덮어쓰지 않는다. 원본 음성은 읽기만 한다.
"""
import argparse
import glob
import json
import os
import sys
import time

import soundfile as sf
import torch
import torchaudio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def list_inputs(args):
    if args.inputs:
        paths = args.inputs
    elif args.input_dir:
        paths = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.wav"), recursive=True))
    else:
        with open(args.filelist, encoding="utf-8") as f:
            paths = [line.split("|")[0].strip() for line in f if line.strip()]
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        sys.exit("처리할 WAV가 없습니다.")
    return paths


def perturb(model, hps, wav, sr, sid, args, stoi_fn):
    """wav: [1, 1, T] (입력 샘플레이트). 원본 protect.perturb()와 같은 갱신 규칙."""
    msr = hps.data.sampling_rate
    eps = args.epsilon / 255.0
    step = eps / 10.0 if args.step is None else args.step
    g = model.emb_g(sid).unsqueeze(-1)  # [1, gin, 1]

    def forward(p_wav):
        x = p_wav if sr == msr else torchaudio.functional.resample(p_wav, sr, msr)
        spec = C.linear_spec(hps, x.squeeze(1))  # [1, F, frames]
        spec_len = torch.LongTensor([spec.size(-1)]).to(spec.device)
        z, _, _, _ = model.enc_q(spec, spec_len, g=g)
        wav_hat = model.dec(z, g=g)  # 원본 is_clip=False와 같이 전체 길이를 복원
        return x, wav_hat

    p_wav = wav.clone().requires_grad_(True)
    history = []
    for it in range(args.iters):
        x, wav_hat = forward(p_wav)
        torch.manual_seed(args.seed)
        random_z = torch.randn(wav_hat.shape).to(wav_hat.device)
        loss_mel = C.reconstruction_loss(hps, x, wav_hat)  # pivotal objective
        loss_kl = C.kl_divergence(hps, wav_hat, random_z)
        loss_nr = C.reconstruction_loss(hps, wav_hat, random_z)
        loss = loss_mel + args.beta * (loss_nr + loss_kl)
        items = {"loss_mel": loss_mel.item(), "loss_nr": loss_nr.item(), "loss_kl": loss_kl.item()}
        if args.mode == "SafeSpeech":
            loss_per = C.stft_loss(wav.squeeze(1), p_wav.squeeze(1))
            if stoi_fn is not None:
                loss_per = loss_per + stoi_fn(wav.squeeze(1), p_wav.squeeze(1)).mean()
            loss = loss + args.alpha * loss_per
            items["loss_perception"] = loss_per.item()
        if it in (0, args.iters - 1) or (args.log_every and it % args.log_every == 0):
            history.append({"iter": it, **items})

        grad, = torch.autograd.grad(loss, p_wav)
        with torch.no_grad():  # 원본: p_wav - α·sign(∇) → δ를 ±ε로 자르기 → [-1, 1]로 자르기
            p_wav = p_wav - step * grad.sign()
            delta = torch.clamp(p_wav - wav, -eps, eps)
            p_wav = torch.clamp(wav + delta, -1.0, 1.0)
        p_wav.requires_grad_(True)

    p_wav = p_wav.detach()
    delta = p_wav - wav
    with torch.no_grad():
        x0, rec0 = forward(wav)
        x1, rec1 = forward(p_wav)
        torch.manual_seed(args.seed)
        stats = {
            "recon_mel_l1_clean": C.reconstruction_loss(hps, x0, rec0).item(),
            "recon_mel_l1_protected": C.reconstruction_loss(hps, x1, rec1).item(),
            "max_abs_delta": delta.abs().max().item(),
            "snr_db": (10 * torch.log10(wav.pow(2).sum() / delta.pow(2).sum().clamp_min(1e-12))).item(),
            "history": history,
        }
    return p_wav, stats


def main():
    p = argparse.ArgumentParser(description="MeloTTS-Korean 대리 모델로 SafeSpeech 보호 섭동 생성")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--input-dir", help="이 폴더 아래의 모든 .wav")
    src.add_argument("--inputs", nargs="*", help="WAV 경로 직접 지정")
    src.add_argument("--filelist", help="'경로|...' 형식 목록 파일 (첫 열만 사용)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-dir", default=os.path.join(C.REPO_ROOT, "checkpoints", "melo_kr"))
    p.add_argument("--melo-root", default=C.DEFAULT_MELO_ROOT)
    p.add_argument("--random-init", action="store_true", help="가중치 없이 코드 경로만 점검")
    p.add_argument("--mode", choices=["SPEC", "SafeSpeech"], default="SPEC")
    p.add_argument("--epsilon", type=float, default=8, help="섭동 상한 (x/255), 원본 기본 8")
    p.add_argument("--iters", type=int, default=200, help="원본 --perturbation-epochs 기본 200")
    p.add_argument("--step", type=float, default=None, help="한 걸음 크기, 기본 ε/10 (원본과 같음)")
    p.add_argument("--beta", type=float, default=10.0, help="잡음 손실 가중치, 원본 10")
    p.add_argument("--alpha", type=float, default=0.05, help="지각 손실 가중치(SafeSpeech 모드), 원본 0.05")
    p.add_argument("--speaker", default="KR", help="config의 spk2id에서 쓸 화자 이름")
    p.add_argument("--seed", type=int, default=None, help="잡음 z 시드, 기본은 config의 train.seed")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--log-every", type=int, default=0, help="N회마다 손실 기록 (0이면 처음과 끝만)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if not (args.inputs or args.input_dir or args.filelist):
        p.error("--input-dir, --inputs, --filelist 중 하나가 필요합니다")
    if os.path.exists(args.out_dir):
        sys.exit(f"{args.out_dir} 이(가) 이미 있습니다. 덮어쓰지 않으니 다른 폴더를 지정하세요.")

    C.add_melo_to_path(args.melo_root)
    model, hps = C.load_melo(args.model_dir, args.device, random_init=args.random_init)
    model.train()  # 원본 perturb()와 같이 train 상태, 가중치는 고정
    for prm in model.parameters():
        prm.requires_grad_(False)
    if args.seed is None:
        args.seed = getattr(hps.train, "seed", 52)
    spk2id = getattr(hps.data, "spk2id", None)
    spk2id = spk2id._dict if spk2id is not None and hasattr(spk2id, "_dict") else {"KR": 0}
    if args.speaker not in spk2id:
        sys.exit(f"화자 '{args.speaker}'가 config에 없습니다: {list(spk2id)}")
    sid = torch.LongTensor([spk2id[args.speaker]]).to(args.device)

    stoi_cls = None
    if args.mode == "SafeSpeech":
        try:
            from torch_stoi import NegSTOILoss as stoi_cls
        except ImportError:
            print("[알림] torch_stoi가 없어 지각 손실은 STFT 항만 씁니다.")

    paths = list_inputs(args)
    os.makedirs(args.out_dir)
    report = {"args": vars(args), "model_sampling_rate": hps.data.sampling_rate, "files": []}
    t_all = time.time()
    stoi_by_sr = {}
    for i, path in enumerate(paths, 1):
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        if audio.shape[1] > 1:
            print(f"[알림] {path}: 스테레오를 모노로 합칩니다.")
        wav = torch.from_numpy(audio.mean(1)).view(1, 1, -1).to(args.device)
        stoi_fn = None
        if stoi_cls is not None:
            stoi_fn = stoi_by_sr.setdefault(sr, stoi_cls(sample_rate=sr).to(args.device))
        t0 = time.time()
        p_wav, stats = perturb(model, hps, wav, sr, sid, args, stoi_fn)
        out = os.path.join(args.out_dir, os.path.basename(path))
        sf.write(out, p_wav.view(-1).cpu().numpy(), sr, subtype="PCM_16")
        stats.update(input=path, output=out, sample_rate=sr, audio_seconds=round(wav.shape[-1] / sr, 2),
                     seconds=round(time.time() - t0, 2))
        report["files"].append(stats)
        print(f"[{i}/{len(paths)}] {os.path.basename(path)}  복원 멜 L1 {stats['recon_mel_l1_clean']:.3f} → "
              f"{stats['recon_mel_l1_protected']:.3f}  SNR {stats['snr_db']:.1f}dB  {stats['seconds']}s")

    report["total_seconds"] = round(time.time() - t_all, 2)
    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"완료: {len(paths)}개, {report['total_seconds']}s → {args.out_dir}")


if __name__ == "__main__":
    main()
