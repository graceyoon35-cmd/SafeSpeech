#!/usr/bin/env python3
"""공격자 역할: 참조 음성(원본 또는 보호 음성)으로 XTTS-v2 한국어 제로샷 복제를 한다.

같은 문장, 같은 시드로 원본과 보호 음성을 각각 참조로 넣어 비교할 수 있게 만든다.
XTTS는 섭동 생성에 쓰지 않았으므로 이 결과는 "다른 모델에도 통하는지(전이성)"를 뜻한다.

    python melo_kr/attack_xtts.py --refs data/korean/clean --label clean --out-dir runs/xtts_attack
    python melo_kr/attack_xtts.py --refs data/korean/protected/SPEC --label SPEC --out-dir runs/xtts_attack

출력: <out-dir>/<label>/<참조파일명>/<문장번호>.wav, <out-dir>/<label>/metadata.csv
      metadata.csv 열: reference, sentence_id, text, output
같은 <out-dir>/<label>이 이미 있으면 멈춘다.
필요: pip install "coqui-tts[ko]" (README_KO.md의 XTTS 환경 참고), checkpoints/xtts_v2
"""
import argparse
import csv
import glob
import os
import sys
import time

import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description="XTTS-v2 한국어 제로샷 복제 공격")
    p.add_argument("--refs", nargs="+", required=True, help="참조 WAV 파일들 또는 폴더")
    p.add_argument("--label", required=True, help="조건 이름 (예: clean, SPEC, SafeSpeech)")
    p.add_argument("--out-dir", default=os.path.join(ROOT, "runs", "xtts_attack"))
    p.add_argument("--texts", default=os.path.join(ROOT, "melo_kr", "test_sentences_ko.txt"))
    p.add_argument("--language", default="ko")
    p.add_argument("--model-dir", default=os.path.join(ROOT, "checkpoints", "xtts_v2"))
    p.add_argument("--seed", type=int, default=1234, help="문장마다 같은 시드를 써서 조건 간 비교를 공정하게")
    p.add_argument("--temperature", type=float, default=0.75, help="XTTS 기본값")
    p.add_argument("--limit-refs", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_root = os.path.join(args.out_dir, args.label)
    if os.path.exists(out_root):
        sys.exit(f"{out_root} 이(가) 이미 있습니다. 덮어쓰지 않으니 다른 --label이나 --out-dir를 쓰세요.")

    refs = []
    for r in args.refs:
        refs += sorted(glob.glob(os.path.join(r, "**", "*.wav"), recursive=True)) if os.path.isdir(r) else [r]
    if args.limit_refs:
        refs = refs[: args.limit_refs]
    with open(args.texts, encoding="utf-8") as f:
        texts = [t.strip() for t in f if t.strip()]
    if not refs or not texts:
        sys.exit("참조 음성이나 문장이 없습니다.")

    missing = [r for r in refs if not os.path.isfile(r)]
    if missing:
        sys.exit(f"없는 참조 음성: {missing[:3]}")
    for name in ("config.json", "model.pth", "vocab.json"):
        if not os.path.isfile(os.path.join(args.model_dir, name)):
            sys.exit(f"{args.model_dir}/{name} 가 없습니다. 먼저 python melo_kr/download_models.py --xtts 를 실행하세요.")

    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts
    cfg = XttsConfig()
    cfg.load_json(os.path.join(args.model_dir, "config.json"))
    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_dir=args.model_dir, eval=True)
    model.to(args.device)

    os.makedirs(out_root)
    rows, t0 = [], time.time()
    for i, ref in enumerate(refs, 1):
        stem = os.path.splitext(os.path.basename(ref))[0]
        os.makedirs(os.path.join(out_root, stem))
        gpt_cond, spk = model.get_conditioning_latents(audio_path=[ref])
        for j, text in enumerate(texts):
            torch.manual_seed(args.seed + j)
            out = model.inference(text, args.language, gpt_cond, spk, temperature=args.temperature)
            path = os.path.join(out_root, stem, f"{j:03d}.wav")
            sf.write(path, torch.as_tensor(out["wav"]).cpu().numpy(), 24000, subtype="PCM_16")
            rows.append({"reference": ref, "sentence_id": j, "text": text, "output": path})
        print(f"[{i}/{len(refs)}] {stem}: {len(texts)}문장 합성 ({time.time() - t0:.0f}s 경과)")

    with open(os.path.join(out_root, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["reference", "sentence_id", "text", "output"])
        w.writeheader()
        w.writerows(rows)
    print(f"완료: {len(rows)}개 → {out_root}/metadata.csv")


if __name__ == "__main__":
    main()
