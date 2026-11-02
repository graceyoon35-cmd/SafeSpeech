#!/usr/bin/env python3
"""파인튜닝된 MeloTTS로 테스트 문장을 합성한다 (MeloTTS 환경에서 실행). evaluate_kr.py용 metadata.csv를 만든다.

    python melo_kr/synth_melo.py --ckpt third_party/MeloTTS/melo/logs/melo_ft_SPEC/G_2000.pth \
        --config data/melo_ft/SPEC/config.json --label SPEC --out-dir runs/melo_attack
"""
import argparse
import csv
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser(description="파인튜닝된 MeloTTS 한국어 합성")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True, help="preprocess_text.py가 만든 config.json")
    p.add_argument("--label", required=True)
    p.add_argument("--out-dir", default=os.path.join(ROOT, "runs", "melo_attack"))
    p.add_argument("--texts", default=os.path.join(ROOT, "melo_kr", "test_sentences_ko.txt"))
    p.add_argument("--speaker", default="target")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_root = os.path.join(args.out_dir, args.label)
    if os.path.exists(out_root):
        sys.exit(f"{out_root} 이(가) 이미 있습니다.")
    sys.path.insert(0, os.path.join(ROOT, "third_party", "MeloTTS"))
    from melo.api import TTS

    tts = TTS(language="KR", device=args.device, config_path=args.config, ckpt_path=args.ckpt)
    spk2id = tts.hps.data.spk2id
    if args.speaker not in spk2id:
        sys.exit(f"화자 '{args.speaker}'가 없습니다: {list(spk2id.keys())}")
    with open(args.texts, encoding="utf-8") as f:
        texts = [t.strip() for t in f if t.strip()]

    os.makedirs(out_root)
    rows = []
    for j, text in enumerate(texts):
        torch.manual_seed(args.seed + j)
        path = os.path.join(out_root, f"{j:03d}.wav")
        tts.tts_to_file(text, spk2id[args.speaker], path, quiet=True)
        rows.append({"reference": args.ckpt, "sentence_id": j, "text": text, "output": path})
    with open(os.path.join(out_root, "metadata.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["reference", "sentence_id", "text", "output"])
        w.writeheader()
        w.writerows(rows)
    print(f"완료: {len(rows)}개 → {out_root}/metadata.csv")


if __name__ == "__main__":
    main()
