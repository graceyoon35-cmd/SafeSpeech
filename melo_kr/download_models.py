#!/usr/bin/env python3
"""MeloTTS-Korean, XTTS-v2 가중치를 Hugging Face에서 받는다. 이미 있는 파일은 건드리지 않는다.

    python melo_kr/download_models.py --melo          # checkpoints/melo_kr/  (config.json, checkpoint.pth)
    python melo_kr/download_models.py --xtts          # checkpoints/xtts_v2/  (약 1.9GB)
    python melo_kr/download_models.py --melo --xtts

라이선스: MeloTTS = MIT, XTTS-v2 = Coqui Public Model License (모델과 출력물 모두 비상업 용도만).
"""
import argparse
import os
import shutil
import sys

from huggingface_hub import hf_hub_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGETS = {
    "melo": ("myshell-ai/MeloTTS-Korean", ["config.json", "checkpoint.pth"], "checkpoints/melo_kr"),
    "xtts": ("coqui/XTTS-v2", ["config.json", "model.pth", "vocab.json", "speakers_xtts.pth"], "checkpoints/xtts_v2"),
}


def fetch(repo, files, out_dir):
    out_dir = os.path.join(ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for name in files:
        dst = os.path.join(out_dir, name)
        if os.path.exists(dst):
            print(f"[있음] {dst} (건너뜀)")
            continue
        try:
            src = hf_hub_download(repo, name)
        except Exception as e:  # speakers_xtts.pth는 기본 화자 목록이라 없어도 복제에는 지장 없음
            if name == "speakers_xtts.pth":
                print(f"[건너뜀] {repo}/{name}: {type(e).__name__} (제로샷 복제에는 필요 없음)")
                continue
            raise
        shutil.copyfile(src, dst + ".part")
        os.replace(dst + ".part", dst)
        print(f"[받음] {repo}/{name} → {dst} ({os.path.getsize(dst) / 1e6:.1f}MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--melo", action="store_true")
    p.add_argument("--xtts", action="store_true")
    a = p.parse_args()
    if not (a.melo or a.xtts):
        sys.exit("--melo 또는 --xtts를 지정하세요.")
    for key in ("melo", "xtts"):
        if getattr(a, key):
            fetch(*TARGETS[key])


if __name__ == "__main__":
    main()
