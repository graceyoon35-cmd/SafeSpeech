#!/usr/bin/env python3
"""MeloTTS 파인튜닝 공격용 데이터 목록을 만든다 (공격자가 보호된 음성으로 학습하는 상황).

- 음성을 44.1kHz 모노 PCM16으로 복사한다 (MeloTTS 학습 코드는 다른 샘플레이트를 받지 않는다). 원본은 건드리지 않는다.
- MeloTTS 형식 metadata.list(경로|화자|KR|문장)를 만든다.
- 대본: --transcripts CSV(파일명,문장)를 쓰거나, 없으면 --asr로 Whisper가 받아 적는다.
  보호 음성 세트에는 원본 세트에서 만든 transcripts.csv를 그대로 넣어, 두 세트의 문장을 같게 맞춘다.

    python melo_kr/prepare_finetune.py --audio-dir data/korean/clean --name clean --asr
    python melo_kr/prepare_finetune.py --audio-dir data/korean/protected/SPEC --name SPEC \
        --transcripts data/melo_ft/clean/transcripts.csv

출력: data/melo_ft/<name>/wavs/*.wav, metadata.list, transcripts.csv
"""
import argparse
import csv
import glob
import os
import sys

import soundfile as sf
import torch
import torchaudio

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MELO_SR = 44100


def main():
    p = argparse.ArgumentParser(description="MeloTTS 파인튜닝 공격용 데이터 준비")
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--name", required=True, help="세트 이름 (clean, SPEC, SafeSpeech ...)")
    p.add_argument("--speaker", default="target", help="metadata의 화자 이름")
    p.add_argument("--out-root", default=os.path.join(ROOT, "data", "melo_ft"))
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--transcripts", help="파일명,문장 CSV (머리글 file,text)")
    src.add_argument("--asr", action="store_true", help="Whisper로 대본 생성")
    p.add_argument("--whisper-model", default="medium")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    out_dir = os.path.join(args.out_root, args.name)
    if os.path.exists(out_dir):
        sys.exit(f"{out_dir} 이(가) 이미 있습니다. 다른 --name을 쓰세요.")
    wavs = sorted(glob.glob(os.path.join(args.audio_dir, "**", "*.wav"), recursive=True))
    if not wavs:
        sys.exit("WAV가 없습니다.")

    texts = {}
    if args.transcripts:
        with open(args.transcripts, encoding="utf-8-sig") as f:
            texts = {r["file"]: r["text"] for r in csv.DictReader(f)}
        lack = [os.path.basename(w) for w in wavs if os.path.basename(w) not in texts]
        if lack:
            sys.exit(f"대본이 없는 파일 {len(lack)}개: {lack[:3]}")
    else:
        import whisper
        asr = whisper.load_model(args.whisper_model, device=args.device)
        for w in wavs:
            texts[os.path.basename(w)] = asr.transcribe(w, language="ko")["text"].strip()

    os.makedirs(os.path.join(out_dir, "wavs"))
    lines = []
    for w in wavs:
        name = os.path.basename(w)
        text = texts[name].replace("|", " ").strip()
        if not text:
            print(f"[건너뜀] {name}: 대본 없음")
            continue
        a, sr = sf.read(w, dtype="float32", always_2d=True)
        x = torch.from_numpy(a.mean(1))
        if sr != MELO_SR:
            x = torchaudio.functional.resample(x, sr, MELO_SR)
        dst = os.path.abspath(os.path.join(out_dir, "wavs", name))
        sf.write(dst, x.clamp(-1, 1).numpy(), MELO_SR, subtype="PCM_16")
        lines.append(f"{dst}|{args.speaker}|KR|{text}\n")

    with open(os.path.join(out_dir, "metadata.list"), "w", encoding="utf-8") as f:
        f.writelines(lines)
    with open(os.path.join(out_dir, "transcripts.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "text"])
        for name in sorted(texts):
            w.writerow([name, texts[name]])
    print(f"완료: {len(lines)}개 → {out_dir}/metadata.list")


if __name__ == "__main__":
    main()
