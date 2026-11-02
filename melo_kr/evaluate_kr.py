#!/usr/bin/env python3
"""복제된(합성된) 한국어 음성을 채점한다: 화자 유사도(SIM) + 글자 오류율(CER).

- SIM: ECAPA-TDNN 코사인 유사도. 원본 evaluate.py와 같은 모델·같은 방식(verify_batch, 기준 0.25).
       단, 테스트 문장을 실제 화자가 읽은 녹음이 없으므로, 화자의 원본(보호 전) 음성 여러 개와 비교해 평균을 쓴다.
- ASR(공격 성공률): SIM > 0.25인 합성 음성의 비율. 낮을수록 보호가 잘 된 것.
- CER: Whisper(다국어, 한국어 지정)로 받아 적고 문장 부호·공백을 뺀 글자 단위 오류율. 높을수록 보호가 잘 된 것.

    python melo_kr/evaluate_kr.py --meta runs/xtts_attack/clean/metadata.csv runs/xtts_attack/SPEC/metadata.csv \
        --speaker-refs data/korean/clean --out runs/eval_xtts

metadata.csv 형식(attack_xtts.py, synth_melo.py가 만듦): reference, sentence_id, text, output
출력: <out>/per_file.csv, <out>/summary.csv (조건별 평균) — 실험 로그 시트에 옮겨 적는다.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import soundfile as sf
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_THRESHOLD = 0.25  # SpeechBrain verify_batch 기본값, 논문의 ASR 기준과 같음


def norm_ko(s):
    return re.sub(r"[^0-9A-Za-z가-힣]", "", s)


def cer(ref, hyp):
    import jiwer
    ref, hyp = norm_ko(ref), norm_ko(hyp)
    if not ref:
        return float("nan")
    return jiwer.cer(ref, hyp)


def load_rows(meta_paths):
    rows = []
    for m in meta_paths:
        label = os.path.basename(os.path.dirname(os.path.abspath(m)))
        with open(m, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                r["label"] = label
                rows.append(r)
    return rows


def main():
    p = argparse.ArgumentParser(description="한국어 복제 음성 채점 (SIM, ASR, CER)")
    p.add_argument("--meta", nargs="+", required=True, help="조건별 metadata.csv (조건 이름 = 그 파일이 있는 폴더 이름)")
    p.add_argument("--speaker-refs", required=True, help="실제 화자의 원본(보호 전) WAV 폴더 또는 파일들")
    p.add_argument("--max-refs", type=int, default=5, help="SIM 비교에 쓸 원본 음성 수")
    p.add_argument("--out", required=True)
    p.add_argument("--whisper-model", default="medium", help="다국어 모델 (medium.en 같은 영어 전용 모델은 안 됨)")
    p.add_argument("--skip-cer", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="모델 없이 입력 파일만 확인")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    rows = load_rows(args.meta)
    refs = sorted(glob.glob(os.path.join(args.speaker_refs, "**", "*.wav"), recursive=True)) \
        if os.path.isdir(args.speaker_refs) else [args.speaker_refs]
    refs = refs[: args.max_refs]
    missing = [r["output"] for r in rows if not os.path.isfile(r["output"])] + [r for r in refs if not os.path.isfile(r)]
    labels = sorted({r["label"] for r in rows})
    print(f"합성 음성 {len(rows)}개 (조건: {', '.join(labels)}), 비교용 원본 음성 {len(refs)}개")
    if missing:
        sys.exit(f"없는 파일 {len(missing)}개: {missing[:3]}")
    if not refs:
        sys.exit("--speaker-refs에 WAV가 없습니다.")
    if args.dry_run:
        print("입력 확인 끝 (--dry-run)")
        return
    if os.path.exists(args.out):
        sys.exit(f"{args.out} 이(가) 이미 있습니다. 다른 폴더를 지정하세요.")

    from speechbrain.inference.speaker import SpeakerRecognition
    sv = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                         savedir=os.path.join(ROOT, "encoders", "spkrec-ecapa-voxceleb"),
                                         run_opts={"device": args.device})

    def load(path):
        a, sr = sf.read(path, dtype="float32", always_2d=True)
        return sv.audio_normalizer(torch.from_numpy(a), sr).unsqueeze(0)  # 원본 compute_sim과 같은 전처리

    ref_audio = [load(r) for r in refs]
    asr_model = None
    if not args.skip_cer:
        import whisper
        asr_model = whisper.load_model(args.whisper_model, device=args.device)

    os.makedirs(args.out)
    out_rows = []
    with torch.no_grad():
        for i, r in enumerate(rows, 1):
            syn = load(r["output"])
            scores = [sv.verify_batch(ra, syn)[0][0].item() for ra in ref_audio]
            sim = sum(scores) / len(scores)
            res = {**r, "sim": round(sim, 4), "attack_success": int(sim > SIM_THRESHOLD)}
            if asr_model is not None:
                hyp = asr_model.transcribe(r["output"], language="ko")["text"]
                res.update(transcript=hyp.strip(), cer=round(cer(r["text"], hyp), 4))
            out_rows.append(res)
            if i % 20 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}")

    with open(os.path.join(args.out, "per_file.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    summary = []
    for lab in labels:
        sel = [r for r in out_rows if r["label"] == lab]
        s = {"조건": lab, "합성 수": len(sel),
             "SIM": round(sum(r["sim"] for r in sel) / len(sel), 4),
             "ASR(%)": round(100 * sum(r["attack_success"] for r in sel) / len(sel), 1)}
        if asr_model is not None:
            vals = [r["cer"] for r in sel if r["cer"] == r["cer"]]
            s["CER(%)"] = round(100 * sum(vals) / max(len(vals), 1), 1)
        summary.append(s)
        print("  ".join(f"{k} {v}" for k, v in s.items()))
    with open(os.path.join(args.out, "summary.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "speaker_refs": refs, "summary": summary}, f, ensure_ascii=False, indent=2)
    print(f"→ {args.out}/summary.csv")


if __name__ == "__main__":
    main()
