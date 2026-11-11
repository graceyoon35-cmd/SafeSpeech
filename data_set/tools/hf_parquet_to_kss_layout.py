#!/usr/bin/env python3
"""Hugging Face 미러(Bingsu/KSS_Dataset)의 parquet 파일을 KSS 원본과 비슷한 폴더 구조로 푼다.

필요한 패키지: pyarrow
쓰기: python hf_parquet_to_kss_layout.py --parquet-dir <parquet 폴더> --out-dir <새 폴더>

결과:
  <out>/wavs/kss_hf_00000.wav ...   parquet 안의 WAV 바이트를 그대로 저장 (변환하지 않음)
  <out>/transcript.hf.txt           경로|원문|확장문|분해문|길이|영문 (KSS 대본 형식과 같은 6칸)
  <out>/extract_report.json         shard별 행 수, parquet SHA-256, 총 행 수

주의: 미러에는 원본 파일 이름(예: 1_0470.wav)이 들어 있지 않다(audio.path가 비어 있음).
그래서 파일 이름은 shard 순서대로 붙인 번호(kss_hf_00000부터)이며, 원본 번호와의 대응은 확인할 수 없다.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def clean(s):
    return (s or "").replace("|", " ").replace("\r", " ").replace("\n", " ").strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    shards = sorted(Path(args.parquet_dir).glob("train-*.parquet"))
    if not shards:
        sys.exit("parquet 파일을 찾지 못했습니다")
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        sys.exit("출력 폴더가 이미 있고 비어 있지 않습니다: " + str(out))
    (out / "wavs").mkdir(parents=True, exist_ok=True)

    report = {"shards": [], "total_rows": 0, "non_riff_audio": 0, "empty_audio": 0}
    idx = 0
    with open(out / "transcript.hf.txt", "w", encoding="utf-8", newline="\n") as tf:
        for shard in shards:
            pf = pq.ParquetFile(shard)
            rows_here = 0
            for batch in pf.iter_batches(batch_size=200):
                for r in batch.to_pylist():
                    audio = r["audio"]["bytes"] or b""
                    if not audio:
                        report["empty_audio"] += 1
                    elif audio[:4] != b"RIFF":
                        report["non_riff_audio"] += 1
                    name = "kss_hf_%05d.wav" % idx
                    (out / "wavs" / name).write_bytes(audio)
                    dur = r.get("duration")
                    tf.write("|".join([
                        "wavs/" + name, clean(r.get("original_script")), clean(r.get("expanded_script")),
                        clean(r.get("decomposed_script")), "" if dur is None else "%.3f" % dur,
                        clean(r.get("english_translation")),
                    ]) + "\n")
                    idx += 1
                    rows_here += 1
            report["shards"].append({"file": shard.name, "rows": rows_here, "sha256": sha256_of(shard)})
            print(shard.name, rows_here, "rows")
    report["total_rows"] = idx
    (out / "extract_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("총", idx, "행, RIFF가 아닌 오디오", report["non_riff_audio"], "개, 빈 오디오", report["empty_audio"], "개")


if __name__ == "__main__":
    main()
