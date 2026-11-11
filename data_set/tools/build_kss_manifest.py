#!/usr/bin/env python3
"""KSS Dataset에서 일부 발화를 골라 manifest, 분할, 전달 폴더를 만든다.

표준 라이브러리만 쓴다. 원본 KSS 폴더는 읽기만 하고 수정하지 않는다.

만들기:
  python build_kss_manifest.py --kss-root <KSS 폴더> --out-dir <새 폴더> --n 30

다시 확인하기 (전달 폴더):
  python build_kss_manifest.py --verify <전달 폴더>

SPEC 실행 결과 확인하기 (clean, spec_1, spec_10 폴더가 같은 발화 ID로 있는지):
  python build_kss_manifest.py --check-run <전달 폴더> <runs/korean_melo_spec_...>
"""
import argparse
import array
import csv
import hashlib
import json
import math
import random
import re
import shutil
import sys
import unicodedata
import wave
from pathlib import Path

DATASET_SOURCE = "KSS"
DATASET_NAME = "KSS Dataset (Korean Single Speaker Speech Dataset, Kyubyong Park)"
DATASET_URL = "https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset"
LICENSE = "CC BY-NC-SA 4.0"
SPEAKER_ID = "kss_f1"
SPLITS = ("train", "validation", "test")
HANGUL = re.compile(r"[가-힣]")
REQUIRED_COLUMNS = ["utterance_id", "speaker_id", "wav_path", "transcript", "split",
                    "sample_rate", "duration", "sha256", "dataset_source", "license", "usage_status"]
EXTRA_COLUMNS = ["channels", "bit_depth", "transcript_expanded", "source_relpath", "transcript_line", "notes"]


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def norm_text(s):
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"[\s.,!?\"'`]+", "", s)


def parse_transcript(path):
    """줄 형식: 경로|원문|확장문|분해문|길이(초)|영문 번역. 앞 두 칸만 꼭 필요하다."""
    entries = []
    with open(path, encoding="utf-8-sig") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\r\n")
            if not line.strip():
                continue
            cols = line.split("|")
            if len(cols) < 2:
                continue
            declared = None
            if len(cols) >= 5:
                try:
                    declared = float(cols[4])
                except ValueError:
                    declared = None
            entries.append({
                "relpath": cols[0].strip(),
                "text": unicodedata.normalize("NFC", cols[1].strip()),
                "expanded": unicodedata.normalize("NFC", cols[2].strip()) if len(cols) > 2 else "",
                "declared": declared,
                "line": lineno,
            })
    return entries


def analyze_wav(path):
    """(정보, 거부 사유). 정상이면 사유는 None."""
    try:
        with wave.open(str(path), "rb") as w:
            ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(n)
    except (wave.Error, EOFError, OSError) as e:
        return None, "corrupt:" + type(e).__name__
    if sw != 2:
        return None, "not_pcm16"
    if ch != 1:
        return None, "not_mono"
    if n == 0 or len(raw) != n * ch * sw:
        return None, "truncated_or_empty"
    a = array.array("h")
    a.frombytes(raw)
    if sys.byteorder == "big":
        a.byteswap()
    peak = max(map(abs, a))
    rms = math.sqrt(sum(x * x for x in a) / len(a))
    clipped = sum(1 for x in a if abs(x) >= 32767) / len(a)
    if peak < 500 or rms < 50:
        return None, "silent_or_too_quiet"
    if clipped > 0.001:
        return None, "clipped"
    return {"sample_rate": sr, "channels": ch, "bit_depth": sw * 8, "duration": n / sr,
            "peak": peak, "rms": rms}, None


def make_uid(relpath):
    stem = re.sub(r"[^0-9A-Za-z_-]", "_", Path(relpath).stem)
    return stem if stem.startswith("kss_") else "kss_" + stem


def split_counts(n, val_ratio, test_ratio):
    n_test = max(1, round(n * test_ratio))
    n_val = max(1, round(n * val_ratio))
    return n - n_val - n_test, n_val, n_test


def build(args):
    root = Path(args.kss_root)
    if args.transcript:
        tfile = Path(args.transcript)
    else:
        found = sorted(root.rglob("transcript*.txt"))
        if not found:
            sys.exit("transcript 파일을 찾지 못했습니다: --transcript로 직접 지정하세요")
        tfile = found[-1]
    base = tfile.parent
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        sys.exit("출력 폴더가 이미 있고 비어 있지 않습니다. 다른 이름을 쓰세요: " + str(out))
    if args.n < 3:
        sys.exit("--n은 3 이상이어야 합니다 (train, validation, test에 하나씩)")

    entries = parse_transcript(tfile)
    entries.sort(key=lambda e: e["relpath"])
    rng = random.Random(args.seed)
    rng.shuffle(entries)

    accepted, rejects = [], []
    seen_text, seen_sha = set(), set()
    scanned = 0
    for e in entries:
        if len(accepted) >= args.n:
            break
        src = base / e["relpath"]
        if e["declared"] is not None and not (args.min_sec - 0.5 <= e["declared"] <= args.max_sec + 0.5):
            continue
        scanned += 1
        if not e["text"] or not HANGUL.search(e["text"]):
            rejects.append((e["relpath"], "bad_transcript"))
            continue
        if not src.is_file():
            rejects.append((e["relpath"], "missing_file"))
            continue
        info, why = analyze_wav(src)
        if why:
            rejects.append((e["relpath"], why))
            continue
        if not (args.min_sec <= info["duration"] <= args.max_sec):
            rejects.append((e["relpath"], "duration_out_of_range"))
            continue
        key = norm_text(e["text"])
        digest = sha256_of(src)
        if key in seen_text or digest in seen_sha:
            rejects.append((e["relpath"], "duplicate_text_or_audio"))
            continue
        seen_text.add(key)
        seen_sha.add(digest)
        notes = ""
        if e["declared"] is not None and abs(e["declared"] - info["duration"]) > 0.3:
            notes = "declared_duration_differs:%.2f" % e["declared"]
        e.update(src=src, info=info, sha=digest, notes=notes,
                 uid=make_uid(e["relpath"]))
        accepted.append(e)

    if len(accepted) < args.n:
        print("경고: 조건에 맞는 발화가 %d개뿐입니다 (요청 %d개)" % (len(accepted), args.n))
    if len(accepted) < 3:
        sys.exit("조건에 맞는 발화가 3개 미만이라 분할할 수 없습니다")
    if len({e["uid"] for e in accepted}) != len(accepted):
        sys.exit("발화 ID가 겹칩니다")

    n_train, n_val, n_test = split_counts(len(accepted), args.val_ratio, args.test_ratio)
    for i, e in enumerate(accepted):
        e["split"] = "train" if i < n_train else "validation" if i < n_train + n_val else "test"

    usage_status = "confirmed_noncommercial_research" if args.usage_confirmed else "pending_team_confirmation"
    rows = []
    for split in SPLITS:
        (out / split / "wavs").mkdir(parents=True, exist_ok=True)
    for e in sorted(accepted, key=lambda x: (SPLITS.index(x["split"]), x["uid"])):
        dst = out / e["split"] / "wavs" / (e["uid"] + ".wav")
        shutil.copyfile(e["src"], dst)
        if sha256_of(dst) != e["sha"]:
            sys.exit("복사본의 해시가 원본과 다릅니다: " + str(dst))
        rows.append({
            "utterance_id": e["uid"], "speaker_id": SPEAKER_ID,
            "wav_path": dst.relative_to(out).as_posix(), "transcript": e["text"], "split": e["split"],
            "sample_rate": e["info"]["sample_rate"], "duration": "%.3f" % e["info"]["duration"],
            "sha256": e["sha"], "dataset_source": DATASET_SOURCE, "license": LICENSE,
            "usage_status": usage_status, "channels": e["info"]["channels"],
            "bit_depth": e["info"]["bit_depth"], "transcript_expanded": e["expanded"],
            "source_relpath": e["relpath"], "transcript_line": e["line"], "notes": e["notes"],
        })

    cols = REQUIRED_COLUMNS + EXTRA_COLUMNS
    with open(out / "manifest.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    meta = {
        "dataset": {"name": DATASET_NAME, "url": DATASET_URL, "license": LICENSE,
                    "usage_status": usage_status, "speaker_id": SPEAKER_ID,
                    "actual_source": args.source_note or "(기록 안 함)"},
        "transcript_file": tfile.name, "transcript_sha256": sha256_of(tfile),
        "parameters": {"seed": args.seed, "n": args.n, "min_sec": args.min_sec, "max_sec": args.max_sec,
                       "val_ratio": args.val_ratio, "test_ratio": args.test_ratio},
        "counts": {s: sum(1 for r in rows if r["split"] == s) for s in SPLITS},
        "utterances": rows,
    }
    (out / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    (out / "splits").mkdir()
    for s in SPLITS:
        ids = [r["utterance_id"] for r in rows if r["split"] == s]
        (out / "splits" / (s + ".txt")).write_text("\n".join(ids) + "\n", encoding="utf-8")
        # BOM을 붙인다: 엑셀에서 한글이 깨지지 않고, MeloTTS prepare_finetune.py도 utf-8-sig로 읽는다
        with open(out / s / "transcripts.csv", "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "text"])
            for r in rows:
                if r["split"] == s:
                    w.writerow([r["utterance_id"] + ".wav", r["transcript"]])

    with open(out / "transcripts_all.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utterance_id", "split", "transcript", "duration", "wav_path"])
        for r in rows:
            w.writerow([r["utterance_id"], r["split"], r["transcript"], r["duration"], r["wav_path"]])
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "manifest"
        ws.append(cols)
        for r in rows:
            ws.append([r[c] for c in cols])
        for letter, width in zip("ABCDEFGHIJK", (16, 10, 34, 50, 12, 11, 9, 66, 9, 16, 30)):
            ws.column_dimensions[letter].width = width
        ws.freeze_panes = "A2"
        wb.save(out / "manifest.xlsx")
    except ImportError:
        pass  # openpyxl이 없으면 xlsx는 만들지 않는다. csv와 json은 항상 만든다
    with open(out / "transcript_check.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utterance_id", "wav_path", "transcript", "heard_and_matches(예/아니오)", "checked_by", "date", "memo"])
        for r in rows:
            w.writerow([r["utterance_id"], r["wav_path"], r["transcript"], "", "", "", ""])
    with open(out / "rejects.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_relpath", "reason"])
        w.writerows(rejects)

    spec_utts = []
    for r in rows:
        spec_utts.append({
            "utterance_id": r["utterance_id"], "input_kind": "human_recording", "language": "ko",
            "speaker_id": r["speaker_id"], "model_speaker": "KR", "split": r["split"],
            "text": r["transcript"], "transcript_verified": bool(args.transcripts_verified),
            "transcript_reference": "KSS %s line %s, 사람 대조 기록은 transcript_check.csv" % (tfile.name, r["transcript_line"]),
            "original_path": r["wav_path"], "original_sha256": r["sha256"], "sample_rate": int(r["sample_rate"]),
            "source": DATASET_NAME + ", " + DATASET_URL,
            "research_use_permitted": bool(args.usage_confirmed),
            "permission_scope": "spec_and_finetuning",
            "permission_reference": "공개 데이터셋 라이선스 " + LICENSE + " (개별 동의서 없음), 팀 확인 상태: " + usage_status,
        })
    (out / "spec_input.json").write_text(
        json.dumps({"schema_version": 1, "utterances": spec_utts}, ensure_ascii=False, indent=2), encoding="utf-8")

    write_readme(out, meta, rows, rejects, scanned, tfile, args)
    print("완료: %s" % out)
    print("발화 %d개 (train %d, validation %d, test %d), 검사한 파일 %d개, 제외 %d개" % (
        len(rows), n_train, n_val, n_test, scanned, len(rejects)))
    print("사용 상태: %s / 대본 사람 검수: %s" % (usage_status, "완료로 표시" if args.transcripts_verified else "아직 안 함"))


def write_readme(out, meta, rows, rejects, scanned, tfile, args):
    reasons = {}
    for _, why in rejects:
        key = why.split(":")[0]
        reasons[key] = reasons.get(key, 0) + 1
    reason_lines = "\n".join("- %s: %d개" % (k, v) for k, v in sorted(reasons.items())) or "- 없음"
    sr = sorted({r["sample_rate"] for r in rows})
    text = f"""# KSS 부분 데이터 전달 폴더

만든 도구: `build_kss_manifest.py` (1학년 A, 데이터 담당)

## 출처와 이용 조건

- 데이터: {DATASET_NAME}
- 원본 배포처: {DATASET_URL}
- 실제로 받은 곳: {meta['dataset']['actual_source']}
- 라이선스: {LICENSE} (비상업 목적만, 출처 표시, 변경본은 같은 조건으로 공유)
- 이용 상태: `{meta['dataset']['usage_status']}`
  - `pending_team_confirmation`은 팀이 음성 복제 학습에 쓰는 것을 확인하기 전이라는 뜻입니다.
  - 확인이 끝나면 `--usage-confirmed`로 다시 만들거나 manifest의 값을 팀 기록과 함께 바꿉니다.
- 화자는 전문 성우 1명입니다. 화자 ID는 `{SPEAKER_ID}`로 통일했습니다.
- 원본 KSS 폴더는 읽기만 했고 수정하지 않았습니다.

## 고른 방법

- 대본 파일: `{tfile.name}` (SHA-256은 manifest.json에 기록)
- 무작위 시드 {args.seed}로 섞은 순서대로 검사해서 조건에 맞는 {len(rows)}개를 골랐습니다. 같은 시드와 같은 KSS 파일이면 같은 결과가 나옵니다.
- 길이 {args.min_sec}초에서 {args.max_sec}초 사이, 모노, 16비트 PCM만 받았습니다.

## 전처리와 검증 (자동)

- WAV가 열리는지, 프레임 수가 헤더와 맞는지
- 모노, 16비트 PCM인지
- 무음이거나 너무 작은 소리인지, 깨질 만큼 큰 소리(클리핑)인지
- 대본이 비어 있거나 한글이 없는지
- 같은 대본이나 같은 음성 파일(SHA-256이 같음)이 겹치는지
- 복사한 파일의 SHA-256이 원본과 같은지

검사한 파일 {scanned}개 중 제외된 파일 사유 (rejects.csv):
{reason_lines}

## 분할

| 분할 | 발화 수 |
|---|---:|
| train | {meta['counts']['train']} |
| validation | {meta['counts']['validation']} |
| test | {meta['counts']['test']} |

- 발화 단위로 나눴고 한 발화는 한 분할에만 있습니다. 같은 대본은 걸러서 분할 사이에 겹치지 않습니다.
- 화자가 1명이라 화자 단위로는 나눌 수 없습니다. 같은 책에서 나온 문장이라는 점은 같습니다.
- clean, SPEC 1회, SPEC 10회가 같은 발화 ID와 분할을 쓰도록 이 분할 파일을 기준으로 삼습니다.
- 샘플레이트: {', '.join(str(x) for x in sr)} Hz (MeloTTS-Korean은 44100 Hz)

## 폴더 구조

```text
manifest.csv          발화별 정보 (필수 11개 열 + 참고 열)
manifest.xlsx         manifest.csv와 같은 내용 (엑셀용, openpyxl이 있을 때만 생성)
manifest.json         같은 내용과 선택 조건, 대본 파일 해시
transcripts_all.csv   발화별 대본 전체 (발화 ID, 분할, 대본, 길이, 파일 경로)
spec_input.json       한국어 SPEC 실행 도구(run_korean_melo_spec.py)용 입력
splits/               train.txt, validation.txt, test.txt (발화 ID 목록)
train/ validation/ test/
  wavs/               원본과 바이트가 같은 WAV (파일 이름은 발화 ID)
  transcripts.csv     file,text 형식 (MeloTTS prepare_finetune용)
transcript_check.csv  사람이 듣고 대본을 대조해서 채우는 표
rejects.csv           제외한 파일과 사유
```

## 아직 사람이 해야 하는 일

1. `transcript_check.csv`에서 발화마다 듣고 대본이 맞는지 확인하고 표시합니다. 자동 검사는 대본이 실제 발음과 맞는지 알지 못합니다.
2. 전부 맞으면 `--transcripts-verified`로 `spec_input.json`을 다시 만들거나 값을 직접 바꿉니다.
3. 팀이 이용 조건을 확인하면 `--usage-confirmed`를 사용합니다.

## 다른 팀과 연동

- 섭동 담당: `spec_input.json`과 `train/validation/test/wavs`를 전달합니다. 실행 도구는 `transcript_verified`와 `research_use_permitted`가 true여야 실행합니다.
- MeloTTS 담당: 분할별 `transcripts.csv`(file,text)를 쓰면 됩니다. 파일 열의 이름 형식은 `prepare_finetune.py`와 맞는지 확인이 필요합니다.
- 평가 담당: `splits/test.txt`가 별도 test 발화 목록입니다.
- 실제 음성 복제 학습은 이용 조건 확인 후에 진행합니다.

## 모양 확인

`python build_kss_manifest.py --verify <이 폴더>` 로 파일, 해시, 길이, 분할 겹침을 다시 확인할 수 있습니다.
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def load_manifest(folder):
    folder = Path(folder)
    with open(folder / "manifest.csv", encoding="utf-8-sig", newline="") as f:
        return folder, list(csv.DictReader(f))


def verify(folder):
    folder, rows = load_manifest(folder)
    problems = []
    ids = [r["utterance_id"] for r in rows]
    if len(set(ids)) != len(ids):
        problems.append("발화 ID 중복")
    texts = [norm_text(r["transcript"]) for r in rows]
    if len(set(texts)) != len(texts):
        problems.append("대본 중복")
    shas = [r["sha256"] for r in rows]
    if len(set(shas)) != len(shas):
        problems.append("같은 음성 파일이 중복됨")
    for col in REQUIRED_COLUMNS:
        if rows and col not in rows[0]:
            problems.append("필수 열 없음: " + col)
    split_files = {}
    for s in SPLITS:
        p = folder / "splits" / (s + ".txt")
        split_files[s] = set(p.read_text(encoding="utf-8").split()) if p.exists() else set()
    for a in SPLITS:
        for b in SPLITS:
            if a < b and split_files[a] & split_files[b]:
                problems.append("분할 사이에 겹치는 발화: %s, %s" % (a, b))
    for r in rows:
        p = folder / r["wav_path"]
        if not p.is_file():
            problems.append("파일 없음: " + r["wav_path"])
            continue
        if sha256_of(p) != r["sha256"]:
            problems.append("해시 불일치: " + r["wav_path"])
        info, why = analyze_wav(p)
        if why:
            problems.append("%s: %s" % (why, r["wav_path"]))
            continue
        if info["sample_rate"] != int(r["sample_rate"]):
            problems.append("샘플레이트 불일치: " + r["wav_path"])
        if abs(info["duration"] - float(r["duration"])) > 0.01:
            problems.append("길이 불일치: " + r["wav_path"])
        if r["utterance_id"] not in split_files.get(r["split"], set()):
            problems.append("분할 목록에 없음: " + r["utterance_id"])
    for s in SPLITS:
        t = folder / s / "transcripts.csv"
        if t.exists():
            with open(t, encoding="utf-8-sig", newline="") as f:
                got = {row["file"]: row["text"] for row in csv.DictReader(f)}
            want = {r["utterance_id"] + ".wav": r["transcript"] for r in rows if r["split"] == s}
            if got != want:
                problems.append("transcripts.csv가 manifest와 다름: " + s)
    print("발화 %d개 확인" % len(rows))
    print("문제 없음" if not problems else "문제 %d건" % len(problems))
    for p in problems:
        print(" -", p)
    return 0 if not problems else 1


def check_run(folder, run_dir):
    """run_korean_melo_spec.py 결과 폴더가 같은 발화 ID로 clean, spec_1, spec_10을 가졌는지 본다.
    폴더 구조는 섭동 담당 handoff 문서(2026-11-10) 기준이다."""
    folder, rows = load_manifest(folder)
    run = Path(run_dir)
    problems = []
    for r in rows:
        u = r["utterance_id"]
        clean = run / u / "clean" / (u + ".wav")
        if not clean.is_file():
            problems.append("clean 없음: " + u)
        elif sha256_of(clean) != r["sha256"]:
            problems.append("clean이 원본과 다름: " + u)
        for sub in ("spec_1", "spec_10"):
            if not (run / u / sub).is_dir():
                problems.append("%s 폴더 없음: %s" % (sub, u))
    extra = [p.name for p in run.iterdir() if p.is_dir() and p.name not in {r["utterance_id"] for r in rows}
             and p.name not in {"team", "finetune_inputs"}]
    for name in extra:
        problems.append("manifest에 없는 폴더: " + name)
    print("발화 %d개 확인" % len(rows))
    print("문제 없음" if not problems else "문제 %d건" % len(problems))
    for p in problems:
        print(" -", p)
    return 0 if not problems else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kss-root")
    ap.add_argument("--transcript", help="대본 파일을 직접 지정 (기본: kss-root 아래에서 찾음)")
    ap.add_argument("--out-dir")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--min-sec", type=float, default=2.0)
    ap.add_argument("--max-sec", type=float, default=4.0)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--test-ratio", type=float, default=0.15)
    ap.add_argument("--source-note", default="", help="실제로 받은 곳을 README와 manifest.json에 적음")
    ap.add_argument("--usage-confirmed", action="store_true", help="팀이 이용 조건을 확인한 뒤에만")
    ap.add_argument("--transcripts-verified", action="store_true", help="사람이 대본을 대조한 뒤에만")
    ap.add_argument("--verify", metavar="FOLDER")
    ap.add_argument("--check-run", nargs=2, metavar=("FOLDER", "RUN_DIR"))
    args = ap.parse_args()
    if args.verify:
        sys.exit(verify(args.verify))
    if args.check_run:
        sys.exit(check_run(*args.check_run))
    if not args.kss_root or not args.out_dir:
        ap.error("--kss-root와 --out-dir가 필요합니다")
    build(args)


if __name__ == "__main__":
    main()
