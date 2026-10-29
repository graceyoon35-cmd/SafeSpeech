"""Run the original SPEC perturb() on one copied utterance, without training.

Run from the repository root using the WSL virtual environment.
This checks execution and epsilon bounds, not protection effectiveness.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timedelta, timezone

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT / "bert_vits2"))

import soundfile as sf
import torch
from torch.utils.data import DataLoader
import protect
from bert_vits2 import utils
from bert_vits2.data_utils import TextAudioSpeakerLoader, TextAudioSpeakerCollate
from bert_vits2.models_noise import SynthesizerTrn
from bert_vits2.text import english_bert_mock
from bert_vits2.text.symbols import symbols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    assert torch.cuda.is_available(), "CUDA GPU required"
    hps = utils.get_hparams_from_file("bert_vits2/configs/libritts_bert_vits2.json")
    torch.manual_seed(hps.train.seed)
    rows = [line.split("|") for line in Path(hps.data.training_files).read_text().splitlines() if line.strip()]
    candidates = [(sf.info(row[0]).duration, row) for row in rows if row[2] == "EN"]
    duration, row = min((duration, row) for duration, row in candidates if duration >= 1)
    run = ROOT / "runs" / ("smoke_" + datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S_%f"))
    run.mkdir(parents=True, exist_ok=False)
    source = row[0]
    wav = run / Path(source).name
    shutil.copyfile(source, wav)
    row[0] = wav.relative_to(ROOT).as_posix()
    filelist = run / "input.cleaned"
    filelist.write_text("|".join(row) + "\n", encoding="utf-8")
    print(f"RUN_DIR {run}\nINPUT {source} ({duration:.2f}s)", flush=True)

    word2ph = [int(n) for n in row[6].split()]
    if hps.data.add_blank:
        word2ph = [n * 2 for n in word2ph]
        word2ph[0] += 1
    # Use the original English feature function directly; avoid unrelated language setup.
    features = english_bert_mock.get_bert_feature(row[3], word2ph, device="cpu")
    expected_length = len(row[4].split()) * (2 if hps.data.add_blank else 1) + int(hps.data.add_blank)
    assert features.shape == (1024, expected_length) and torch.isfinite(features).all()
    torch.save(features, wav.with_suffix(".bert.pt"))
    print(f"BERT_OK {tuple(features.shape)}", flush=True)
    del features
    english_bert_mock.models.clear()
    gc.collect()

    dataset = TextAudioSpeakerLoader(str(filelist.relative_to(ROOT)), hps.data)
    assert len(dataset) == 1
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=TextAudioSpeakerCollate())))
    model = SynthesizerTrn(
        len(symbols), hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers, mas_noise_scale_initial=0.01,
        noise_scale_delta=2e-6, **hps.model,
    )
    checkpoint = torch.load("checkpoints/base_models/G_0.pth", map_location="cpu", weights_only=True)
    # The published base checkpoint omits the speaker table. The original
    # utils.load_checkpoint also retains initialization for this missing key.
    compatibility = model.load_state_dict(checkpoint["model"], strict=False)
    assert compatibility.missing_keys == ["emb_g.weight"], compatibility
    assert not compatibility.unexpected_keys, compatibility
    print("CHECKPOINT_NOTE emb_g.weight retains seeded initialization", flush=True)
    del checkpoint
    model.cuda()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.cuda.reset_peak_memory_stats()
    epsilon = 8 / 255
    losses, noise = protect.perturb(hps, model, batch, epsilon, epsilon / 10, args.iterations, [0.05, 10], "SPEC", "cuda:0")
    assert all(math.isfinite(float(value)) for value in losses.values()), losses
    assert torch.isfinite(noise).all()
    maximum = noise.abs().max().item()
    assert 0 < maximum <= epsilon + 1e-7
    original = batch[4]
    noise = noise.detach().cpu()
    protected = torch.clamp(original + noise, -1, 1)
    length = int(batch[5][0])
    torch.save(noise, run / "perturbation.pt")
    sf.write(run / "protected.wav", protected[0, 0, :length].numpy(), hps.data.sampling_rate, subtype="FLOAT")
    report = {"status": "PASS", "mode": "SPEC", "iterations": args.iterations, "source": source,
              "seed": hps.train.seed,
              "duration_seconds": duration, "epsilon": epsilon, "max_abs_noise": maximum,
              "losses": losses, "peak_allocated_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
              "checkpoint_missing_keys": compatibility.missing_keys,
              "speaker_embedding": "seeded initialization, matching original loader fallback",
              "scope": "Original generator and perturb function; excludes full CLI and protection evaluation"}
    (run / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
