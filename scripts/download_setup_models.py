"""Download only the pinned models used in the WSL setup (about 2.58GB).

Existing files are verified, never overwritten. Run from the WSL environment.
"""
import hashlib
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
PLANS = [
    ("OedoSoldier/Bert-VITS2-2.3", "3eac455e350982b84267125c370713948b011694",
     "checkpoints/base_models", ["DUR_0.pth", "D_0.pth", "G_0.pth", "WD_0.pth"]),
    ("microsoft/deberta-v3-large", "64a8c8eab3e352a784c658aef62be1662607476f",
     "bert/deberta-v3-large", ["pytorch_model.bin", "spm.model"]),
    ("microsoft/wavlm-base-plus", "4c66d4806a428f2e922ccfa1a962776e232d487b",
     "bert_vits2/slm/wavlm-base-plus", ["pytorch_model.bin", "config.json"]),
]


def main():
    for repo, revision, folder, names in PLANS:
        metadata = HfApi().model_info(repo, revision=revision, files_metadata=True)
        entries = {entry.rfilename: entry for entry in metadata.siblings}
        for name in names:
            path = ROOT / folder / name
            entry = entries[name]
            if not path.exists():
                print(f"DOWNLOAD {repo}/{name}", flush=True)
                hf_hub_download(repo, name, revision=revision, local_dir=ROOT / folder)
            if path.stat().st_size != entry.size:
                raise RuntimeError(f"Size mismatch; file preserved: {path}")
            digest = hashlib.sha256() if entry.lfs else hashlib.sha1()
            if not entry.lfs:
                digest.update(f"blob {entry.size}\0".encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            expected = entry.lfs.sha256 if entry.lfs else entry.blob_id
            if digest.hexdigest() != expected:
                raise RuntimeError(f"Hash mismatch; file preserved: {path}")
            print(f"VERIFIED {path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
