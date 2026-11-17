"""Read-only WSL/CUDA audit; no model downloads or human audio experiments."""
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def main():
    result = {"python": sys.version, "platform": platform.platform(),
              "executable": sys.executable, "tools": {}, "packages": {}}
    for name in ("torch", "torchaudio", "numpy", "librosa", "soundfile", "pystoi",
                 "torch-stoi", "transformers", "huggingface-hub", "melo", "unidecode",
                 "speechbrain", "openai-whisper", "jiwer"):
        try:
            result["packages"][name] = md.version(name)
        except md.PackageNotFoundError:
            result["packages"][name] = None
    for name, args in {"nvidia-smi": ["nvidia-smi"], "ffmpeg": ["ffmpeg", "-version"],
                       "nvcc": ["nvcc", "--version"]}.items():
        location = shutil.which(name)
        result["tools"][name] = {"present": bool(location)}
        if location:
            proc = subprocess.run(args, capture_output=True, text=True)
            result["tools"][name].update(returncode=proc.returncode, output=proc.stdout)
    try:
        import torch
        import torchaudio
        result["cuda"] = {"available": torch.cuda.is_available(), "runtime": torch.version.cuda,
                          "architectures": torch.cuda.get_arch_list()}
        if torch.cuda.is_available():
            result["cuda"].update(gpu=torch.cuda.get_device_name(),
                                  capability=list(torch.cuda.get_device_capability()))
            torch.cuda.reset_peak_memory_stats()
            x = torch.randn(1, 1, 8192, device="cuda", requires_grad=True)
            layer = torch.nn.Conv1d(1, 16, 5).cuda()
            layer(x).square().mean().backward()
            assert torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad)
            a = torch.randn(128, 128, device="cuda")
            assert torch.isfinite(a @ a.T).all()
            spectrum = torch.stft(x.detach().flatten(), 1024,
                                  window=torch.hann_window(1024, device="cuda"), return_complex=True)
            assert torch.isfinite(spectrum).all()
            restored = torchaudio.functional.resample(x.detach().cpu().flatten(), 44100, 16000)
            assert torch.isfinite(restored).all()
            torch.cuda.synchronize()
            result["cuda"].update(kernel_backward_stft_resample="PASS",
                                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                  free_total_bytes=list(torch.cuda.mem_get_info()))
    except Exception as error:
        result["cuda_error"] = f"{type(error).__name__}: {error}"
    result["melo_importable"] = importlib.util.find_spec("melo") is not None
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int("cuda_error" in result or not result.get("cuda", {}).get("available"))


if __name__ == "__main__":
    raise SystemExit(main())
