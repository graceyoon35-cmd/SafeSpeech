"""Check the common environment without downloading models or touching data."""
import io
import json

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio


def main():
    assert torch.cuda.is_available(), "CUDA GPU is unavailable"
    # Exercise kernels, cuBLAS, cuDNN and gradients used by perturbation generation.
    x = torch.randn(1, 1, 2048, device="cuda", requires_grad=True)
    conv = torch.nn.Conv1d(1, 4, 5).cuda()
    loss = conv(x).square().mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    a = torch.randn(64, 64, device="cuda")
    assert torch.isfinite(a @ a.T).all()
    spectrum = torch.stft(
        x.detach().flatten(), 256,
        window=torch.hann_window(256, device="cuda"), return_complex=True,
    )
    assert torch.isfinite(spectrum).all()
    torch.cuda.synchronize()

    wave = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, wave, 16000, format="WAV", subtype="FLOAT")
    buffer.seek(0)
    restored, rate = sf.read(buffer, dtype="float32")
    assert rate == 16000 and np.allclose(wave, restored)
    resampled = torchaudio.functional.resample(torch.from_numpy(restored), rate, 8000)
    assert resampled.shape == (8000,) and torch.isfinite(resampled).all()
    mel = librosa.feature.melspectrogram(y=restored, sr=rate, n_mels=40)
    assert mel.shape[0] == 40 and np.isfinite(mel).all()
    print(json.dumps({
        "status": "PASS",
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "numpy": np.__version__,
        "librosa": librosa.__version__,
        "soundfile": sf.__version__,
    }, indent=2))


if __name__ == "__main__":
    main()
