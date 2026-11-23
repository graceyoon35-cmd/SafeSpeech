"""Connect the desktop media pipeline to the SPEC team's validated WAV client."""
from configparser import ConfigParser
from dataclasses import dataclass, replace
from pathlib import Path
import os
import sys
import tempfile
import wave

import numpy as np

from audio_pipeline import AudioProtectionPipeline, PipelineConfig, PipelineError
from spec_desktop_bridge import SpecClient, SpecConfig


SAMPLE_RATE = 44_100


def desktop_config():
    app_dir = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    data_dir = Path(os.getenv("LOCALAPPDATA") or Path.home()) / "VoiceProtect" / "results"
    settings = ConfigParser(interpolation=None)
    settings.read(app_dir / "voiceprotect.ini", encoding="utf-8-sig")

    def option(key, env, default):
        return os.getenv(env) or settings.get("spec", key, fallback=str(default))

    return SpecConfig(
        repository=Path(option("repository", "VOICEPROTECT_SPEC_REPO", app_dir / "spec-runtime")),
        output_root=Path(option("output_root", "VOICEPROTECT_RESULT_DIR", data_dir)),
        distribution=option("distribution", "VOICEPROTECT_WSL_DISTRO", SpecConfig.distribution),
        python_executable=option("python_executable", "VOICEPROTECT_SPEC_PYTHON", SpecConfig.python_executable),
        assets=option("assets", "VOICEPROTECT_SPEC_ASSETS", SpecConfig.assets),
        epsilon=float(option("epsilon", "VOICEPROTECT_SPEC_EPSILON", SpecConfig.epsilon)),
    )


@dataclass(frozen=True)
class ProtectionResult:
    original_path: Path
    elapsed_seconds: float
    output_path: Path

    @property
    def protected_name(self):
        return self.output_path.name


class SpecChunks:
    tag = "spec-pcm16"

    def __init__(self, client, scratch):
        self.client = client
        self.wav = scratch / "input.wav"
        self.protected_chunks = 0

    def process_batch(self, chunks):
        return [self._protect_chunk(chunk) for chunk in chunks]

    def _protect_chunk(self, chunk):
        pcm = np.clip(np.rint(chunk * 32768), -32768, 32767).astype("<i2")
        if not np.any(pcm):
            # Pure silence carries no voice; the SPEC CLI rejects it.
            return chunk.copy()
        if pcm.size < SAMPLE_RATE:
            pcm = np.pad(pcm, (0, SAMPLE_RATE - pcm.size))
        with wave.open(str(self.wav), "wb") as target:
            target.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            target.writeframes(pcm.tobytes())

        result = self.client.protect(self.wav)
        # SpecClient has already checked both hashes, format, length and epsilon.
        with wave.open(str(result.output_path), "rb") as protected:
            raw = protected.readframes(len(chunk))
        result.output_path.unlink()
        self.protected_chunks += 1
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0


class MediaProtectionClient:
    def __init__(self, config=None):
        self.config = config if config is not None else desktop_config()

    def protect(self, item, on_progress=None):
        src = Path(getattr(item, "path", item)).resolve(strict=True)
        if not (self.config.repository / "scripts" / "protect_wav.py").is_file():
            raise ValueError("SPEC 실행기를 찾을 수 없습니다. VOICEPROTECT_SPEC_REPO 경로를 확인해주세요.")
        self.config.output_root.mkdir(parents=True, exist_ok=True)
        job = Path(tempfile.mkdtemp(prefix="media-", dir=self.config.output_root))
        with tempfile.TemporaryDirectory(dir=job, prefix="work-") as work:
            scratch = Path(work)
            model = SpecClient(replace(self.config, output_root=job / "diagnostics"))
            chunks = SpecChunks(model, scratch)
            options = PipelineConfig(model_sample_rate=SAMPLE_RATE, work_root=scratch,
                                     batch_size=1, max_retries=0)
            # In a frozen application these executables live inside the bundle.
            if getattr(sys, "frozen", False):
                bundle_dir = Path(sys._MEIPASS)
                options.ffmpeg_path = str(bundle_dir / "ffmpeg.exe")
                options.ffprobe_path = str(bundle_dir / "ffprobe.exe")

            pipeline = AudioProtectionPipeline(chunks, options, on_progress)
            staged = scratch / (src.stem + "_protected" + src.suffix.lower())
            try:
                result = pipeline.run(src, staged)
            except PipelineError as error:
                raise RuntimeError(error.detail or error.user_message) from error
            if chunks.protected_chunks == 0:
                raise ValueError("파일 전체가 무음입니다. 보호할 음성이 있는 파일을 선택해주세요.")
            output = job / staged.name
            staged.replace(output)
            return ProtectionResult(src, result.elapsed_sec, output)
