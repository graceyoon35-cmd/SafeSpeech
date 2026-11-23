"""Adapter regression tests. The recording fixture is a test double, not a model."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import json
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave

import numpy as np

from spec_pipeline import MediaProtectionClient, SAMPLE_RATE, SpecChunks
from spec_desktop_bridge import SpecConfig, pcm16


class RecordingClient:
    def __init__(self):
        self.inputs = []

    def protect(self, path):
        samples = np.array(pcm16(path), dtype="<i2")
        self.inputs.append(samples.copy())
        result = path.with_name("test-output.wav")
        with wave.open(str(result), "wb") as target:
            target.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            target.writeframes((samples ^ 1).tobytes())
        return SimpleNamespace(output_path=result)


class SpecChunkTests(unittest.TestCase):
    def test_subsecond_input_pads_to_one_second_and_restores_original_length(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = RecordingClient()
            adapter = SpecChunks(client, Path(tmp))
            original = np.array([-32768, -1, 0, 1, 32767], dtype="<i2")
            (result,) = adapter.process_batch([original.astype(np.float32) / 32768])
            np.testing.assert_array_equal(client.inputs[0][:5], original)
            self.assertEqual(len(client.inputs[0]), SAMPLE_RATE)
            self.assertFalse(np.any(client.inputs[0][5:]))
            np.testing.assert_array_equal(result * 32768, original ^ 1)

    def test_silent_chunk_preserves_timeline_without_calling_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = RecordingClient()
            adapter = SpecChunks(client, Path(tmp))
            silent = np.zeros(400, dtype=np.float32)
            (result,) = adapter.process_batch([silent])
            np.testing.assert_array_equal(result, silent)
            self.assertEqual(adapter.protected_chunks, 0)
            self.assertFalse(client.inputs)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires FFmpeg")
    def test_pipeline_resamples_splits_and_rejoins_in_source_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            (root / "scripts" / "protect_wav.py").touch()
            config = SpecConfig(repository=root, output_root=root / "results")
            engine = MediaProtectionClient(replace(config, epsilon=4))
            src = root / "시험 음성.wav"
            rate = 48000
            frames = np.sin(np.arange(rate * 12) * 0.04) * 8000
            with wave.open(str(src), "wb") as stream:
                stream.setparams((2, 2, rate, 0, "NONE", "not compressed"))
                stream.writeframes(np.column_stack((frames, frames)).astype("<i2").tobytes())
            client = RecordingClient()
            with patch("spec_pipeline.SpecClient", return_value=client) as factory:
                result = engine.protect(src)
            self.assertEqual(factory.call_args.args[0].epsilon, 4)
            self.assertEqual([len(chunk) for chunk in client.inputs], [352800, 198450])
            probed = subprocess.check_output([
                "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(result.output_path)
            ])
            stream = json.loads(probed)["streams"][0]
            self.assertEqual((stream["channels"], int(stream["sample_rate"]), stream["duration_ts"]),
                             (2, rate, len(frames)))
            self.assertTrue(src.exists())
            self.assertFalse(list(result.output_path.parent.glob("work-*")))

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "requires FFmpeg")
    def test_silent_file_does_not_publish_a_protected_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            (root / "scripts" / "protect_wav.py").touch()
            config = SpecConfig(repository=root, output_root=root / "results")
            src = root / "silent.wav"
            with wave.open(str(src), "wb") as stream:
                stream.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
                stream.writeframes(bytes(2 * SAMPLE_RATE))
            with self.assertRaisesRegex(ValueError, "무음"):
                MediaProtectionClient(config).protect(src)
            self.assertFalse(list(config.output_root.rglob("*_protected.wav")))
