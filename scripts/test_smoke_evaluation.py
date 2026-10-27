"""Validate pair rejection and STOI controls without synthesis dependencies."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import soundfile as sf

from smoke_evaluation import DEFAULT_AUDIO, load_audio, score_pair
from robustness_transforms import gaussian_noise


class EvaluationContractTests(unittest.TestCase):
    def test_identity_and_noise_control(self):
        audio, rate = load_audio(DEFAULT_AUDIO)
        self.assertAlmostEqual(score_pair(audio, audio, rate, rate)['stoi'], 1, places=6)
        noisy = gaussian_noise(audio, seed=42, std=.005)
        self.assertLess(score_pair(audio, noisy, rate, rate)['stoi'], .99)

    def test_misaligned_pairs_are_rejected(self):
        audio = np.ones(24000)
        for other, rate in [(audio[:-1], 24000), (audio, 16000)]:
            with self.assertRaises(ValueError):
                score_pair(audio, other, 24000, rate)

    def test_invalid_audio_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.wav'
            for audio in [np.zeros(24000), np.ones((24000, 2)) * .1,
                          np.ones(100) * .1, np.full(24000, np.nan), np.ones(24000) * 2]:
                sf.write(path, audio, 24000, subtype='FLOAT')
                with self.assertRaises(ValueError):
                    load_audio(path)


if __name__ == '__main__':
    unittest.main()
