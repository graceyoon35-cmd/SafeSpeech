"""Integration checks for immutable inputs, reproducibility and failed experiments."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from run_experiment import ROOT, run_experiment, sha256
from robustness_transforms import gaussian_noise


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.manifest = json.loads((ROOT / 'experiments/gaussian_smoke.json').read_text())
        self.source = (ROOT / 'experiments' / self.manifest['audio']['original']).resolve()
        self.manifest['audio']['original'] = str(self.source)
        self.path = self.directory / 'manifest.json'
        self.output = self.directory / 'results'

    def run_manifest(self, manifest=None):
        self.path.write_text(json.dumps(self.manifest if manifest is None else manifest))
        return run_experiment(self.path, self.output)

    def test_pipeline_preserves_inputs_and_reproduces_artifact(self):
        original_bytes = self.source.read_bytes()
        _, first = self.run_manifest()
        report_path, second = self.run_manifest()
        self.assertEqual(self.source.read_bytes(), original_bytes)
        self.assertEqual(first['metrics'], second['metrics'])
        self.assertEqual(first['artifact']['sha256'], second['artifact']['sha256'])
        # libsndfile floating WAVs can include a timestamped PEAK chunk.
        self.assertNotIn(b'PEAK', Path(second['artifact']['path']).read_bytes()[:80])
        self.assertNotEqual(first['artifact']['path'], second['artifact']['path'])
        self.assertEqual(json.loads(report_path.read_text()), second)
        entry = second['inputs']['original']
        self.assertEqual(entry['sha256_before'], entry['sha256_after'])
        self.assertEqual(sha256(self.source), entry['sha256_before'])
        self.assertAlmostEqual(second['metrics']['before']['stoi'], 1, places=6)
        self.assertLess(second['metrics']['after']['stoi'], .99)
        changed = copy.deepcopy(self.manifest)
        changed['transform']['seed'] = 43
        _, third = self.run_manifest(changed)
        self.assertNotEqual(first['artifact']['sha256'], third['artifact']['sha256'])

    def test_transform_does_not_mutate_array(self):
        audio = np.linspace(-.5, .5, 24000)
        saved = audio.copy()
        first = gaussian_noise(audio, seed=42, std=.005)
        np.testing.assert_array_equal(audio, saved)
        np.testing.assert_array_equal(first, gaussian_noise(audio, seed=42, std=.005))
        self.assertFalse(np.shares_memory(audio, first))

    def test_manifest_errors_produce_no_success_output(self):
        changes = [
            ('schema_version', 2),
            ('experiment_id', ''),
            ('audio', {**self.manifest['audio'], 'original': 'missing.wav'}),
            ('audio', {**self.manifest['audio'], 'original': None}),
            ('evaluation', {**self.manifest['evaluation'], 'target_role': 'protected'}),
            ('evaluation', {**self.manifest['evaluation'], 'alignment': 'truncate'}),
            ('transform', {'name': 'unknown', 'seed': 42, 'std': .005}),
        ]
        for key, value in changes:
            with self.subTest(key=key, value=value):
                invalid = copy.deepcopy(self.manifest)
                invalid[key] = value
                with self.assertRaises(ValueError):
                    self.run_manifest(invalid)
                self.assertFalse(self.output.exists())

    def test_invalid_transform_parameters(self):
        for seed, std in [(-1, .005), (True, .005), (1.2, .005),
                          (42, -1), (42, float('nan')), (42, '0.005'), (42, True)]:
            with self.subTest(seed=seed, std=std), self.assertRaises(ValueError):
                gaussian_noise([.1], seed=seed, std=std)

    def test_relative_paths_and_protected_role(self):
        audio, rate = sf.read(self.source)
        sf.write(self.directory / 'protected.wav', audio, rate, subtype='DOUBLE')
        self.manifest['audio']['protected'] = 'protected.wav'
        self.manifest['evaluation']['target_role'] = 'protected'
        _, result = self.run_manifest()
        self.assertEqual(result['manifest']['evaluation']['target_role'], 'protected')
        self.assertEqual(result['inputs']['protected']['path'], str((self.directory / 'protected.wav').resolve()))

    def test_bad_audio_and_alignment_fail_before_output(self):
        for audio, rate in [(np.zeros(24000), 24000), (np.ones((24000, 2)) * .1, 24000),
                            (np.ones(24480) * .1, 24000), (np.ones(24481) * .1, 16000)]:
            with self.subTest(shape=audio.shape, rate=rate):
                bad = self.directory / 'bad.wav'
                sf.write(bad, audio, rate, subtype='FLOAT')
                self.manifest['audio']['synthetic'] = str(bad)
                self.manifest['evaluation']['target_role'] = 'synthetic'
                with self.assertRaises(ValueError):
                    self.run_manifest()
                self.assertFalse(self.output.exists())


if __name__ == '__main__':
    unittest.main()
