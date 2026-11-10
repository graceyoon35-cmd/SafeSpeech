"""44.1 kHz plumbing checks with mathematical signals, never Korean/SPEC evidence."""
import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

import numpy as np
from scipy.io import wavfile
import soundfile as sf

from export_robustness_handoff_b import export_handoff, verify_handoff
from run_robustness_b import run_pair, sha256
from run_robustness_sweep_b import default_conditions, run_sweep


class Rate44100Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rate = 44100
        # Odd length matches the documented Melo synthesis length; content is tones.
        time = np.arange(196096) / self.rate
        self.original = .2 * np.sin(2 * np.pi * 220 * time)
        self.fixture = self.original + .001 * np.sin(2 * np.pi * 997 * time)
        wavfile.write(self.root / 'original.wav', self.rate, self.original)
        wavfile.write(self.root / 'fixture_not_spec.wav', self.rate, self.fixture)
        self.value = {
            'schema_version': 1, 'experiment_id': 'mathematical_fixture_not_korean_not_spec_44100',
            'audio': {'original': 'original.wav', 'protected': 'fixture_not_spec.wav'},
            'conditions': default_conditions(),
        }
        self.path = self.root / 'input.json'
        self.save_manifest()

    def save_manifest(self):
        self.path.write_text(json.dumps(self.value), encoding='utf-8')

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg required for full MP3 readiness check')
    def test_full_grid_repetition_and_portable_handoff(self):
        before = {name: sha256(self.root / name)
                  for name in ('original.wav', 'fixture_not_spec.wav', 'input.json')}
        summary_path, summary = run_sweep(self.path, self.root / 'runs')
        self.assertEqual(summary['status'], 'PASS')
        self.assertEqual(summary['wav_count'], 60)
        self.assertEqual(summary['mp3_count'], 12)
        self.assertTrue(summary['reproducibility_pass'])
        self.assertTrue(summary['input_files_unchanged'])
        self.assertEqual(len(summary['reproducibility']), 30)
        for report_name in summary['reports']:
            report = json.loads((summary_path.parent / report_name).read_text(encoding='utf-8'))
            for entry in report['outputs']:
                parameters = entry['parameters']
                expected_rate = parameters.get('target_hz', self.rate)
                expected_samples = (len(self.original) * expected_rate + self.rate - 1) // self.rate
                audio, actual_rate = sf.read(entry['path'])
                self.assertEqual(actual_rate, expected_rate)
                self.assertEqual(audio.shape, (expected_samples,))
                self.assertTrue(np.isfinite(audio).all())
                self.assertEqual(entry['input_rate'], self.rate)
                if parameters['name'] == 'resample':
                    self.assertLessEqual(abs(len(audio) / actual_rate - len(self.original) / self.rate),
                                         1 / actual_rate)
        folder, archive = export_handoff(summary_path, self.root / 'exports')
        with zipfile.ZipFile(archive) as zipped:
            self.assertIsNone(zipped.testzip())
            zipped.extractall(self.root / 'relocated')
        moved = self.root / 'relocated' / folder.name
        verify_handoff(moved)
        with (moved / 'evaluation_pairs.csv').open(encoding='utf-8-sig', newline='') as stream:
            pairs = list(csv.DictReader(stream))
        self.assertEqual(len(pairs), 30)
        self.assertTrue(all(pair['evaluation_status'] == 'not_measured' for pair in pairs))
        for name, digest in before.items():
            self.assertEqual(sha256(self.root / name), digest)

    def test_stereo_pair_is_rejected_without_downmix(self):
        stereo = np.column_stack((self.fixture, self.fixture * .9))
        wavfile.write(self.root / 'fixture_not_spec.wav', self.rate, stereo)
        with self.assertRaisesRegex(ValueError, 'mono'):
            run_pair(self.path, self.root / 'runs')
        self.assertFalse((self.root / 'runs').exists())

    def test_unequal_sample_count_is_rejected_without_trimming(self):
        wavfile.write(self.root / 'fixture_not_spec.wav', self.rate, self.fixture[:-1])
        with self.assertRaisesRegex(ValueError, 'matching'):
            run_pair(self.path, self.root / 'runs')
        self.assertFalse((self.root / 'runs').exists())


if __name__ == '__main__':
    unittest.main()
