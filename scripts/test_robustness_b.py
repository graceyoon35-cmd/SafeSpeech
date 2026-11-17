"""Regression checks for paired transforms, metadata and immutable sources."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile
import soundfile as sf

from robustness_b import transform_audio, validate_condition
from run_robustness_b import ROOT, run_pair, sha256


class RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.audio, self.rate = sf.read(ROOT / 'data/LibriTTS/5339/5339_14134_000092_000003.wav')

    def test_gaussian_seed_and_no_mutation(self):
        saved = self.audio.copy()
        condition = {'name': 'gaussian_noise', 'seed': 42, 'std': .005}
        first, rate, info = transform_audio(self.audio, self.rate, condition)
        second, _, _ = transform_audio(self.audio, self.rate, condition)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(saved, self.audio)
        self.assertFalse(np.shares_memory(first, self.audio))
        self.assertEqual(rate, self.rate)
        self.assertEqual(info['output_samples'], len(saved))

    def test_resample_native_rate_and_duration(self):
        saved = self.audio.copy()
        out, rate, info = transform_audio(self.audio, self.rate, {'name': 'resample', 'target_hz': 16000})
        self.assertEqual(rate, 16000)
        self.assertLessEqual(abs(len(out) / rate - len(self.audio) / self.rate), 1 / rate)
        np.testing.assert_array_equal(saved, self.audio)
        self.assertEqual(info['output_rate'], 16000)
        np.testing.assert_array_equal(out, transform_audio(self.audio, self.rate, {'name': 'resample', 'target_hz': 16000})[0])

    @unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg unavailable')
    def test_mp3_round_trip_and_reproducibility(self):
        saved = self.audio.copy()
        condition = {'name': 'mp3', 'bitrate_kbps': 64}
        out, rate, info = transform_audio(self.audio, self.rate, condition)
        repeated, _, other = transform_audio(self.audio, self.rate, condition)
        self.assertEqual(out.shape, saved.shape)
        self.assertEqual(rate, self.rate)
        self.assertGreater(info['encoded_bytes'], 0)
        np.testing.assert_array_equal(out, repeated)
        self.assertEqual(info['_encoded_mp3'], other['_encoded_mp3'])
        np.testing.assert_array_equal(saved, self.audio)

    def test_invalid_options(self):
        for condition in [
            {'name': 'mp3', 'bitrate_kbps': True}, {'name': 'mp3', 'bitrate_kbps': 123},
            {'name': 'resample', 'target_hz': 0}, {'name': 'resample', 'target_hz': 16000.0},
            {'name': 'resample', 'target_hz': 16000, 'extra': 1},
            {'name': 'gaussian_noise', 'seed': -1, 'std': .005}, {'name': 'unknown'},
        ]:
            with self.subTest(condition=condition), self.assertRaises(ValueError):
                validate_condition(condition)

    def test_invalid_audio(self):
        for audio in [[], [[0.1, 0.2]], [float('nan')], [2.0]]:
            with self.subTest(audio=audio), self.assertRaises(ValueError):
                transform_audio(audio, self.rate, {'name': 'resample', 'target_hz': 16000})

    def manifest(self):
        # This is a synthetic fixture, explicitly not a real SPEC result.
        wavfile.write(self.directory / 'original.wav', self.rate, self.audio)
        wavfile.write(self.directory / 'protected_fixture.wav', self.rate, self.audio * .99)
        value = {'schema_version': 1, 'experiment_id': 'synthetic_pair_fixture',
                 'audio': {'original': 'original.wav', 'protected': 'protected_fixture.wav'},
                 'conditions': [{'name': 'gaussian_noise', 'seed': 42, 'std': .005},
                                {'name': 'resample', 'target_hz': 16000}]}
        path = self.directory / 'manifest.json'
        path.write_text(json.dumps(value), encoding='utf-8')
        return path, value

    def test_pair_report_roundtrip_input_hash_and_reproducibility(self):
        path, value = self.manifest()
        if shutil.which('ffmpeg'):
            value['conditions'].append({'name': 'mp3', 'bitrate_kbps': 64})
            path.write_text(json.dumps(value), encoding='utf-8')
        source = self.directory / 'original.wav'
        initial = sha256(source)
        report_path, first = run_pair(path, self.directory / 'runs')
        _, second = run_pair(path, self.directory / 'runs')
        self.assertEqual(first['status'], 'PASS')
        self.assertEqual(json.loads(report_path.read_text(encoding='utf-8')), first)
        self.assertEqual(initial, sha256(source))
        self.assertEqual(len(first['outputs']), len(value['conditions']) * 2)
        self.assertEqual([v['sha256'] for v in first['outputs']], [v['sha256'] for v in second['outputs']])
        for entry in first['outputs']:
            data, rate = sf.read(entry['path'])
            self.assertEqual(len(data), entry['output_samples'])
            self.assertEqual(rate, entry['output_rate'])
        for entry in first['inputs'].values():
            self.assertEqual(entry['sha256_before'], entry['sha256_after'])

    def test_bad_pairs_fail_before_output(self):
        path, value = self.manifest()
        for bad in [None, '', 'missing.wav', 'original.wav']:
            value['audio']['protected'] = bad
            path.write_text(json.dumps(value), encoding='utf-8')
            with self.subTest(path=bad), self.assertRaises((ValueError, FileNotFoundError)):
                run_pair(path, self.directory / 'runs')
            self.assertFalse((self.directory / 'runs').exists())

    def test_mismatched_pair_fails(self):
        path, _ = self.manifest()
        wavfile.write(self.directory / 'protected_fixture.wav', 16000, self.audio * .99)
        with self.assertRaises(ValueError):
            run_pair(path, self.directory / 'runs')

    def test_transform_failure_records_fail(self):
        from unittest.mock import patch
        path, _ = self.manifest()
        with patch('run_robustness_b.transform_audio', side_effect=RuntimeError('test encoder failure')):
            with self.assertRaises(RuntimeError):
                run_pair(path, self.directory / 'runs')
        report_path = next((self.directory / 'runs').glob('*/report.json'))
        report = json.loads(report_path.read_text(encoding='utf-8'))
        self.assertEqual(report['status'], 'FAIL')
        self.assertIn('encoder failure', report['error'])


if __name__ == '__main__':
    unittest.main()
