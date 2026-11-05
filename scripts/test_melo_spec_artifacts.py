"""Model-free input-contract tests; no real Melo/SPEC performance measurements."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

import evaluate_melo_spec_stoi as evaluator
import validate_melo_spec_pair as validator


class MeloArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / 'clean').mkdir()
        (self.root / 'protected').mkdir()
        self.original = self.root / 'clean' / 'fixture.wav'
        self.protected = self.root / 'protected' / 'fixture.wav'
        self.report = self.root / 'protected' / 'report.json'
        rng = np.random.default_rng(52)
        self.wave = rng.uniform(-.15, .15, 48000).astype(np.float32)
        sf.write(self.original, self.wave, 24000, subtype='FLOAT')
        self.baseline = validator.sha256(self.original)
        self.delta = 8 / 255
        sf.write(self.protected, self.wave.astype(np.float64) - self.delta, 24000, subtype='PCM_16')
        self.generation = {
            'args': {'epsilon': 8, 'seed': 52, 'iters': 10, 'mode': 'SPEC', 'random_init': True},
            'model_sampling_rate': 44100,
            'files': [{'input': str(self.original), 'output': str(self.protected),
                       'sample_rate': 24000, 'max_abs_delta': self.delta,
                       'recon_mel_l1_clean': 1., 'recon_mel_l1_protected': 1.1, 'snr_db': 12.,
                       'history': [{'iter': 0, 'loss_mel': 1., 'loss_nr': 2., 'loss_kl': .1},
                                   {'iter': 9, 'loss_mel': 1.1, 'loss_nr': 2., 'loss_kl': .1}]}],
            'total_seconds': .1,
        }
        self.write_report()

    def write_report(self):
        self.report.write_text(json.dumps(self.generation))

    def validate(self, **kwargs):
        settings = dict(expected_original_sha256=self.baseline, execution_id='MOCK_FIXTURE_ONLY',
                        input_kind='mock_contract')
        settings.update(kwargs)
        return validator.validate(self.report, self.original, self.protected, **settings)

    def save_validation(self):
        result = self.validate()
        path = self.root / 'validation.json'
        path.write_text(json.dumps(result))
        return path

    def test_pcm16_tolerance_and_provenance(self):
        before = {p: p.read_bytes() for p in (self.original, self.protected, self.report)}
        result = self.validate()
        self.assertEqual(result['status'], 'MOCK_PASS')
        self.assertFalse(result['model_weights_verified'])
        self.assertEqual(result['model_kind'], 'MeloTTS-Korean')
        self.assertEqual(result['model_sampling_rate'], 44100)
        self.assertEqual(result['original']['sample_rate'], 24000)
        self.assertGreater(result['max_abs_delta'], self.delta)
        self.assertLessEqual(result['max_abs_delta'], self.delta + validator.PCM16_LSB + 1e-7)
        self.assertEqual(result['files_sha256'], result['files_sha256_after'])
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_hash_baseline_and_during_execution_mutation(self):
        with self.assertRaisesRegex(ValueError, 'baseline'):
            self.validate(expected_original_sha256='0' * 64)
        original_loader = validator.load_audio
        def mutate(path):
            wave, rate = original_loader(path)
            if Path(path) == self.protected:
                with self.report.open('a') as stream:
                    stream.write('\n')
            return wave, rate
        with patch.object(validator, 'load_audio', side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, 'changed during'):
                self.validate()

    def test_rate_and_length_mismatches(self):
        for length, rate in [(48000, 16000), (47999, 24000)]:
            with self.subTest(length=length, rate=rate):
                sf.write(self.protected, self.wave[:length] - .01, rate, subtype='PCM_16')
                with self.assertRaisesRegex(ValueError, 'rate or length mismatch'):
                    self.validate()

    def test_stereo_silence_and_short_audio(self):
        for wave in [np.column_stack([self.wave, self.wave]), np.zeros(48000), self.wave[:100]]:
            with self.subTest(shape=wave.shape):
                sf.write(self.protected, wave, 24000, subtype='PCM_16')
                with self.assertRaises(ValueError):
                    self.validate()

    def test_nan_inf_and_overrange(self):
        for value in [np.nan, np.inf, -np.inf, 1.1]:
            with self.subTest(value=value):
                damaged = self.wave.copy()
                damaged[10] = value
                sf.write(self.protected, damaged, 24000, subtype='FLOAT')
                with self.assertRaises(ValueError):
                    self.validate()

    def test_saved_epsilon_excess_and_reported_excess(self):
        sf.write(self.protected, self.wave - .04, 24000, subtype='PCM_16')
        with self.assertRaisesRegex(ValueError, 'exceeds epsilon'):
            self.validate()
        self.generation['files'][0]['max_abs_delta'] = .04
        self.write_report()
        with self.assertRaisesRegex(ValueError, 'reported delta exceeds'):
            self.validate()

    def test_boundary_samples_are_reported_not_hidden(self):
        self.wave[0] = -1
        sf.write(self.original, self.wave, 24000, subtype='FLOAT')
        self.baseline = validator.sha256(self.original)
        candidate = np.clip(self.wave.astype(np.float64) - self.delta, -1, 1)
        sf.write(self.protected, candidate, 24000, subtype='PCM_16')
        result = self.validate()
        self.assertGreater(result['protected']['boundary_samples'], 0)
        self.assertIn('do not prove prior clipping', result['limitations'])

    def test_corrupt_missing_and_wrong_subtype(self):
        sf.write(self.protected, self.wave, 24000, subtype='FLOAT')
        with self.assertRaisesRegex(ValueError, 'PCM_16'):
            self.validate()
        self.protected.write_bytes(b'not a WAV file')
        with self.assertRaises((RuntimeError, ValueError)):
            self.validate()
        self.protected.unlink()
        with self.assertRaises(FileNotFoundError):
            self.validate()

    def test_report_contract_errors(self):
        for key, value in [('seed', None), ('epsilon', -8), ('iters', 0), ('mode', 'SafeSpeech')]:
            original = self.generation['args'][key]
            self.generation['args'][key] = value
            self.write_report()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()
            self.generation['args'][key] = original
        self.generation['files'][0]['history'].pop()
        self.write_report()
        with self.assertRaisesRegex(ValueError, 'first and final'):
            self.validate()
        self.generation['files'][0]['history'].append(
            {'iter': 9, 'loss_mel': 1.1, 'loss_nr': 2., 'loss_kl': .1})
        self.generation['files'][0]['history'][0]['loss_kl'] = float('nan')
        self.write_report()
        with self.assertRaisesRegex(ValueError, 'finite'):
            self.validate()

    def test_ambiguous_names_and_wrong_record(self):
        self.generation['files'].append(self.generation['files'][0].copy())
        self.write_report()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.validate()
        self.generation['files'].pop()
        self.generation['files'][0]['output'] = '/other/wrong.wav'
        self.write_report()
        with self.assertRaisesRegex(ValueError, 'filenames'):
            self.validate()

    def test_random_init_cannot_be_declared_real(self):
        with self.assertRaisesRegex(ValueError, 'Random-initialized'):
            self.validate(input_kind='melo_artifact')

    def test_stoi_json_csv_and_repeated_scoring(self):
        path = self.save_validation()
        first = evaluator.evaluate_validation(path)
        self.assertEqual(first, evaluator.evaluate_validation(path))
        self.assertEqual(first['status'], 'MOCK_PASS')
        self.assertAlmostEqual(first['identity_stoi'], 1, places=6)
        self.assertTrue(np.isfinite(first['stoi']))
        args = ['evaluate_melo_spec_stoi.py', '--validation', str(path), '--output-root', str(self.root / 'scores')]
        with patch('sys.argv', args), patch('builtins.print'):
            self.assertEqual(evaluator.main(), 0)
        report_path = next((self.root / 'scores').glob('*/report.json'))
        report = json.loads(report_path.read_text())
        self.assertEqual(report['status'], 'MOCK_PASS')
        with report_path.with_name('comparison.csv').open() as stream:
            row = next(csv.DictReader(stream))
        self.assertEqual(row['model_kind'], 'MeloTTS-Korean')
        self.assertEqual(row['status'], 'MOCK_PASS')
        self.assertEqual(row['original_sha256'], self.baseline)

    def test_evaluator_rejects_post_validation_corruption(self):
        path = self.save_validation()
        with self.protected.open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed since validation'):
            evaluator.evaluate_validation(path)

    def test_evaluator_storage_failure_and_cli_no_overwrite(self):
        path = self.save_validation()
        args = ['evaluate_melo_spec_stoi.py', '--validation', str(path), '--output-root', str(self.root / 'scores')]
        with (patch('sys.argv', args), patch.object(evaluator.csv, 'DictWriter', side_effect=OSError('disk error')),
              patch('builtins.print')):
            self.assertEqual(evaluator.main(), 1)
        report = json.loads(next((self.root / 'scores').glob('*/report.json')).read_text())
        self.assertEqual(report['status'], 'FAIL')
        old = self.original.read_bytes()
        args = ['validate_melo_spec_pair.py', '--generation-report', str(self.report),
                '--original', str(self.original), '--protected', str(self.protected),
                '--expected-original-sha256', self.baseline, '--execution-id', 'fixture',
                '--input-kind', 'mock_contract', '--output', str(self.original)]
        with patch('sys.argv', args), patch('sys.stderr'):
            with self.assertRaises(SystemExit) as error:
                validator.main()
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(self.original.read_bytes(), old)


if __name__ == '__main__':
    unittest.main()
