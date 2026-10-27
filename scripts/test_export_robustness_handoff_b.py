"""Portable handoff integrity checks; protected input is a synthetic fixture."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scipy.io import wavfile
import soundfile as sf

from export_robustness_handoff_b import child, export_handoff, verify_handoff
from run_robustness_b import ROOT
from run_robustness_sweep_b import run_sweep


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        audio, rate = sf.read(ROOT / 'data/LibriTTS/5339/5339_14134_000092_000003.wav')
        wavfile.write(self.directory / 'original.wav', rate, audio)
        wavfile.write(self.directory / 'fixture.wav', rate, audio * .99)
        manifest = {'schema_version': 1, 'experiment_id': 'synthetic_fixture_handoff',
                    'audio': {'original': 'original.wav', 'protected': 'fixture.wav'},
                    'conditions': [{'name': 'gaussian_noise', 'seed': 42, 'std': .005},
                                   {'name': 'resample', 'target_hz': 16000}]}
        path = self.directory / 'manifest.json'
        path.write_text(json.dumps(manifest))
        self.summary, _ = run_sweep(path, self.directory / 'runs')

    def test_zip_relocation_checksums_and_pair_list(self):
        folder, archive = export_handoff(self.summary, self.directory / 'exports')
        verify_handoff(folder)
        with zipfile.ZipFile(archive) as zipped:
            self.assertIsNone(zipped.testzip())
            zipped.extractall(self.directory / 'moved')
        moved = self.directory / 'moved' / folder.name
        verify_handoff(moved)
        with (moved / 'evaluation_pairs.csv').open(encoding='utf-8-sig', newline='') as stream:
            pairs = list(csv.DictReader(stream))
        self.assertEqual(len(pairs), 4)
        for pair in pairs:
            for key in ('transformed_original', 'transformed_protected', 'input_original', 'input_protected'):
                self.assertTrue(child(moved, pair[key]).is_file())
            self.assertEqual(pair['evaluation_status'], 'not_measured')
        for path in list(moved.glob('*.json')) + list(moved.glob('*.csv')):
            self.assertNotIn(str(self.directory), path.read_text(encoding='utf-8-sig'))
        (moved / 'inputs/original.wav').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            verify_handoff(moved)

    def test_changed_source_artifact_is_rejected(self):
        with (self.summary.parent / 'artifacts.csv').open(encoding='utf-8-sig', newline='') as stream:
            row = next(csv.DictReader(stream))
        child(self.summary.parent, row['wav_path']).write_bytes(b'changed')
        with self.assertRaises(ValueError):
            export_handoff(self.summary, self.directory / 'exports')
        self.assertFalse((self.directory / 'exports').exists())

    def test_changed_input_is_rejected(self):
        (self.directory / 'fixture.wav').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            export_handoff(self.summary, self.directory / 'exports')

    def test_unsafe_paths_and_parameter_mismatch_are_rejected(self):
        for path in ('../escape.wav', '..\\escape.wav', '/absolute.wav', 'C:/absolute.wav'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                child(self.directory, path)
        csv_path = self.summary.parent / 'artifacts.csv'
        with csv_path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        rows[0]['parameters_json'] = json.dumps({'name': 'resample', 'target_hz': 8000})
        with csv_path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(ValueError):
            export_handoff(self.summary, self.directory / 'exports')


if __name__ == '__main__':
    unittest.main()
