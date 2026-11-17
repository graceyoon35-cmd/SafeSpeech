"""Reject failed or tampered runs at the SPEC/robustness boundary."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from export_spec_robustness_inputs import export_inputs
from run_korean_melo_spec import sha256


class BridgeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.run = self.root / 'source'
        self.run.mkdir()
        x = .1 * np.sin(np.arange(44100) * .1)
        for role, audio in [('clean', x), ('protected', x + .001)]:
            sf.write(self.run / f'{role}.wav', audio, 44100, subtype='PCM_16')
        identity = dict(split='test', text='수학 신호입니다.', speaker_id='mock')
        row = dict(utterance_id='fixture', input_kind='mock', iterations=1, epsilon=8/255,
                   **identity, clean=identity.copy(), protected=identity.copy())
        for role in ('clean', 'protected'):
            row[f'{role}_path'] = str(self.run / f'{role}.wav')
            row[f'{role}_sha256'] = sha256(self.run / f'{role}.wav')
        self.handoff = dict(status='NONHUMAN_NOT_FOR_FINETUNING', utterances=[row])
        self.report = dict(status='NONHUMAN_EXECUTION_PASS', input_files_unchanged=True,
                           assets_unchanged=True, handoff_count=1, human_recording_experiment=False)
        self.write()

    def write(self):
        for name, value in [('run.json', self.report), ('handoff_manifest.json', self.handoff)]:
            (self.run / name).write_text(json.dumps(value), encoding='utf-8')

    def test_exports_exact_pair_and_preserves_mock_classification(self):
        paths = export_inputs(self.run, self.root / 'out')
        payload = json.loads(paths[0].read_text())
        self.assertEqual(payload['audio']['protected'], str(self.run / 'protected.wav'))
        self.assertEqual(len(payload['conditions']), 15)
        provenance = json.loads((paths[0].parent / 'provenance.json').read_text())
        self.assertFalse(provenance['human_recording_experiment'])

    def test_rejects_failed_run(self):
        self.report['status'] = 'FAIL'
        self.write()
        with self.assertRaises(ValueError):
            export_inputs(self.run, self.root / 'out')
        self.assertFalse((self.root / 'out').exists())

    def test_rejects_changed_wav(self):
        (self.run / 'protected.wav').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            export_inputs(self.run, self.root / 'out')

    def test_rejects_false_human_claim(self):
        self.report['status'] = 'PASS'
        self.handoff['status'] = 'READY_FOR_MODEL_REVIEW'
        self.write()
        with self.assertRaisesRegex(ValueError, 'classification'):
            export_inputs(self.run, self.root / 'out')


if __name__ == '__main__':
    unittest.main()
