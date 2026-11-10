"""Model-free contract tests. Fixture signals are never real Korean recordings."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
import run_korean_melo_spec as R


class KoreanContractTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.wav = self.root/'fixture.wav'
        self.wave = np.random.default_rng(1).uniform(-.15, .15, 48000)
        sf.write(self.wav, self.wave, 24000, subtype='FLOAT')
        self.row = dict(utterance_id='ko_001', input_kind='mock', language='ko',
            speaker_id='anonymous', model_speaker='KR', split='train', text='계약 검사용 가상 대본입니다.',
            transcript_verified=True, transcript_reference='fixture declaration only',
            original_path='fixture.wav', original_sha256=R.sha256(self.wav), sample_rate=24000,
            source='mathematical fixture, no speech', research_use_permitted=True,
            permission_scope='test_only', permission_reference='generated fixture')
        self.manifest = self.root/'input.json'

    def check(self, row=None, allow=True):
        self.manifest.write_text(json.dumps({'schema_version': 1, 'utterances': [row or self.row]}))
        return R.check_manifest(self.manifest, allow_nonhuman=allow)

    def test_valid_mock_and_default_rejection(self):
        self.assertEqual(self.check()[0]['input_kind'], 'mock')
        with self.assertRaisesRegex(ValueError, 'Nonhuman'):
            self.check(allow=False)

    def test_permissions_and_transcript_required(self):
        for field, value in [('research_use_permitted', False), ('permission_reference', ''),
                             ('transcript_verified', False), ('transcript_reference', '')]:
            row = {**self.row, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.check(row)

    def test_human_permission_scope(self):
        row = {**self.row, 'input_kind': 'human_recording'}
        with self.assertRaisesRegex(ValueError, 'Permission must cover'):
            self.check(row)
        row['permission_scope'] = 'spec_and_finetuning'
        self.assertEqual(self.check(row)[0]['input_kind'], 'human_recording')
        # This checks declarations only: waveform cannot establish consent or natural speech.

    def test_metadata_rejections(self):
        for field, value in [('utterance_id', '../bad'), ('language', 'en'), ('split', 'unknown'),
                             ('model_speaker', 'missing'), ('text', 'no Korean'), ('text', '안녕|하세요'),
                             ('speaker_id', ''), ('sample_rate', 44100), ('original_sha256', '0'*64)]:
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.check({**self.row, field: value})

    def test_duplicate_ids_sources_and_content(self):
        for duplicate in [self.row, {**self.row, 'utterance_id': 'ko_002', 'split': 'test'}]:
            self.manifest.write_text(json.dumps({'schema_version': 1, 'utterances': [self.row, duplicate]}))
            with self.assertRaises(ValueError):
                R.check_manifest(self.manifest, allow_nonhuman=True)
        other = self.root/'other.wav'
        other.write_bytes(self.wav.read_bytes())
        row = {**self.row, 'utterance_id': 'ko_002', 'original_path': 'other.wav', 'split': 'test'}
        self.manifest.write_text(json.dumps({'schema_version': 1, 'utterances': [self.row, row]}))
        with self.assertRaisesRegex(ValueError, 'Duplicate recording content'):
            R.check_manifest(self.manifest, allow_nonhuman=True)

    def test_invalid_audio(self):
        for wave, rate in [(np.zeros(48000), 24000), (self.wave[:100], 24000),
                           (np.column_stack([self.wave, self.wave]), 24000),
                           (np.full(48000, np.nan), 24000), (np.full(48000, 1.1), 24000)]:
            sf.write(self.wav, wave, rate, subtype='FLOAT')
            with self.assertRaises(ValueError):
                self.check({**self.row, 'original_sha256': R.sha256(self.wav)})

    def pair(self):
        protected = self.root/'protected.wav'
        sf.write(protected, self.wave-.003, 24000, subtype='PCM_16')
        identity = {k: self.row[k] for k in ('split', 'text', 'speaker_id')}
        return {**self.row, 'clean_path': str(self.wav), 'protected_path': str(protected),
                'clean_sha256': R.sha256(self.wav), 'protected_sha256': R.sha256(protected),
                'clean': identity.copy(), 'protected': identity.copy(), 'epsilon': 8/255, 'iterations': 1}

    def test_pair_metadata_and_condition_consistency(self):
        pair = self.pair()
        self.assertTrue(R.validate_handoff([pair]))
        for key in ('split', 'text', 'speaker_id'):
            bad = copy.deepcopy(pair)
            bad['protected'][key] = 'mismatch'
            with self.subTest(key=key), self.assertRaises(ValueError):
                R.validate_handoff([bad])
        other = copy.deepcopy(pair)
        other['iterations'] = 10
        other['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            R.validate_handoff([pair, other])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            R.validate_handoff([pair, pair])

    def test_pair_mutation_and_epsilon(self):
        pair = self.pair()
        pair['epsilon'] = .00001
        with self.assertRaisesRegex(ValueError, 'epsilon'):
            R.validate_handoff([pair])
        pair = self.pair()
        with Path(pair['protected_path']).open('ab') as f:
            f.write(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash'):
            R.validate_handoff([pair])

    def test_exclusive_json_output(self):
        out = self.root/'out.json'
        R.save(out, {'old': True})
        with self.assertRaises(FileExistsError):
            R.save(out, {'old': False})
        self.assertEqual(json.loads(out.read_text()), {'old': True})

    def test_invalid_run_settings_before_model_or_outputs(self):
        for kwargs in [dict(seed=-1), dict(epsilon=float('nan')), dict(iterations=(1, 1)), dict(iterations=(0,))]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                R.execute(self.manifest, self.root, **kwargs)

    def test_finetune_export_rejects_nonhuman(self):
        with self.assertRaisesRegex(ValueError, 'Nonhuman'):
            R.export_finetune_inputs(self.root/'out', [self.pair()])
        self.assertFalse((self.root/'out').exists())

    def test_split_export_preserves_wave_and_transcript(self):
        pair = self.pair()
        pair['input_kind'] = 'human_recording'  # Declaration fixture; no real recording used.
        second = copy.deepcopy(pair)
        second['iterations'] = 10
        R.export_finetune_inputs(self.root/'out', [pair, second])
        for name in ['clean', 'SPEC_1', 'SPEC_10']:
            base = self.root/'out'/'finetune_inputs'/name/'train'/'anonymous'
            self.assertEqual(len(list((base/'wavs').glob('*.wav'))), 1)
            self.assertIn(pair['text'], (base/'transcripts.csv').read_text())
        self.assertEqual(R.sha256(self.root/'out'/'finetune_inputs'/'clean'/'train'/'anonymous'/'wavs'/'ko_001.wav'),
                         pair['clean_sha256'])
        with self.assertRaises(FileExistsError):
            R.export_finetune_inputs(self.root/'out', [pair, second])

    def test_failed_run_records_error_without_handoff(self):
        self.check()
        with patch.object(R, 'ROOT', self.root), \
             patch.object(R, 'check_assets', return_value=(self.root, {}, {})), \
             patch.object(R, 'git', return_value=b'test-head'), \
             patch.object(R, 'load_backend', side_effect=RuntimeError('model load failure')):
            with self.assertRaisesRegex(RuntimeError, 'model load failure'):
                R.execute(self.manifest, self.root, allow_nonhuman=True)
        runs = list((self.root/'runs').iterdir())
        self.assertEqual(len(runs), 1)
        report = json.loads((runs[0]/'run.json').read_text())
        self.assertEqual(report['status'], 'FAIL')
        self.assertFalse((runs[0]/'handoff_manifest.json').exists())
        self.assertEqual(R.sha256(self.wav), self.row['original_sha256'])

    def test_config_and_pinned_weights_rejected(self):
        with patch.object(R, 'git', side_effect=[(R.OFFICIAL+'\n').encode(), b'']), \
             patch.object(R, 'sha256', return_value='wrong'):
            with self.assertRaisesRegex(ValueError, 'model asset mismatch'):
                R.check_assets(self.root)


if __name__ == '__main__':
    unittest.main()
