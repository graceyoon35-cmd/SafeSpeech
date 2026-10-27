"""Model-free contract tests; generated fixtures are never real cloning results."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from run_experiment import sha256, ROOT
from run_xtts_pair import run_pair
from xtts_inference import XttsSynthesizer, LocalXttsBackend, XttsError, CudaMemoryError


class FakeBackend:
    def __init__(self):
        self.calls = []

    def synthesize(self, reference, text, language, seed):
        self.calls.append((reference, text, language, seed))
        wave = .1 * np.sin(2 * np.pi * 220 * np.arange(24000) / 24000)
        return wave, 24000, {'device': 'mock', 'load_memory': None, 'inference_memory': None}


class XttsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        wave = .1 * np.sin(2 * np.pi * 180 * np.arange(48000) / 24000)
        self.clean = self.folder / 'clean.wav'
        self.protected = self.folder / 'protected.wav'
        # Synthetic fixture only; this is not a SPEC-protected sample.
        sf.write(self.clean, wave, 24000, subtype='FLOAT')
        sf.write(self.protected, wave * .99, 24000, subtype='FLOAT')
        self.fake = FakeBackend()
        self.synth = XttsSynthesizer(consent_confirmed=True, cpml_accepted=True, backend=self.fake)
        self.data = json.loads((ROOT / 'experiments/xtts_pair_pending.json').read_text())
        self.data.update(consent_confirmed=True, cpml_accepted=True)
        self.data['references'] = {'clean': 'clean.wav', 'protected': 'protected.wav'}
        self.manifest = self.folder / 'manifest.json'

    def run_manifest(self):
        self.manifest.write_text(json.dumps(self.data))
        return run_pair(self.manifest, self.folder / 'outputs', backend=self.fake)

    def test_ko_en_pair_and_input_preservation(self):
        before = [sha256(self.clean), sha256(self.protected)]
        path, report = self.run_manifest()
        self.assertEqual(report['status'], 'MOCK_PASS')
        self.assertEqual(report['backend'], 'mock')
        self.assertEqual(len(report['outputs']), 4)
        self.assertEqual(report['reference_hashes_before'], report['reference_hashes_after'])
        self.assertEqual(before, [sha256(self.clean), sha256(self.protected)])
        self.assertEqual(json.loads(path.read_text()), report)
        for index in [0, 2]:
            clean, protected = self.fake.calls[index:index+2]
            self.assertEqual(clean[1:], protected[1:])
        self.assertEqual([call[2] for call in self.fake.calls], ['ko', 'ko', 'en', 'en'])
        for output in report['outputs']:
            self.assertTrue(Path(output['output_path']).is_file())
            self.assertEqual(sf.info(output['output_path']).samplerate, 24000)

    def test_missing_reference_rejected_before_model(self):
        with self.assertRaises(FileNotFoundError):
            self.synth.synthesize(self.folder / 'missing.wav', '안녕', 'ko', self.folder / 'out.wav')
        self.assertFalse(self.fake.calls)

    def test_invalid_language_text_seed(self):
        for text, language, seed in [('안녕', 'KO', 42), ('Hello', 'fr', 42),
                                     (' ', 'en', 42), ('Hello', 'en', -1), ('Hello', 'en', True)]:
            with self.subTest(language=language, seed=seed), self.assertRaises(ValueError):
                self.synth.synthesize(self.clean, text, language, self.folder / 'out.wav', seed=seed)
        self.assertFalse(self.fake.calls)

    def test_existing_output_is_never_overwritten(self):
        before = sha256(self.protected)
        with self.assertRaises(FileExistsError):
            self.synth.synthesize(self.clean, 'Hello', 'en', self.protected)
        self.assertEqual(sha256(self.protected), before)

    def test_permissions_are_explicit(self):
        with self.assertRaises(ValueError):
            XttsSynthesizer(backend=self.fake)
        self.data['consent_confirmed'] = False
        with self.assertRaises(ValueError):
            self.run_manifest()
        self.assertFalse((self.folder / 'outputs').exists())

    def test_reference_alignment_and_bad_audio(self):
        sf.write(self.protected, np.ones(16000) * .1, 16000)
        with self.assertRaises(ValueError):
            self.run_manifest()
        sf.write(self.clean, np.zeros(24000), 24000)
        with self.assertRaises(ValueError):
            self.synth.synthesize(self.clean, 'Hello', 'en', self.folder / 'out.wav')

    def test_oom_is_recorded_as_failure(self):
        self.fake.synthesize = lambda *args: (_ for _ in ()).throw(CudaMemoryError('CUDA OOM (mock)'))
        with self.assertRaises(CudaMemoryError):
            self.run_manifest()
        reports = list((self.folder / 'outputs').glob('*/report.json'))
        self.assertEqual(len(reports), 1)
        report = json.loads(reports[0].read_text())
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['error_type'], 'CudaMemoryError')
        self.assertEqual(report['outputs'], [])

    def test_cuda_unavailable_and_load_oom_translation(self):
        model = self.folder / 'model'
        model.mkdir()
        for name in ['model.pth', 'config.json', 'vocab.json']:
            (model / name).write_text('fixture')
        class FakeOOM(RuntimeError):
            pass
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False,
                                     OutOfMemoryError=FakeOOM, reset_peak_memory_stats=lambda d: None))
        config = SimpleNamespace(load_json=lambda p: None)
        def fail(config):
            raise FakeOOM('load OOM')
        modules = {'torch': fake_torch,
                   'TTS.tts.configs.xtts_config': SimpleNamespace(XttsConfig=lambda: config),
                   'TTS.tts.models.xtts': SimpleNamespace(Xtts=SimpleNamespace(init_from_config=fail))}
        with patch.dict('sys.modules', modules):
            with self.assertRaisesRegex(XttsError, 'CUDA unavailable'):
                LocalXttsBackend(model, 'cuda:0')
            fake_torch.cuda.is_available = lambda: True
            with self.assertRaises(CudaMemoryError):
                LocalXttsBackend(model, 'cuda:0')


if __name__ == '__main__':
    unittest.main()
