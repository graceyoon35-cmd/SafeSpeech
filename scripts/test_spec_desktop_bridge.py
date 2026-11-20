from array import array
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import wave

from spec_desktop_bridge import SpecClient, SpecConfig, SpecIntegrationError, sha256, windows_to_wsl


def make_wave(path, *, rate=44100, channels=1, width=2, samples=44100, silent=False):
    with wave.open(str(path), 'wb') as stream:
        stream.setparams((channels, width, rate, 0, 'NONE', 'not compressed'))
        values = array('h', [0 if silent else 1000] * (samples * channels))
        stream.writeframes(values.tobytes())


class DesktopBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "한글 input 'sample'.wav"
        make_wave(self.source)
        self.config = SpecConfig(self.root, self.root/'jobs', execution='local')
        self.client = SpecClient(self.config)

    def backend(self, command, **kwargs):
        self.assertFalse(kwargs['shell'])
        source = Path(command[command.index('--input')+1])
        directory = Path(command[command.index('--output-dir')+1])
        directory.mkdir()
        output = directory/'protected.wav'
        output.write_bytes(source.read_bytes())
        reply = {'status':'PASS','input':str(source),'output':str(output),
                 'input_sha256':sha256(source),'output_sha256':sha256(output),
                 'iterations':10,'epsilon_units_255':8.0,'seed':52,
                 'samples':44100,'strict_linf':0.0,'sample_rate':44100,'subtype':'PCM_16'}
        self.mutate(reply, output)
        (directory/'result.json').write_text(json.dumps(reply),encoding='utf-8')
        return subprocess.CompletedProcess(command,0,'backend diagnostic\n'+json.dumps(reply)+'\n','')

    def mutate(self, reply, output):
        pass

    def test_local_result_mapping_and_unique_jobs_without_overwrite(self):
        original = self.source.read_bytes()
        with patch('spec_desktop_bridge.subprocess.run',side_effect=self.backend):
            a = self.client.protect(self.source)
            b = self.client.protect(self.source)
        self.assertEqual(a.original_path,self.source)
        self.assertIsNone(a.confidence)
        self.assertEqual(a.protected_name,"한글 input 'sample'_protected.wav")
        self.assertNotEqual(a.output_path,b.output_path)
        self.assertEqual(a.output_path.read_bytes(),original)
        self.assertEqual(self.source.read_bytes(),original)

    def test_wsl_path_conversion_preserves_spaces_unicode_and_shell_characters(self):
        self.assertEqual(windows_to_wsl("C:\\한글 folder\\a 'quote' $x.wav"),"/mnt/c/한글 folder/a 'quote' $x.wav")
        with self.assertRaises(ValueError):
            windows_to_wsl('relative.wav')
        with self.assertRaises(ValueError):
            windows_to_wsl('\\\\server\\share\\file.wav')

    def test_reject_mp3_without_creating_jobs(self):
        mp3 = self.source.with_suffix('.mp3')
        mp3.write_bytes(b'not audio')
        with self.assertRaisesRegex(ValueError,'WAV'):
            self.client.protect(mp3)
        self.assertFalse(self.config.output_root.exists())

    def test_reject_wrong_rate_stereo_short_and_silence(self):
        for options in ({'rate':16000},{'channels':2},{'samples':100},{'silent':True}):
            with self.subTest(options=options):
                make_wave(self.source,**options)
                with self.assertRaises(ValueError):
                    self.client.protect(self.source)
        self.assertFalse(self.config.output_root.exists())

    def test_reject_truncated_wav(self):
        data=self.source.read_bytes()
        self.source.write_bytes(data[:-2])
        with self.assertRaisesRegex(ValueError,'잘렸'):
            self.client.protect(self.source)

    def test_exit_failure_is_not_success_and_diagnostics_remain(self):
        with patch('spec_desktop_bridge.subprocess.run',return_value=subprocess.CompletedProcess([],1,'','failure')):
            with self.assertRaises(SpecIntegrationError):
                self.client.protect(self.source)
        self.assertEqual(len(list(self.config.output_root.glob('*/desktop_failure.json'))),1)

    def test_invalid_json_is_not_success(self):
        with patch('spec_desktop_bridge.subprocess.run',return_value=subprocess.CompletedProcess([],0,'not json','')):
            with self.assertRaises(SpecIntegrationError):
                self.client.protect(self.source)

    def test_fail_status_is_not_success(self):
        self.mutate=lambda reply,output:reply.update(status='FAIL')
        with patch('spec_desktop_bridge.subprocess.run',side_effect=self.backend):
            with self.assertRaises(SpecIntegrationError):self.client.protect(self.source)

    def test_wrong_output_hash_is_not_success(self):
        self.mutate=lambda reply,output:reply.update(output_sha256='0'*64)
        with patch('spec_desktop_bridge.subprocess.run',side_effect=self.backend):
            with self.assertRaisesRegex(SpecIntegrationError,'해시'):self.client.protect(self.source)

    def test_wrong_output_path_is_not_trusted(self):
        self.mutate=lambda reply,output:reply.update(output='/unexpected/other.wav')
        with patch('spec_desktop_bridge.subprocess.run',side_effect=self.backend):
            with self.assertRaisesRegex(SpecIntegrationError,'경로'):self.client.protect(self.source)

    def test_wrong_seed_is_not_success(self):
        self.mutate=lambda reply,output:reply.update(seed=999)
        with patch('spec_desktop_bridge.subprocess.run',side_effect=self.backend):
            with self.assertRaisesRegex(SpecIntegrationError,'설정'):self.client.protect(self.source)

    def test_independent_epsilon_check_rejects_forged_pass(self):
        def corrupt(reply,output):
            with wave.open(str(output),'wb') as stream:
                stream.setparams((1,2,44100,0,'NONE','not compressed'))
                stream.writeframes(array('h',[3000]*44100).tobytes())
            reply.update(output_sha256=sha256(output))
        self.mutate=corrupt
        with patch('spec_desktop_bridge.subprocess.run',side_effect=self.backend):
            with self.assertRaisesRegex(SpecIntegrationError,'ε'):self.client.protect(self.source)

    def test_invalid_configuration(self):
        for change in ({'iterations':0},{'epsilon':float('nan')},{'seed':-1},{'execution':'shell'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                SpecClient(replace(self.config,**change))


if __name__=='__main__':
    unittest.main()
