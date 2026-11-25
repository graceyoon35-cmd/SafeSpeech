"""Safety contracts that matter when replaying private archived experiments."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from runner import reserve_output


class AnalysisTests(unittest.TestCase):
    def test_all_imports_without_data_or_output(self):
        with tempfile.TemporaryDirectory() as td:
            snippet = '''import importlib,sys
sys.path.insert(0,sys.argv[1])
for name in ('analyze_existing_only','package_existing','analyse_stages','report_and_verify','report_results','verify_saved_pcm'):
    importlib.import_module(name)
'''
            p = subprocess.run([sys.executable, '-B', '-c', snippet, str(Path(__file__).resolve().parent)],
                               cwd=td, env=dict(os.environ, SAFESPEECH_WORKSPACE=td, PYTHONDONTWRITEBYTECODE='1'),
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(list(Path(td).iterdir()), [])

    def test_existing_output_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sentinel = root / 'keep.txt'
            sentinel.write_text('unchanged')
            with self.assertRaises(FileExistsError):
                reserve_output(root, root)
            self.assertEqual(sentinel.read_text(), 'unchanged')

    def test_historical_output_root_is_refused(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(ValueError):
                reserve_output(root / 'runs/psychoacoustic_spec_refinement_20261126/new', root)
            self.assertEqual(list(root.iterdir()), [])

    def test_entrypoints_help_without_assets(self):
        folder = Path(__file__).resolve().parent
        for name in ('analyze_existing_only', 'package_existing', 'analyse_stages',
                     'report_and_verify', 'report_results', 'verify_saved_pcm'):
            p = subprocess.run([sys.executable, '-B', str(folder / (name+'.py')), '--help'],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)


if __name__ == '__main__':
    unittest.main()
