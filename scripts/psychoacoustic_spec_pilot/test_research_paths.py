"""Import isolation checks; no private audio/model assets or GPU required."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PathTests(unittest.TestCase):
    def test_ignored_segment_module_is_not_used(self):
        code = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            ignored = root / 'runs/segment_noise_balance_20261124'
            ignored.mkdir(parents=True)
            (ignored / 'segment_balance.py').write_text("raise RuntimeError('ignored source imported')")
            snippet = '''import sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from research_paths import configure_paths
configure_paths(sys.argv[2],Path(sys.argv[2])/'backend')
import segment_balance,workspace_paths
assert Path(segment_balance.__file__).parent.name=='session_tools_20261125'
assert workspace_paths.ROOT==Path(sys.argv[2])
'''
            p = subprocess.run([sys.executable, '-B', '-c', snippet, str(code), str(root)],
                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)


if __name__ == '__main__':
    unittest.main()
