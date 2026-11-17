"""Isolated overlay environment: reuse existing CUDA wheels without modifying them."""
import json
from pathlib import Path
import subprocess
import sys
import sysconfig

target = Path('/home/safespeech/.venvs/melo-training-20261118')
if not (target/'pyvenv.cfg').exists():
    subprocess.run([sys.executable,'-m','venv',str(target)],check=True)
venv_config = target / 'pyvenv.cfg'
venv_config.write_text(venv_config.read_text().replace(
    'include-system-site-packages = true', 'include-system-site-packages = false'))
site = target / 'lib/python3.10/site-packages'
site.mkdir(parents=True, exist_ok=True)
pointer = site / 'safespeech_existing_environment.pth'
content = sysconfig.get_path('purelib') + '\n'
if pointer.exists() and pointer.read_text() != content:
    raise RuntimeError('Refuse to change existing environment inheritance')
pointer.write_text(content)
subprocess.run([str(target / 'bin/python'), '-m', 'pip', 'install', '-r',
                str(Path(__file__).resolve().parents[1]/'requirements-melo-training-wsl.txt')], check=True)
print(json.dumps({'environment': str(target), 'inherits': content.strip()}))
