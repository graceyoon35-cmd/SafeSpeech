"""Versioned imports with explicit private workspace/backend overrides."""
import os
from pathlib import Path
import sys


def setup():
    code = Path(__file__).resolve().parents[2]
    root = Path(os.environ.get('SAFESPEECH_WORKSPACE', str(code))).resolve()
    backend = Path(os.environ.get('SAFESPEECH_LEGACY_WORKSPACE',
                   str(root.parent / 'SafeSpeech-desktop-integration-20261118'))).resolve()
    sys.path[:0] = [str(code / 'scripts/session_tools_20261125'),
                   str(code / 'scripts/psychoacoustic_spec_pilot')]
    return code, root, backend
