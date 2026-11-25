"""Resolve versioned helpers separately from private, existing experiment data."""
import os
from pathlib import Path
import sys


def configure_paths(workspace, backend=None):
    """Configure imports only. Never create data, load models, or run experiments."""
    workspace = Path(workspace).resolve()
    backend = Path(backend or os.environ.get(
        'SAFESPEECH_LEGACY_WORKSPACE',
        str(workspace.parent / 'SafeSpeech-desktop-integration-20261118'))).resolve()
    shared = Path(__file__).resolve().parents[1] / 'session_tools_20261125'
    if not (shared / 'segment_balance.py').is_file():
        raise ImportError('Missing versioned scripts/session_tools_20261125/segment_balance.py')
    loaded = sys.modules.get('workspace_paths')
    if loaded is not None and (loaded.ROOT != workspace or loaded.LEGACY != backend):
        raise RuntimeError('Workspace already configured differently; use a fresh Python process')
    os.environ['SAFESPEECH_WORKSPACE'] = str(workspace)
    os.environ['SAFESPEECH_LEGACY_WORKSPACE'] = str(backend)
    sys.path[:0] = [str(shared), str(backend / 'scripts')]
    return backend
