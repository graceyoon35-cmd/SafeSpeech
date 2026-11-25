"""Paths for archived session tools; importing never runs an experiment.

Set SAFESPEECH_WORKSPACE to the existing experiment workspace when running from
an isolated code worktree. SAFESPEECH_LEGACY_WORKSPACE optionally overrides the
read-only legacy code/data root. Existing result files are never overwritten.
"""
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
ROOT = Path(os.environ.get("SAFESPEECH_WORKSPACE", str(REPOSITORY))).resolve()
LEGACY = Path(os.environ.get("SAFESPEECH_LEGACY_WORKSPACE", str(ROOT.parent / "SafeSpeech-desktop-integration-20261118"))).resolve()
STUDY = ROOT / "runs/segment_noise_balance_20261124"
LISTENING = ROOT / "runs/human_listening_20261125"
CLOSEOUT = ROOT / "runs/session_closeout_20261125"
