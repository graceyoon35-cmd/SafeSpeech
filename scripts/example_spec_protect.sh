#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "Usage: bash scripts/example_spec_protect.sh INPUT.wav NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
task_python="${SPEC_PYTHON:-/home/safespeech/.venvs/melo-training-20261118/bin/python}"
task_assets="${SPEC_ASSETS:-/home/safespeech/models/melo-spec-pinned-20261118}"
exec "$task_python" scripts/protect_wav.py --input "$1" --output-dir "$2" \
  --assets "$task_assets" --device cuda --iterations 10 --epsilon 8 --seed 52
