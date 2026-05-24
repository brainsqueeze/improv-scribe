#!/usr/bin/env bash
# Install basic-pitch into the auto-sheet-music conda env.
#
# basic-pitch 0.4+ has a base dependency on `tensorflow-macos` gated on
# `python_version > "3.11"`, but tensorflow-macos has no Python 3.13 wheel.
# We work around this by installing basic-pitch with --no-deps and relying
# on the `basic-pitch` extra in pyproject.toml for the transitives.
#
# Usage:
#   bash scripts/install_basic_pitch.sh
#
# Idempotent: re-running upgrades basic-pitch and confirms transitives.

set -euo pipefail

ENV_NAME="${ATS_CONDA_ENV:-auto-sheet-music}"

echo "[install_basic_pitch] Installing transitive deps via pyproject extra..."
conda run -n "$ENV_NAME" pip install -e ".[basic-pitch]"

echo "[install_basic_pitch] Installing basic-pitch itself with --no-deps..."
conda run -n "$ENV_NAME" pip install --no-deps 'basic-pitch>=0.4'

echo "[install_basic_pitch] Verifying import..."
conda run -n "$ENV_NAME" python -c "
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
print(f'basic-pitch model: {ICASSP_2022_MODEL_PATH}')
print('basic-pitch installed OK')
"
