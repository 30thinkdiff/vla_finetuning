#!/usr/bin/env bash
# Independent pinned train environment. Dry run unless --apply and user approval.
set -euo pipefail
ROOT="/root/autodl-tmp/vla_finetuning"
SHA="7e241bd630a3719a56157a497ce5d08f244784f1"
SOURCE="$ROOT/vendor/lerobot-$SHA"
VENV="$ROOT/.venvs/train"
# LeRobot 0.6.1 at this SHA requires Python >=3.12 and NumPy >=2.
# Do not reuse .venvs/data (Python 3.10/NumPy 1.26) or Corrector.
if [[ -n "$(printenv VLA_TRAIN_PYTHON 2>/dev/null || true)" ]]; then
  PYTHON="$(printenv VLA_TRAIN_PYTHON)"
elif command -v python3.12 >/dev/null 2>&1; then
  PYTHON="$(command -v python3.12)"
elif command -v python3.13 >/dev/null 2>&1; then
  PYTHON="$(command -v python3.13)"
else
  PYTHON="/root/miniconda3/bin/python"
fi
verify_python312() {
  [[ -x "$1" ]] || { echo "STOP: Python executable missing: $1"; return 2; }
  "$1" -c 'import sys; print("PYTHON:", sys.executable, sys.version.split()[0]); sys.exit(0 if sys.version_info[:2] >= (3, 12) else 2)' || {
    echo "STOP: pinned LeRobot 0.6.1 requires Python >=3.12. Set VLA_TRAIN_PYTHON to a Python 3.12 interpreter."
    return 2
  }
}
echo "PLAN: create isolated training environment at $VENV (NOT Corrector)."
echo "PLAN: clone LeRobot source pinned at $SHA; install SmolVLA, PEFT, Torch, datasets and video dependencies."
echo "PLAN: selected Python interpreter is $PYTHON"
echo "REQUIREMENT: pinned LeRobot 0.6.1 requires Python>=3.12 and NumPy>=2,<2.3."
echo "WARNING: dependency downloads and installation can consume several GiB and substantial CPU/IO."
if [[ "$#" -ne 1 || "$1" != "--apply" ]]; then
  if [[ -x "$PYTHON" ]]; then "$PYTHON" -V || true; else echo "Candidate Python executable missing"; fi
  echo "DRY RUN: after checking resources run VLA_APPROVED_CPU_IO=YES VLA_TRAIN_PYTHON=<python3.12 path> bash scripts/bootstrap_train_env.sh --apply"
  exit 0
fi
[[ "$(printenv VLA_APPROVED_CPU_IO 2>/dev/null || true)" == "YES" ]] || { echo "STOP: missing explicit CPU/IO approval"; exit 2; }
# Version gate runs before any filesystem writes or dependency downloads.
verify_python312 "$PYTHON" || exit 2
if [[ -x "$VENV/bin/python" ]]; then verify_python312 "$VENV/bin/python" || exit 2; fi
[[ -d "$ROOT" && -d "/root/autodl-tmp" ]] || { echo "STOP: missing persistent data disk"; exit 2; }
avail="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
[[ "$avail" -ge 52428800 ]] || { echo "STOP: require >=50GiB free"; exit 2; }
command -v nvidia-smi >/dev/null || { echo "STOP: cannot verify GPU"; exit 2; }
pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "$(printf '%s' "$pids" | tr -d '[:space:]')" ]] || { echo "STOP: GPU has running compute process"; exit 2; }
mkdir -p "$ROOT/vendor" "$ROOT/.cache" "$ROOT/.venvs"
export HF_HOME="$ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export XDG_CACHE_HOME="$ROOT/.cache"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
if [[ ! -d "$SOURCE/.git" ]]; then
  [[ ! -e "$SOURCE" ]] || { echo "STOP: vendor path exists and is not a Git checkout"; exit 2; }
  mkdir -p "$SOURCE"
  git -C "$SOURCE" init
  git -C "$SOURCE" remote add origin https://github.com/huggingface/lerobot.git
  git -C "$SOURCE" fetch --depth=1 origin "$SHA"
  git -C "$SOURCE" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$SHA" ]] || { echo "STOP: LeRobot revision drift"; exit 2; }
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip --version
## Resolve declared extras from the pinned LeRobot source, not a moving PyPI lerobot.
"$VENV/bin/python" -m pip install --disable-pip-version-check -e "$SOURCE[smolvla,peft,dataset]"
"$VENV/bin/python" -m pip check
"$VENV/bin/python" - <<'PY'
from importlib.metadata import version
for name in ("lerobot","torch","peft","huggingface_hub","pyarrow","av"):
    print(name, version(name))
PY
"$VENV/bin/python" -c 'import torch, lerobot, transformers, peft, pyarrow, av; print("Imports OK; torch=", torch.__version__)'
echo "Dependencies prepared. No model loaded, no training or simulation executed."
