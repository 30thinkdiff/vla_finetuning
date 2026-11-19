#!/usr/bin/env bash
# Independent pinned train environment. Dry run unless --apply and user approval.
set -euo pipefail
ROOT="/root/autodl-tmp/vla_finetuning"
SHA="7e241bd630a3719a56157a497ce5d08f244784f1"
SOURCE="$ROOT/vendor/lerobot-$SHA"
VENV="$ROOT/.venvs/train"
PYTHON="/root/miniconda3/bin/python"
echo "PLAN: create isolated training environment at $VENV (NOT Corrector)."
echo "PLAN: clone LeRobot source pinned at $SHA; install PEFT, Torch, PyArrow and PyAV."
echo "WARNING: dependency downloads and installation can consume several GiB and substantial CPU/IO."
if [[ "$#" -ne 1 || "$1" != "--apply" ]]; then
  echo "DRY RUN: after checking CPU/GPU/network/disk run VLA_APPROVED_CPU_IO=YES bash scripts/bootstrap_train_env.sh --apply"
  exit 0
fi
[[ "$VLA_APPROVED_CPU_IO" == "YES" ]] || { echo "STOP: missing explicit CPU/IO approval"; exit 2; }
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
"$VENV/bin/python" -m pip install --disable-pip-version-check -e "$SOURCE[peft]" "pyarrow>=17" av
"$VENV/bin/python" -m pip check
"$VENV/bin/python" - <<'PY'
from importlib.metadata import version
for name in ("lerobot","torch","peft","huggingface_hub","pyarrow","av"):
    print(name, version(name))
PY
echo "Dependencies prepared. No model loaded, no training or simulation executed."
