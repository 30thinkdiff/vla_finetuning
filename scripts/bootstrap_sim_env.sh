#!/usr/bin/env bash
# Isolated LIBERO-plus simulation bootstrap. Dry run by default; no asset download.
set -euo pipefail
ROOT="/root/autodl-tmp/vla_finetuning"
SHA="4976dc30028e805ff8094b55501d532c48fec182"
SOURCE="$ROOT/vendor/LIBERO-plus-$SHA"
LEROBOT="$ROOT/vendor/lerobot-7e241bd630a3719a56157a497ce5d08f244784f1"
VENV="$ROOT/.venvs/sim"
echo "PLAN: pinned LIBERO-plus fork $SHA, separate sim venv at $VENV"
echo "PLAN: install MuJoCo / robosuite 1.4.1 and LIBERO-plus; requires CPU, network, disk and EGL."
echo "PLAN: benchmark assets.zip must be obtained/verified separately; no implicit download."
if [[ "$#" -ne 1 || "$1" != "--apply" ]]; then
  echo "DRY RUN: after examining resources, use VLA_APPROVED_CPU_IO=YES bash scripts/bootstrap_sim_env.sh --apply"
  exit 0
fi
[[ "$(printenv VLA_APPROVED_CPU_IO 2>/dev/null || true)" == "YES" ]] || { echo "STOP: CPU/IO approval required"; exit 2; }
[[ -d "$LEROBOT/src/lerobot" ]] || { echo "STOP: pinned LeRobot source absent"; exit 2; }
[[ "$(git -C "$LEROBOT" rev-parse HEAD)" == "7e241bd630a3719a56157a497ce5d08f244784f1" ]] || { echo "STOP: LeRobot SHA mismatch"; exit 2; }
available="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
[[ "$available" -ge 52428800 ]] || { echo "STOP: disk below 50GiB"; exit 2; }
command -v nvidia-smi >/dev/null || { echo "STOP: cannot inspect GPU"; exit 2; }
pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "$(printf '%s' "$pids" | tr -d '[:space:]')" ]] || { echo "STOP: GPU occupied"; exit 2; }
mkdir -p "$ROOT/vendor" "$ROOT/.cache" "$ROOT/.venvs"
export HF_HOME="$ROOT/.cache/huggingface" PIP_CACHE_DIR="$ROOT/.cache/pip"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
if [[ ! -d "$SOURCE/.git" ]]; then
  [[ ! -e "$SOURCE" ]] || { echo "STOP: existing non-git vendor location"; exit 2; }
  mkdir -p "$SOURCE"
  git -C "$SOURCE" init
  git -C "$SOURCE" remote add origin https://github.com/sylvestf/LIBERO-plus.git
  git -C "$SOURCE" fetch --depth=1 origin "$SHA"
  git -C "$SOURCE" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$SHA" ]] || { echo "STOP: fork SHA mismatch"; exit 2; }
if [[ ! -x "$VENV/bin/python" ]]; then
  /root/miniconda3/bin/python -m venv "$VENV"
fi
# Keep the LIBERO-plus namespace separate from vanilla hf-libero. Do not install
# the LeRobot [libero] extra here, as it may replace the desired fork.
"$VENV/bin/python" -m pip install --disable-pip-version-check -e "$LEROBOT[peft]" \
  "robosuite==1.4.1" bddl easydict mujoco wand scikit-image gym
"$VENV/bin/python" -m pip uninstall -y hf-libero || true
"$VENV/bin/python" -m pip install --no-deps -e "$SOURCE"
"$VENV/bin/python" - "$SOURCE" <<'PY'
from pathlib import Path
import site
import sys
base = Path(sys.argv[1])
target = Path(site.getsitepackages()[0]) / "vla_libero_plus_source.pth"
target.write_text(str(base) + "\n" + str(base / "libero") + "\n")
print("Registered LIBERO-plus source roots:", target)
PY
"$VENV/bin/python" -m pip check
echo "Simulation Python dependencies prepared. Rendering, assets and official eval NOT verified."
