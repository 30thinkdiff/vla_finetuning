#!/usr/bin/env bash
# Exact official Recovery simulator, kept separate from 3.12 LeRobot training.
set -euo pipefail
ROOT="/root/autodl-tmp/vla_finetuning"
SHA="832d54dfc43f0c9e4240bb8d92f5fda7001bb093"
SOURCE="$ROOT/vendor/LIBERO-Recovery-$SHA"
VENV="$ROOT/.venvs/recovery_sim"
echo "PLAN: official LIBERO-Recovery $SHA -> $SOURCE"
echo "PLAN: new Python3.10 conda env $VENV, NumPy1.24.4, MuJoCo3.2.3, robosuite1.4.1"
echo "PLAN: ~230MB Git repo plus Python dependencies; no GPU inference/scenes download"
if [[ "$#" -ne 1 ]]; then
  echo "DRY RUN: VLA_APPROVED_CPU_IO=YES bash scripts/bootstrap_recovery_sim_env.sh --apply"
  exit 0
fi
if [[ "$1" != "--apply" ]]; then
  echo "DRY RUN: VLA_APPROVED_CPU_IO=YES bash scripts/bootstrap_recovery_sim_env.sh --apply"
  exit 0
fi
[[ "$(printenv VLA_APPROVED_CPU_IO 2>/dev/null || true)" == "YES" ]] || { echo "STOP: explicit CPU/IO approval required"; exit 2; }
[[ "$PWD" == "$ROOT" ]] || { echo "STOP: cd $ROOT first"; exit 2; }
[[ -d "$ROOT" && ! -L "$ROOT" ]] || { echo "STOP: unexpected project path"; exit 2; }
[[ -x /root/miniconda3/bin/conda ]] || { echo "STOP: conda not available"; exit 2; }
[[ ! -e "$SOURCE" || -d "$SOURCE/.git" ]] || { echo "STOP: unrecognized source directory"; exit 2; }
[[ ! -e "$VENV" || -x "$VENV/bin/python" ]] || { echo "STOP: existing sim environment malformed"; exit 2; }
free_kib="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
[[ "$free_kib" -ge 52428800 ]] || { echo "STOP: require >=50GiB free"; exit 2; }
command -v nvidia-smi >/dev/null || { echo "STOP: cannot inspect GPU"; exit 2; }
pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "$(printf '%s' "$pids" | tr -d '[:space:]')" ]] || { echo "STOP: GPU busy"; exit 2; }
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export XDG_CACHE_HOME="$ROOT/.cache"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$ROOT/vendor" "$ROOT/.venvs" "$PIP_CACHE_DIR"
if [[ ! -d "$SOURCE/.git" ]]; then
  mkdir "$SOURCE"
  git -C "$SOURCE" init
  git -C "$SOURCE" remote add origin https://github.com/liulin815/LIBERO-Recovery.git
  git -C "$SOURCE" fetch --depth 1 origin "$SHA"
  git -C "$SOURCE" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$SOURCE" rev-parse HEAD)" == "$SHA" ]] || { echo "STOP: Recovery SHA mismatch"; exit 2; }
[[ -f "$SOURCE/LIBERO/setup.py" ]] || { echo "STOP: expected bundled LIBERO missing"; exit 2; }
if [[ ! -x "$VENV/bin/python" ]]; then
  /root/miniconda3/bin/conda create -y -p "$VENV" python=3.10 pip
fi
[[ "$("$VENV/bin/python" -c 'import sys; print(sys.version_info[:2]==(3,10))')" == "True" ]] || { echo "STOP: requires Python3.10"; exit 2; }
"$VENV/bin/python" -m pip install --disable-pip-version-check \
  "numpy==1.24.4" "mujoco==3.2.3" "robosuite==1.4.1" \
  "bddl==1.0.1" "easydict==1.9" "gym==0.25.2" \
  "scipy==1.10.1" "opencv-python-headless==4.10.0.84" \
  pillow imageio imageio-ffmpeg tyro matplotlib
# Install only the exact bundled Recovery LIBERO source, never hf-libero.
"$VENV/bin/python" -m pip install --disable-pip-version-check --no-deps -e "$SOURCE/LIBERO"
"$VENV/bin/python" -m pip check
PYTHONPATH="$SOURCE/LIBERO:$SOURCE" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  "$VENV/bin/python" - <<'PY'
import sys, numpy, mujoco, robosuite
from libero.libero.envs import OffScreenRenderEnv
assert sys.version_info[:2] == (3,10)
assert numpy.__version__ == "1.24.4"
assert mujoco.__version__ == "3.2.3"
print({
    "status":"ISOLATED_LIBERO_RECOVERY_IMPORTS_READY_NO_SCENE_EXECUTED",
    "python":sys.version.split()[0],
    "numpy":numpy.__version__,"mujoco":mujoco.__version__,
    "robosuite":robosuite.__version__,
})
PY
echo "DONE: isolated sim imports available. Real scene reset/evaluation not yet executed."
