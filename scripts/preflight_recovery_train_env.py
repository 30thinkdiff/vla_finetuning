"""Read-only AutoDL pinned LeRobot/SmolVLA train environment preflight.

No Python packages installed, GPU allocation, network access, or filesystem
writes. Do not use its output as proof training environment is ready: only
bootstrap + imports + real processor smoke can establish that.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def inspect_python(path):
    if not path or not Path(path).is_file():
        return None
    code = (
        "import json,sys;"
        "print(json.dumps({'version':list(sys.version_info[:3]),"
        "'executable':sys.executable}))"
    )
    try:
        completed = subprocess.run(
            [str(path), "-c", code], capture_output=True, text=True,
            timeout=5, check=False)
        if completed.returncode != 0:
            return {"candidate":str(path), "error":completed.stderr.strip()[-200:]}
        data = json.loads(completed.stdout)
        data["candidate"] = str(path)
        data["suitable_for_pinned_lerobot"] = tuple(data["version"][:2]) >= (3, 12)
        return data
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        return {"candidate":str(path), "error":str(exc)[:180]}


def main():
    candidates = []
    for candidate in (
        os.getenv("VLA_TRAIN_PYTHON"),
        ROOT/".venvs/train/bin/python",
        shutil.which("python3.12"), shutil.which("python3.13"),
        "/root/miniconda3/bin/python",
        ROOT/".venvs/data/bin/python",
    ):
        if candidate and str(candidate) not in candidates:
            candidates.append(str(candidate))
    conda_dir=Path("/root/miniconda3/envs")
    if conda_dir.is_dir():
        for candidate in sorted(conda_dir.glob("*/bin/python"))[:20]:
            if str(candidate) not in candidates:
                candidates.append(str(candidate))
    interpreters = [x for x in (inspect_python(p) for p in candidates) if x is not None]
    free=shutil.disk_usage(ROOT)
    data = {
        "status":"READ_ONLY_TRAIN_ENV_PREFLIGHT_NOT_INSTALLED",
        "lerobot_revision":"7e241bd630a3719a56157a497ce5d08f244784f1",
        "lerobot_minimum_python":"3.12",
        "lerobot_numpy_constraint":">=2,<2.3",
        "environment_isolation":"Use .venvs/train only; do not modify .venvs/data or Corrector",
        "interpreters": interpreters,
        "valid_interpreters":[p["executable"] for p in interpreters
                              if p.get("suitable_for_pinned_lerobot")],
        "free_space_gib": round(free.free / 2**30, 2),
        "bootstrap_requires_min_free_gib": 50,
        "conda_command": shutil.which("conda") or
                         ("/root/miniconda3/bin/conda" if
                          Path("/root/miniconda3/bin/conda").is_file() else None),
        "train_env_python_exists":(ROOT/".venvs/train/bin/python").is_file(),
        "gpu_checker":"nvidia-smi",
        "gpu_checker_available": bool(shutil.which("nvidia-smi")),
        "note":"No Torch import, CUDA allocation, dependency installation or remote access performed.",
    }
    print(json.dumps(data,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
