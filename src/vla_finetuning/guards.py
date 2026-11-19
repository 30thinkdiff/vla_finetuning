"""Fail-closed resource and provenance gates shared by downloads, training and evaluation.

No background processes or GPU access on import. All expensive operations require
an explicit opt-in flag at the command line and an externally checked machine.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from .core import ContractError, read_json, GIB

SHA40 = re.compile(r"^[0-9a-f]{40}$")
DATA_DIR = Path("/root/autodl-tmp/vla_finetuning")
MIN_DISK_GIB = 30


def require_sha(sha, name):
    if not isinstance(sha, str) or not SHA40.fullmatch(sha):
        raise ContractError(f"{name} requires a real pinned 40-character lowercase commit SHA")
    if len(set(sha)) == 1:
        raise ContractError(f"{name} contains a placeholder SHA")
    return sha


def require_directory(path, *, name="path"):
    value = Path(path).expanduser().resolve()
    if not value.is_dir():
        raise ContractError(f"{name} directory missing: {value}")
    return value


def require_unused_output(path):
    dest = Path(path).expanduser().resolve()
    if dest.exists():
        raise ContractError(f"Output path already exists (no overwrite): {dest}")
    if dest.is_relative_to(Path("/root/autodl-tmp/corrector")):
        raise ContractError("Corrector assets are immutable from this project")
    return dest


def require_disk(path, *, free_gib=MIN_DISK_GIB):
    path = Path(path).expanduser()
    parent = path
    while not parent.exists():
        if parent == parent.parent:
            raise ContractError("Output location has no existing ancestor")
        parent = parent.parent
    free = shutil.disk_usage(parent).free
    if free < free_gib * GIB:
        raise ContractError(f"Insufficient disk: {free/GIB:.2f}GiB < {free_gib}GiB")
    return {"disk_free_gib": round(free/GIB, 2), "min_required_gib": free_gib}


def compute_pids(*, runner=None):
    """Get active GPU compute PIDs; probe failures are errors, not permission."""
    if runner is None:
        runner = subprocess.run
    try:
        r = runner(["nvidia-smi", "--query-compute-apps=pid",
                    "--format=csv,noheader,nounits"],
                   capture_output=True, text=True, check=True, timeout=12)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContractError("Cannot inspect GPU compute processes") from exc
    pids = []
    for line in r.stdout.splitlines():
        x = line.strip()
        if not x or x.lower().startswith("no running"):
            continue
        if not x.isdecimal():
            raise ContractError("Unexpected GPU process listing; stop: " + x[:100])
        pids.append(int(x))
    return pids


def require_gpu_idle(*, runner=None):
    pids = compute_pids(runner=runner)
    if pids:
        raise ContractError(f"GPU busy with {len(pids)} process(es): {pids}; refusing to compete with Corrector")
    return True


def require_explicit(flag, command):
    if not flag:
        raise ContractError(f"{command} requires --apply after independently checking machine resources")


def ensure_cache_on_data_disk(project_root, *, env=None):
    """Environment for child subprocesses only, does not mutate global os.environ."""
    project_root = require_directory(project_root, name="project_root")
    if not str(project_root).startswith("/root/autodl-tmp/"):
        raise ContractError("This AutoDL profile requires the data-disk mount /root/autodl-tmp")
    env = dict(os.environ if env is None else env)
    hf = project_root / ".cache" / "huggingface"
    env["HF_HOME"] = str(hf)
    env["HF_HUB_CACHE"] = str(hf / "hub")
    env["HF_DATASETS_CACHE"] = str(hf / "datasets")
    env["XDG_CACHE_HOME"] = str(project_root / ".cache")
    env["PIP_CACHE_DIR"] = str(project_root / ".cache" / "pip")
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def load_protocol(path):
    """Cross-file contract, preflight only; no model loading or network."""
    config = read_json(path)
    if not isinstance(config, dict):
        raise ContractError("Protocol JSON must be an object")
    for section in ("software", "model", "dataset", "train", "evaluation"):
        if not isinstance(config.get(section), dict):
            raise ContractError(f"Missing protocol section: {section}")
    software, model, data, train, ev = (config[k] for k in
                                        ("software", "model", "dataset", "train", "evaluation"))
    require_sha(software.get("lerobot_sha"), "lerobot_sha")
    require_sha(software.get("libero_plus_sha"), "libero_plus_sha")
    require_sha(model.get("revision"), "model.revision")
    require_sha(data.get("revision"), "dataset.revision")
    if model.get("repo_id") not in ("lerobot/smolvla_libero", "lerobot/smolvla_base"):
        raise ContractError("Unsupported model; use an approved checkpoint and audit provenance")
    if data.get("repo_id") not in ("lerobot/libero_plus", "Sylvest/libero_plus_lerobot"):
        raise ContractError("Unknown dataset provenance; audit separately before adding")
    episodes = data.get("train_episodes")
    if (not isinstance(episodes, list) or not episodes
        or len(set(map(str, episodes))) != len(episodes)
        or any(type(i) is not int or i < 0 for i in episodes)):
        raise ContractError("train_episodes must be distinct nonnegative integer IDs")
    for section, key in ((train, "steps"), (train, "batch_size"), (train, "save_freq"),
                         (ev, "action_horizon"), (ev, "n_episodes"), (ev, "max_steps")):
        if type(section.get(key)) is not int or section[key] < 1:
            raise ContractError(f"Positive integer required: {key}")
    if train["save_freq"] > train["steps"]:
        raise ContractError("save_freq must not exceed total steps for first run")
    if not isinstance(train.get("output_dir"), str) or not Path(train["output_dir"]).is_absolute():
        raise ContractError("train.output_dir must be absolute")
    if ev.get("suite") not in ("libero_object", "libero_spatial", "libero_goal", "libero_10"):
        raise ContractError("Unsupported LIBERO-plus suite")
    ids = ev.get("variant_task_ids")
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or any(
        type(i) is not int or i < 0 for i in ids
    ):
        raise ContractError("variant_task_ids must be a nonempty list of registered simulation IDs")
    if ev.get("control_mode") not in ("relative", "absolute"):
        raise ContractError("control_mode must be independently verified as relative or absolute")
    return config


def shell_render(args):
    """Preview a command without invoking a shell."""
    import shlex
    return shlex.join([str(x) for x in args])
