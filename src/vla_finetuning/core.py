"""CPU-only metadata, provenance split, preflight, and explicit GPU launch guards."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SHA40 = re.compile(r"^[a-fA-F0-9]{40}$")
GIB = 1024**3


class ContractError(ValueError):
    """Input contradicts the frozen training/data contract."""


def json_digest(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_new_json(path, obj):
    """Create without overwriting a possibly important previous experiment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def preflight(data_root):
    """Read-only: no torch import, GPU access, network, or downloads."""
    root = Path(data_root).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise ContractError("data_root must already exist: " + str(root))
    usage = shutil.disk_usage(root)
    return {
        "status": "read_only",
        "python": sys.version.split()[0],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root),
        "disk_total_gib": round(usage.total / GIB, 2),
        "disk_used_gib": round(usage.used / GIB, 2),
        "disk_free_gib": round(usage.free / GIB, 2),
        "inodes_note": "Use df -i to check inodes; this function only checks bytes.",
        "gpu_note": "No GPU inspected or reserved; ongoing Corrector fit remains untouched.",
        "disk_safe_for_large_download": False,
    }


def audit_lerobot_info(info_path):
    """Inspect only meta/info.json. No claim of real sample decoding or domain labels."""
    info_path = Path(info_path)
    obj = read_json(info_path)
    if not isinstance(obj, dict) or not isinstance(obj.get("features"), dict):
        raise ContractError("Expected LeRobot meta/info.json with features object")
    features = obj["features"]
    images = sorted(k for k in features if k.startswith("observation.images."))
    expected = ["observation.state", "action"]
    missing = [key for key in expected if key not in features]
    issues = []
    if missing:
        issues.append("Missing key(s): " + ", ".join(missing))
    if len(images) != 2:
        issues.append("Expected two image keys, found " + str(len(images)))
    specs = {}
    for key in expected + images:
        if key in features:
            value = features[key]
            if not isinstance(value, dict):
                raise ContractError("Feature spec must be object: " + key)
            specs[key] = {"dtype": value.get("dtype"), "shape": value.get("shape")}
    fps = obj.get("fps")
    if not isinstance(fps, (int, float)) or fps <= 0:
        issues.append("FPS absent or invalid")
    if not isinstance(obj.get("total_episodes"), int):
        issues.append("Episode count absent")
    return {
        "source_file": str(info_path.resolve()),
        "source_sha256": json_digest(info_path),
        "codebase_version": obj.get("codebase_version"),
        "fps": fps,
        "total_episodes": obj.get("total_episodes"),
        "total_frames": obj.get("total_frames"),
        "total_tasks": obj.get("total_tasks"),
        "features": specs,
        "image_keys": images,
        "schema_issues": issues,
        "provenance_warning": "Task index alone does NOT identify camera perturbation, source trajectory, "
                              "or recovery failure scene. Sample decoding and label/source audit pending.",
        "action_time_warning": "Compare dataset FPS with actual control step frequency; "
                               "do not infer timing from action chunk shape.",
    }


def split_provenance_csv(csv_path, seed=42, train_fraction=0.6, val_fraction=0.2):
    """Strict group holdout by source_group, separately within each task.

    Required columns: dataset_id, episode_id, source_group, task_id.
    Example source_group for new-view renditions: original demonstration identity.
    Example recovery source_group: failure-scene identity. If unknown, STOP:
    silently falling back to episode_id is misleading.
    """
    if not (0 < train_fraction < 1 and 0 < val_fraction < 1
            and train_fraction + val_fraction < 1):
        raise ContractError("fractions must be positive and sum to < 1")
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or ())
        required = {"dataset_id", "episode_id", "source_group", "task_id"}
        if not required.issubset(columns):
            raise ContractError("Missing required columns: " + str(sorted(required - columns)))
        rows = list(reader)
    if not rows:
        raise ContractError("No rows")
    seen = set()
    tasks = defaultdict(lambda: defaultdict(list))
    group_task = {}
    for row in rows:
        for key in required:
            if not row[key].strip():
                raise ContractError("Blank " + key + " in provenance CSV")
        key = (row["dataset_id"], row["episode_id"])
        if key in seen:
            raise ContractError("Repeated dataset_id/episode_id: " + repr(key))
        seen.add(key)
        group = row["source_group"].strip()
        task = row["task_id"].strip()
        if group in group_task and group_task[group] != task:
            raise ContractError("source_group spans tasks: " + group + "; resolve globally before split")
        group_task[group] = task
        tasks[task][group].append(dict(row))
    out = {"train": [], "validation": [], "test": []}
    per_task = {}
    for task in sorted(tasks):
        groups = sorted(tasks[task])
        if len(groups) < 3:
            raise ContractError("Need >=3 independent source groups for task " + task)
        task_seed = int.from_bytes(hashlib.sha256(f"{seed}:{task}".encode()).digest()[:8], "big")
        random.Random(task_seed).shuffle(groups)
        n = len(groups)
        train_n = max(1, int(n * train_fraction))
        val_n = max(1, int(n * val_fraction))
        if train_n + val_n >= n:
            raise ContractError("Not enough source groups to leave a test group")
        mapping = {
            "train": groups[:train_n],
            "validation": groups[train_n:train_n + val_n],
            "test": groups[train_n + val_n:],
        }
        per_task[task] = {name: len(assigned) for name, assigned in mapping.items()}
        for name, assigned in mapping.items():
            for group in assigned:
                out[name].extend(tasks[task][group])
    group_sets = {name: {row["source_group"] for row in records} for name, records in out.items()}
    if any(group_sets[a] & group_sets[b] for a, b in
           (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise AssertionError("Cross-split source group leakage")
    return {
        "input_csv": str(Path(csv_path).resolve()),
        "input_sha256": json_digest(csv_path),
        "seed": seed,
        "split_unit": "source_group (NOT frames or episodes)",
        "per_task_groups": per_task,
        "counts": {name: len(records) for name, records in out.items()},
        "splits": out,
        "limitations": [
            "Groups are only as reliable as independently verified source_group metadata.",
            "The pretrained SmolVLA checkpoint may have seen original LIBERO demonstrations.",
            "No simulator perturbation holdout is established without actual camera/scene mapping.",
        ],
    }


def training_plan(path):
    """Prepare official LeRobot CLI arguments without launching anything."""
    cfg = read_json(path)
    needed = ("lerobot", "model", "data", "run")
    if any(not isinstance(cfg.get(k), dict) for k in needed):
        raise ContractError("Config must have lerobot/model/data/run objects")
    lr, mo, da, run = (cfg[k] for k in needed)
    reasons = []
    for obj, field in ((lr, "source_revision"), (mo, "model_revision"), (da, "dataset_revision")):
        if not isinstance(obj.get(field), str) or not SHA40.fullmatch(obj[field]):
            reasons.append("Pin a 40-character SHA for " + field)
    for obj, field in ((mo, "policy_path"), (da, "dataset_root"), (run, "output_dir")):
        value = obj.get(field)
        if not isinstance(value, str) or not Path(value).is_absolute():
            reasons.append("Use an absolute path for " + field)
    if not isinstance(da.get("episode_indices"), list) or not da["episode_indices"] or not all(
        isinstance(n, int) and n >= 0 for n in da["episode_indices"]
    ):
        reasons.append("Specify nonempty nonnegative episode_indices")
    if not isinstance(da.get("repo_id"), str) or "/" not in da["repo_id"]:
        reasons.append("Specify dataset repo_id")
    for k in ("steps", "batch_size", "save_freq"):
        if not isinstance(run.get(k), int) or run[k] <= 0:
            reasons.append(k + " must be positive")
    if run.get("peft_method") != "LORA":
        reasons.append("Only pinned default LORA is implemented; vision-LoRA is NOT implemented")
    rank, alpha = run.get("lora_rank"), run.get("lora_alpha")
    if not isinstance(rank, int) or rank < 1 or not isinstance(alpha, int) or alpha < 1:
        reasons.append("Require positive integer LoRA rank and alpha")
    # A split manifest is mandatory, not merely optional documentation.
    # This ensures the exact CLI train episodes equal the reviewed source groups.
    manifest = da.get("train_split_manifest")
    if not isinstance(manifest, str) or not Path(manifest).is_absolute():
        reasons.append("Provide an absolute path to an audited train_split_manifest")
    elif not Path(manifest).is_file():
        reasons.append("Audited train_split_manifest does not exist: " + manifest)
    else:
        try:
            split = read_json(manifest)
            buckets = split["splits"]
            expected_buckets = ("train", "validation", "test")
            if any(not isinstance(buckets[k], list) or not buckets[k] for k in expected_buckets):
                raise ContractError("train/validation/test buckets must be nonempty lists")
            groups = {}
            train_ids = []
            seen_episodes = set()
            for section in expected_buckets:
                for row in buckets[section]:
                    group = row["source_group"].strip()
                    dataset_id = row["dataset_id"].strip()
                    episode_id = int(row["episode_id"])
                    if not group or dataset_id != da.get("repo_id") or episode_id < 0:
                        raise ContractError("Missing provenance group, repo mismatch or negative episode")
                    if (dataset_id, episode_id) in seen_episodes:
                        raise ContractError("Duplicate episode across provenance manifest")
                    seen_episodes.add((dataset_id, episode_id))
                    if group in groups and groups[group] != section:
                        raise ContractError("Source group crosses train/validation/test")
                    groups[group] = section
                    if section == "train":
                        train_ids.append(episode_id)
            if not isinstance(da.get("episode_indices"), list) or set(train_ids) != set(da["episode_indices"]) or len(train_ids) != len(da["episode_indices"]):
                raise ContractError("CLI training Episode IDs do not equal audited train split")
        except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
            reasons.append("Invalid audited train_split_manifest: " + str(exc))
    if reasons:
        return {"status": "BLOCKED", "reasons": reasons,
                "config_sha256": json_digest(path), "gpu_used": False, "command": None}
    executable = str(Path(sys.executable).with_name("lerobot-train"))
    command = [
        executable,
        f"--policy.path={mo['policy_path']}",
        f"--policy.pretrained_revision={mo['model_revision']}",
        "--policy.input_features=null", "--policy.output_features=null",
        "--policy.device=cuda", "--policy.use_amp=false",
        "--policy.push_to_hub=false",
        f"--dataset.repo_id={da['repo_id']}",
        f"--dataset.root={da['dataset_root']}",
        f"--dataset.revision={da['dataset_revision']}",
        "--dataset.episodes=" + json.dumps(da["episode_indices"], separators=(",", ":")),
        "--dataset.video_backend=pyav",
        "--peft.method_type=LORA",
        f"--peft.r={rank}", f"--peft.lora_alpha={alpha}",
        f"--batch_size={run['batch_size']}",
        "--num_workers=0",
        f"--steps={run['steps']}",
        f"--save_freq={run['save_freq']}",
        "--log_freq=1", "--env_eval_freq=0", "--wandb.enable=false",
        f"--output_dir={run['output_dir']}",
    ]
    return {
        "status": "READY_FOR_MANUAL_ENVIRONMENT_CHECK_NOT_GPU_APPROVED",
        "command": command,
        "model_repo_id": mo.get("repo_id"),
        "lerobot_source_revision": lr["source_revision"],
        "config_sha256": json_digest(path),
        "gpu_used": False,
        "warnings": [
            "No checkpoint, processor, dataset episode, or frozen model SHA independently verified.",
            "The old robot-lab LeRobot training path once needed a scoped pretrained_revision fix.",
            "Only use this command in a separate pinned LeRobot venv with its exact CLI.",
            "Check action normalization, camera mapping, FPS, initial states and data split before execution.",
        ],
    }


def guarded_launch(path, acknowledged=False):
    """Manual-only launch; refuses if any GPU compute process is visible.

    This is a safety gate, NOT a scheduler. GPU-Util=0 at one instant is not
    sufficient permission; an explicit flag plus empty compute-process list
    and validated pinned paths are all required.
    """
    if not acknowledged:
        raise ContractError("Explicit --i-confirm-corrector-finished required")
    plan = training_plan(path)
    if plan["status"] == "BLOCKED":
        raise ContractError("Training config blocked: " + "; ".join(plan["reasons"]))
    cfg = read_json(path)
    mo, da, run = cfg["model"], cfg["data"], cfg["run"]
    for value in (mo["policy_path"], da["dataset_root"]):
        if not Path(value).is_dir():
            raise ContractError("Missing local snapshot/data dir: " + value)
    output = Path(run["output_dir"])
    if output.exists():
        raise ContractError("Refuse to overwrite existing output: " + str(output))
    usage = shutil.disk_usage(output.parent if output.parent.exists() else da["dataset_root"])
    if usage.free < 30 * GIB:
        raise ContractError("Require >=30GiB free space before launching")
    if not Path(plan["command"][0]).is_file():
        raise ContractError("Dedicated lerobot-train executable not found in current venv")
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContractError("GPU process probe unavailable; refuse launch") from exc
    if probe.stdout.strip():
        raise ContractError("GPU has active compute process(es); refuse launch")
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    output.parent.mkdir(parents=True, exist_ok=True)
    # No model training happens until all above checks pass. Caller requested execution.
    return subprocess.run(plan["command"], env=env, check=True).returncode
