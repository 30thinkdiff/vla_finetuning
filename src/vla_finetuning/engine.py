"""SmolVLA/LeRobot integration boundaries for *real* samples, training and eval.

The trusted training engine is LeRobot pinned source, not a replacement training loop.
These functions build exact argv; GPU commands are only run with user --apply and
idle-GPU checks. A real LeRobot smoke is still required before formal runs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from .core import ContractError, read_json, GIB
from .guards import (load_protocol, require_directory, require_unused_output,
                     require_disk, require_explicit, require_gpu_idle,
                     ensure_cache_on_data_disk, shell_render)
from .dataset_files import train_episodes_from_split


def verify_checkout(path, sha):
    path = require_directory(path, name="LeRobot checkout")
    if not (path / "src" / "lerobot").is_dir():
        raise ContractError("LeRobot source checkout missing src/lerobot")
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ContractError("Cannot verify LeRobot source commit") from exc
    if actual != sha:
        raise ContractError(f"LeRobot checkout commit mismatch: expected {sha}; got {actual}")
    return actual


def inspection_argv(config, *, python=None):
    """Construct child command for a REAL sample. No import/model load now."""
    data = config["dataset"]
    python = python or sys.executable
    return [
        python, "-m", "vla_finetuning.real_batch",
        "--repo", data["repo_id"],
        "--revision", data["revision"],
        "--root", data["local_dir"],
        "--episode", str(data["train_episodes"][0]),
    ]


def _train_argv(config, *, binary="lerobot-train"):
    ds, model, t = config["dataset"], config["model"], config["train"]
    prefix = config["paths"]
    if not isinstance(prefix, dict):
        raise ContractError("Missing paths")
    if not Path(ds["local_dir"]).is_absolute() or not Path(model["local_dir"]).is_absolute():
        raise ContractError("Model and dataset paths must be absolute")
    if t.get("method") != "expert_lora":
        raise ContractError("Only official default Expert LoRA is integrated; Vision LoRA is a future ablation")
    if type(t.get("rank")) is not int or type(t.get("lora_alpha")) is not int or min(
        t["rank"], t["lora_alpha"]
    ) < 1:
        raise ContractError("Invalid LoRA rank/alpha")
    if t.get("precision") not in ("fp32",):
        raise ContractError("Only validated prior fp32 configuration permitted until GPU benchmark")
    if type(t.get("seed")) is not int:
        raise ContractError("Training seed must be integer")
    if type(t.get("lr")) not in (float, int) or not (0 < t["lr"] < 0.1):
        raise ContractError("Learning rate out of conservative range")
    return [
        binary,
        f"--policy.path={model['local_dir']}",
        "--policy.input_features=null",
        "--policy.output_features=null",
        "--policy.device=cuda",
        "--policy.use_amp=false",
        "--policy.push_to_hub=false",
        f"--policy.optimizer_lr={t['lr']}",
        f"--dataset.repo_id={ds['repo_id']}",
        f"--dataset.root={ds['local_dir']}",
        f"--dataset.revision={ds['revision']}",
        "--dataset.episodes=" + json.dumps(ds["train_episodes"], separators=(",", ":")),
        "--dataset.video_backend=pyav",
        "--peft.method_type=LORA",
        f"--peft.r={t['rank']}",
        f"--peft.lora_alpha={t['lora_alpha']}",
        f"--batch_size={t['batch_size']}",
        "--num_workers=0",
        f"--steps={t['steps']}",
        f"--save_freq={t['save_freq']}",
        f"--seed={t['seed']}",
        "--log_freq=1",
        "--env_eval_freq=0",
        "--wandb.enable=false",
        f"--output_dir={t['output_dir']}",
    ]


def train_plan(config_path, *, bin_path="lerobot-train"):
    cfg = load_protocol(config_path)
    split = cfg["dataset"]["split_manifest"]
    train_ids = train_episodes_from_split(split, cfg["dataset"]["repo_id"])
    if set(train_ids) != set(cfg["dataset"]["train_episodes"]):
        raise ContractError("Train Episode IDs must match audited train-only source group split")
    command = _train_argv(cfg, binary=bin_path)
    return {"status": "PLAN_ONLY_REQUIRES_REAL_SMOKE", "argv": command,
            "display": shell_render(command),
            "experiment": cfg["train"]["run_id"],
            "source_config": str(Path(config_path).resolve()),
            "checks": ["LeRobot commit", "data/materialization", "sample shapes and semantics",
                       "LoRA trainable-module names", "checkpoint/reload", "CUDA budget"],
            "executed": False}


def eval_argv(config, checkpoint, output_dir, *, bin_path="lerobot-eval"):
    ev = config["evaluation"]
    checkpoint = Path(checkpoint).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not checkpoint.is_dir():
        raise ContractError("Policy checkpoint path missing")
    return [
        bin_path, f"--policy.path={checkpoint}",
        "--env.type=libero_plus", f"--env.task={ev['suite']}",
        "--env.task_ids=" + json.dumps(ev["variant_task_ids"], separators=(",", ":")),
        f"--env.control_mode={ev['control_mode']}",
        "--env.hard_reset=true", "--env.init_states=true",
        "--eval.batch_size=1", f"--eval.n_episodes={ev['n_episodes']}",
        f"--seed={ev['seed']}",
        f"--policy.n_action_steps={ev['action_horizon']}",
        f"--output_dir={output_dir}",
    ]


def eval_plan(config_path, checkpoint, output_dir, *, bin_path="lerobot-eval"):
    cfg = load_protocol(config_path)
    command = eval_argv(cfg, checkpoint, output_dir, bin_path=bin_path)
    return {"status": "PLAN_ONLY_REQUIRES_SIM_COMPATIBILITY_SMOKE",
            "argv": command, "display": shell_render(command),
            "selected_variants": cfg["evaluation"]["variant_task_ids"],
            "n_episodes_per_variant": cfg["evaluation"]["n_episodes"],
            "expected_rollouts": len(cfg["evaluation"]["variant_task_ids"]) *
                                 cfg["evaluation"]["n_episodes"],
            "executed": False,
            "note": "Environment variant task_ids differ from Dataset task_index; "
                    "hard reset/relative actions must be checked in actual simulator."}


def execute_command(plan, cfg, *, apply=False, confirmed_idle=False, runner=None):
    """Opt-in only. Calls official LeRobot executable; no shell."""
    require_explicit(apply and confirmed_idle, "GPU training/evaluation")
    require_gpu_idle()
    if not isinstance(plan, dict) or not isinstance(plan.get("argv"), list):
        raise ContractError("No verified argv list")
    if plan.get("status") not in ("PLAN_ONLY_REQUIRES_REAL_SMOKE",
                                  "PLAN_ONLY_REQUIRES_SIM_COMPATIBILITY_SMOKE"):
        raise ContractError("Cannot execute unsupported plan")
    project_root = require_directory(cfg["paths"]["project_root"], name="project root")
    dest = Path(plan["argv"][-1].split("=", 1)[1])
    require_unused_output(dest)
    require_disk(dest, free_gib=max(30, cfg["paths"]["min_free_gib"]))
    for value in (cfg["model"]["local_dir"], cfg["dataset"]["local_dir"]):
        require_directory(value, name="pre-downloaded local input")
    executable = plan["argv"][0]
    if not Path(executable).is_file():
        raise ContractError("Use an absolute verified executable path inside isolated LeRobot venv")
    if runner is None:
        runner = subprocess.run
    env = ensure_cache_on_data_disk(project_root)
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_DATASETS_OFFLINE"] = "1"
    env["MUJOCO_GL"] = "egl"
    # Running a GPU command requires a distinct, explicit operation by the user.
    return runner(plan["argv"], check=True, env=env).returncode


def checkpoint_inventory(checkpoint):
    """Find actual files but do not load tensors; note incomplete semantics explicitly."""
    root = require_directory(checkpoint, name="checkpoint")
    files = sorted(p for p in root.rglob("*") if p.is_file() and not p.is_symlink())
    if not files:
        raise ContractError("No checkpoint files")
    paths = [str(p.relative_to(root)) for p in files]
    has_model = any("pretrained_model" in p and p.endswith(".safetensors") for p in paths)
    has_train_state = any("training_state" in p for p in paths)
    return {"checkpoint_path": str(root), "file_count": len(files),
            "files": paths, "has_model_weights": has_model,
            "has_training_state": has_train_state,
            "can_infer": has_model,
            "can_resume": has_model and has_train_state,
            "checked_hashes": False, "reloaded_in_real_model": False,
            "note": "File presence alone does not prove model can be loaded or resumed."}
