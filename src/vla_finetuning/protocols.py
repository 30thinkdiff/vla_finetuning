"""Read-only gates for future model loading and LIBERO paired evaluation.

No model deserialization, GPU allocation, simulation, or data downloads.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from .core import ContractError, json_digest, read_json

SHA = re.compile(r"^[0-9a-fA-F]{40}$")


def inspect_model_snapshot(root):
    """Audit manifest-level evidence; DOES NOT verify actual checkpoint correctness.

    Read at most top-level small config files; only stat the weight files.
    Hugging Face snapshots may contain symlinks; no copy or dereference of weights.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ContractError("Model snapshot directory not found: " + str(root))
    configs = {}
    for name in ("config.json", "policy_config.json", "preprocessor_config.json",
                 "processor_config.json", "adapter_config.json"):
        path = root / name
        if not path.is_file():
            continue
        if path.stat().st_size > 1024 * 1024:
            raise ContractError("Unexpectedly large model configuration file: " + name)
        try:
            value = read_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ContractError("Invalid model config: " + name) from exc
        configs[name] = {"sha256": json_digest(path), "keys": sorted(value) if isinstance(value, dict) else []}
    weights = {}
    # Only inspect root file names; do not recurse into a generic user directory.
    for pattern in ("*.safetensors", "*.bin", "*.safetensors.index.json"):
        for path in root.glob(pattern):
            if path.is_file():
                weights[path.name] = path.stat().st_size
    return {"model_dir": str(root), "configs": configs, "weight_files_sizes": weights,
            "weight_contents_read": False, "gpu_used": False, "snapshot_pinned_revision_verified": False,
            "processor_compatibility_verified": False, "ready_to_train": False,
            "next": "Pin HF SHA and inspect policy/processors with the dedicated LeRobot training environment."}


def validate_eval_plan(path):
    """Check same-case B0/L1 evaluation protocol, not run episodes."""
    obj = read_json(path)
    if not isinstance(obj, dict):
        raise ContractError("Eval plan must be an object")
    revisions = obj.get("revisions")
    if not isinstance(revisions, dict):
        raise ContractError("Must pin simulation and baseline policy revisions")
    for key in ("simulator", "baseline_model"):
        if not isinstance(revisions.get(key), str) or not SHA.fullmatch(revisions[key]):
            raise ContractError("Missing 40-character revision: " + key)
    methods = obj.get("methods")
    if not isinstance(methods, list) or len(methods) < 2 or any(
        not isinstance(m, str) or not m.strip() for m in methods
    ) or len(set(methods)) != len(methods):
        raise ContractError("At least two distinct method names required")
    cases = obj.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ContractError("At least one fixed evaluation case required")
    seen = set()
    for case in cases:
        if not isinstance(case, dict):
            raise ContractError("Each case must be an object")
        for key in ("domain", "condition_id", "instruction", "initial_state_sha256"):
            if not isinstance(case.get(key), str) or not case[key].strip():
                raise ContractError("Missing case field: " + key)
        if not isinstance(case.get("seed"), int) or isinstance(case["seed"], bool):
            raise ContractError("Evaluation seed must be int")
        if not SHA.fullmatch(case["initial_state_sha256"]) and not re.fullmatch(
            r"[0-9a-fA-F]{64}", case["initial_state_sha256"]
        ):
            raise ContractError("initial_state_sha256 must be 40/64 hexadecimal chars")
        identity = (case["domain"], case["condition_id"], case["seed"])
        if identity in seen:
            raise ContractError("Duplicate evaluation case: " + repr(identity))
        seen.add(identity)
    control = obj.get("control")
    if not isinstance(control, dict):
        raise ContractError("Control protocol must be pinned")
    for key in ("control_fps", "action_horizon", "max_steps"):
        if not isinstance(control.get(key), int) or isinstance(control[key], bool) or control[key] < 1:
            raise ContractError("Invalid control protocol field: " + key)
    return {"status": "PLAN_ONLY_NOT_EXECUTED", "unique_cases": len(cases),
            "methods": methods, "expected_rollouts": len(cases) * len(methods),
            "simulation_loaded": False, "model_loaded": False,
            "warning": "This does not verify camera, language, physical state, policy noise or actuator compatibility."}
