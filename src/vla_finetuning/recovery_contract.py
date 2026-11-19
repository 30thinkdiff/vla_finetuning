"""Offline contracts for LIBERO-Recovery; never download, simulate or train.

Important: official scene annotations and expert demonstrations are different
artifacts. Their relationship is not inferred by matching task text or numbers.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from .core import ContractError, read_json

SUITES = frozenset({"libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"})
LEVELS = frozenset({"L1", "L2", "L3", "L4"})


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path, *, max_records=100000):
    path = Path(path)
    if not path.is_file():
        raise ContractError("Missing local JSONL file: " + str(path))
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            if len(rows) >= max_records:
                raise ContractError("JSONL record limit exceeded")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"Invalid JSONL at line {number}") from exc
            if not isinstance(row, dict):
                raise ContractError(f"JSONL line {number} is not an object")
            rows.append(row)
    if not rows:
        raise ContractError("No JSONL records")
    return rows


def audit_expert_info(path):
    """Report observed metadata, never invent absent action/control semantics."""
    path = Path(path)
    obj = read_json(path)
    if not isinstance(obj, dict) or not isinstance(obj.get("features"), dict):
        raise ContractError("Expected LeRobot meta/info.json features object")
    features = obj["features"]
    for required in ("action", "observation.state"):
        if required not in features or not isinstance(features[required], dict):
            raise ContractError("Missing feature: " + required)
    images = sorted(k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") in ("video", "image"))
    specs = {k: {"dtype": v.get("dtype"), "shape": v.get("shape"), "names": v.get("names")}
             for k, v in features.items() if isinstance(v, dict)}
    warnings = []
    if len(images) != 2:
        warnings.append("Expected two camera observations; actual ordering/semantics must be verified")
    if features["action"].get("shape") != [7]:
        warnings.append("Action dimensionality is not confirmed [7]")
    if features["observation.state"].get("shape") != [8]:
        warnings.append("State dimensionality is not confirmed [8]")
    fps = obj.get("fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
        warnings.append("Invalid or absent FPS")
    return {"status": "METADATA_ONLY_NOT_SEMANTICALLY_VERIFIED", "path": str(path.resolve()),
            "sha256": _sha_file(path), "codebase_version": obj.get("codebase_version"),
            "total_episodes": obj.get("total_episodes"), "fps": fps,
            "image_keys": images, "features": specs, "warnings": warnings,
            "source_scene_link_verified": False, "action_control_semantics_verified": False,
            "processor_normalization_verified": False, "safe_to_train": False}


def audit_scene_index(path):
    """Read *official* released failure-scene index; no Expert linking claim."""
    rows = read_jsonl(path, max_records=20000)
    levels, suites, stages = Counter(), Counter(), Counter()
    identifiers = set()
    for row in rows:
        scene_id, level, suite = row.get("scene_id"), row.get("level"), row.get("source")
        if not isinstance(scene_id, str) or not scene_id.strip() or scene_id in identifiers:
            raise ContractError("Missing/duplicate scene_id in official scene index")
        if level not in LEVELS or suite not in SUITES:
            raise ContractError("Unexpected scene level or LIBERO suite: " + scene_id)
        identifiers.add(scene_id)
        levels[level] += 1
        suites[suite] += 1
        stages[str(row.get("stage", "unknown"))] += 1
    return {"status": "SCENE_ANNOTATION_INDEX_ONLY", "source": str(Path(path).resolve()),
            "sha256": _sha_file(Path(path)), "scenes": len(rows), "levels": dict(sorted(levels.items())),
            "suites": dict(sorted(suites.items())), "stages": dict(sorted(stages.items())),
            "expert_demonstration_link_verified": False,
            "note": "scene_id/npz_path do not prove the Expert demo originated from this scene"}


def audit_manual_linkage(path):
    """Only human-reviewed provenance; a guessed same-task join is forbidden."""
    rows = read_jsonl(path)
    seen, groups = set(), set()
    for row in rows:
        required = ("dataset_id", "episode_id", "source_group", "source_evidence", "review_status")
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in required):
            raise ContractError("Missing linkage field: " + repr(row))
        key = (row["dataset_id"], row["episode_id"])
        if key in seen:
            raise ContractError("Duplicate expert episode linkage: " + repr(key))
        if row["review_status"] != "verified":
            raise ContractError("Unverified linkage: " + repr(key))
        if "SYNTHETIC" in json.dumps(row).upper() or row["source_group"].lower().startswith("unknown"):
            raise ContractError("Synthetic/unknown linkage cannot be verified")
        seen.add(key)
        groups.add(row["source_group"])
    return {"status": "LINKAGE_MANIFEST_SCHEMA_VALID_ONLY", "rows": len(rows),
            "source_groups": len(groups), "sha256": _sha_file(Path(path)),
            "warning": "Evidence strings were not externally verified; manually audit each original scene"}


def dataset_gripper_to_env(value, *, contract_verified=False):
    """Reference transformation ONLY for an explicitly checked binary convention.

    Official Recovery evaluator uses env +1=CLOSE / -1=OPEN; logged demo
    convention may be 0=CLOSE / 1=OPEN. Never silently apply to arbitrary data.
    """
    if not contract_verified:
        raise ContractError("Gripper conversion requires independently verified action contract")
    if type(value) not in (int, float) or not math.isfinite(value) or float(value) not in (0.0, 1.0):
        raise ContractError("Expected binary dataset gripper: 0=close, 1=open")
    return 1.0 if float(value) == 0.0 else -1.0


def check_seven_dimensional_action(action):
    if not isinstance(action, (list, tuple)) or len(action) != 7:
        raise ContractError("Expected exactly seven action components")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in action):
        raise ContractError("Non-finite or nonnumeric action component")
    return [float(v) for v in action]
