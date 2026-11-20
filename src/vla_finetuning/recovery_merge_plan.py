"""Strict, non-executing plan to materialize one aligned LeRobot v3 training set.

LeRobot pinned at 7e241bd... refuses dataset.repo_id lists. No automatic
conversion until raw Expert files and real action/camera semantics are audited.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from .core import ContractError, read_json
from .recovery_experiments import EXPERT, NORMAL, validate_split


def _info(path):
    path = Path(path)
    obj = read_json(path)
    features = obj.get("features")
    if obj.get("codebase_version") != "v3.0" or not isinstance(features, dict):
        raise ContractError("Only observed, already-v3 LeRobot metadata may enter this materialization plan")
    return obj, hashlib.sha256(path.read_bytes()).hexdigest()


def plan_mixed_v3_dataset(split_path, alignment_path):
    """Validate real metadata and manual alignment; output mapping, not files."""
    split = validate_split(split_path)
    obj = read_json(alignment_path)
    if not isinstance(obj, dict) or obj.get("kind") != "recovery_alignment_v1" or obj.get("verified") is not True:
        raise ContractError("Requires human-verified recovery_alignment_v1")
    paths, cameras = obj.get("metadata_paths"), obj.get("camera_features")
    evidence = obj.get("evidence")
    if not all(isinstance(x, dict) for x in (paths, cameras, evidence)):
        raise ContractError("Missing metadata_paths/camera_features/evidence maps")
    for field in ("gripper", "action_coordinates", "state_order", "camera_transforms", "timebase", "normalization", "task_labels"):
        value = evidence.get(field)
        if not isinstance(value, str) or not value.strip() or "TBD" in value.upper() or "REPLACE" in value.upper():
            raise ContractError("Unverified semantic evidence: " + field)
    feature_hashes = {}
    fps = {}
    for source in (EXPERT, NORMAL):
        if source not in paths or source not in cameras or not isinstance(cameras[source], dict):
            raise ContractError("No local metadata or camera alignment for " + source)
        meta, digest = _info(paths[source])
        specs = meta["features"]
        action, state = specs.get("action"), specs.get("observation.state")
        if not isinstance(action, dict) or not isinstance(state, dict) or action.get("shape") != [7] or state.get("shape") != [8]:
            raise ContractError("Unverified 7D/8D feature compatibility: " + source)
        mapping = cameras[source]
        if set(mapping) != {"front", "wrist"} or len(set(mapping.values())) != 2:
            raise ContractError("Exactly two distinct mapped cameras required")
        if any(not isinstance(specs.get(name), dict) or specs[name].get("dtype") not in ("video", "image")
               for name in mapping.values()):
            raise ContractError("Mapped camera is absent from observed metadata")
        f = meta.get("fps")
        if type(f) not in (int, float) or not 0 < f <= 200:
            raise ContractError("Unknown FPS")
        fps[source] = f
        feature_hashes[source] = digest
    if fps[EXPERT] != fps[NORMAL]:
        raise ContractError("FPS mismatch; explicit timestamp/resampling implementation required before materializing")
    target = obj.get("target_dataset_id")
    if not isinstance(target, str) or not target.strip() or "REPLACE" in target.upper():
        raise ContractError("Need unique output dataset identifier")
    rows = read_json(split_path)["splits"]["train"]
    rows = sorted(rows, key=lambda r: (r["dataset_id"], r["episode_id"]))
    mapping = [{"target_episode_index": i, "source_dataset": r["dataset_id"],
                "source_episode_index": r["episode_id"], "source_group": r["source_group"]}
               for i, r in enumerate(rows)]
    return {"status": "PLAN_ONLY_NO_DATASET_MATERIALIZED", "target_dataset_id": target,
            "source_info_sha256": feature_hashes, "source_split_sha256": split["split_sha256"],
            "source_fps": fps, "episode_mapping": mapping,
            "camera_features": cameras, "alignment_evidence": evidence,
            "blockers": ["Implement and test actual LeRobot v3 dataset serialization with source metadata preserved",
                         "Verify paired image/action/timestamp decoding and processor normalization on real batches",
                         "Audit all converted source groups, filenames, hashes and no train/test leakage",
                         "Run GPU smoke/optimizer/checkpoint/reload only after Corrector GPU is idle"]}
