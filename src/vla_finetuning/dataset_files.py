"""Plan exact LeRobot-v3 shard downloads from real episode metadata.

Needs PyArrow only when the plan is generated. Does not download anything.
"""
from __future__ import annotations
from pathlib import Path, PurePosixPath
from .core import ContractError, read_json
from .guards import require_directory


def _get(record, name):
    if name in record:
        return record[name]
    value = record
    for part in name.split("/"):
        if not isinstance(value, dict) or part not in value:
            raise ContractError(f"Missing Episode source column: {name}")
        value = value[part]
    return value


def _safe_rel_path(value):
    path = PurePosixPath(str(value))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ContractError("Unsafe dataset relative path")
    return str(path)


def plan_v3_episode_shards(meta_root, episode_indices):
    """Resolve data and per-camera video file paths from official v3 metadata."""
    root = require_directory(meta_root, name="meta_root")
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ContractError("Need local pinned LeRobot meta/info.json")
    info = read_json(info_path)
    if info.get("codebase_version") != "v3.0":
        raise ContractError("v3 resolver supports codebase_version v3.0 only")
    if (not isinstance(episode_indices, list) or not episode_indices
        or any(type(n) is not int or n < 0 for n in episode_indices)
        or len(set(episode_indices)) != len(episode_indices)):
        raise ContractError("episode_indices must be unique nonnegative integers")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ContractError("PyArrow required in separate data environment") from exc
    shards = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    if not shards:
        raise ContractError("Need pinned meta/episodes/*.parquet files first")
    data_template = info.get("data_path")
    video_template = info.get("video_path")
    vid_keys = sorted(k for k, v in info.get("features", {}).items()
                      if isinstance(v, dict) and v.get("dtype") == "video")
    if not isinstance(data_template, str) or (vid_keys and not isinstance(video_template, str)):
        raise ContractError("Missing official data/video file templates")
    selected = set(episode_indices)
    seen = {}
    for file in shards:
        table = pq.read_table(file)
        if "episode_index" not in table.schema.names:
            raise ContractError("Episode shard has no episode_index")
        for row in table.to_pylist():
            ep = int(row["episode_index"])
            if ep not in selected:
                continue
            if ep in seen:
                raise ContractError(f"Duplicate episode index {ep}")
            seen[ep] = row
    missing = selected - set(seen)
    if missing:
        raise ContractError("Episode IDs not found: " + str(sorted(missing)))
    result = set()
    mapping = {}
    for ep in sorted(selected):
        row = seen[ep]
        def field(name):
            value = _get(row, name)
            if type(value) is not int or value < 0:
                raise ContractError("Invalid episode shard index: " + name)
            return value
        data = _safe_rel_path(data_template.format(
            chunk_index=field("data/chunk_index"),
            file_index=field("data/file_index")))
        result.add(data)
        videos = {}
        for key in vid_keys:
            rel = _safe_rel_path(video_template.format(
                video_key=key,
                chunk_index=field(f"videos/{key}/chunk_index"),
                file_index=field(f"videos/{key}/file_index")))
            result.add(rel)
            videos[key] = rel
        mapping[str(ep)] = {"data_file": data, "video_files": videos}
    return {
        "format": "lerobot_v3_shard_selection",
        "episodes": sorted(selected), "episode_sources": mapping,
        "payload_files": sorted(result), "payload_files_count": len(result),
        "source_metadata_root": str(root), "metadata_verified": True,
        "camera_perturbation_provenance_verified": False,
        "note": "Multiple Episodes can share one file; payload count is not Episode count."
    }


def train_episodes_from_split(split_path, repo_id):
    """Use only manually audited train split; exclude validation/test."""
    obj = read_json(split_path)
    splits = obj.get("splits", {})
    if any(k not in splits for k in ("train", "validation", "test")):
        raise ContractError("Missing train/validation/test split")
    seen = set()
    episodes = []
    groups = {}
    for name in ("train", "validation", "test"):
        for row in splits[name]:
            if row.get("dataset_id") != repo_id:
                raise ContractError("Dataset repo ID mismatch")
            if row.get("review_status", "").strip().lower() != "verified" or not row.get("source_evidence"):
                raise ContractError("Missing manually audited provenance")
            if row["source_evidence"].upper().startswith("SYNTHETIC"):
                raise ContractError("Cannot use synthetic provenance")
            ep = int(row["episode_id"])
            group = row["source_group"]
            if ep in seen or (group in groups and groups[group] != name):
                raise ContractError("Episode or source group leakage")
            seen.add(ep)
            groups[group] = name
            if name == "train":
                episodes.append(ep)
    if not episodes:
        raise ContractError("No training Episodes in reviewed split")
    return sorted(episodes)
