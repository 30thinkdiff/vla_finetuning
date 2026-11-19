"""Read-only LeRobot v2.1 Expert file resolver based on *observed* local metadata.

LIBERO-RECOVER arXiv:2609.05178v4 Appendix B.5/C.2 specifies LeRobot v2.1,
20Hz, 2 cameras, episode JSONL, per-episode Parquet/MP4. This module never
assumes camera-key names, episode-chunk size, or specific metadata row values.
"""
from __future__ import annotations

import string
from pathlib import Path, PurePosixPath

from .core import ContractError, read_json
from .recovery_contract import read_jsonl, _sha_file


_ALLOWED_FIELDS = frozenset({"episode_chunk", "episode_index", "video_key"})
def _resolve_template(template: str, *, episode_chunk: int, episode_index: int,
                      video_key: str | None = None) -> str:
    if not isinstance(template, str) or not template or "\\" in template:
        raise ContractError("Missing or unsafe LeRobot v2.1 path template")
    kwargs = {"episode_chunk": episode_chunk, "episode_index": episode_index, "video_key": video_key}
    for text, field, spec, conversion in string.Formatter().parse(template):
        if field is None:
            continue
        if field not in _ALLOWED_FIELDS or kwargs[field] is None or conversion is not None:
            raise ContractError("Unsupported dataset template field or conversion")
        if "{" in spec or "}" in spec or len(spec) > 12:
            raise ContractError("Unsupported template formatting")
        if field in ("episode_chunk", "episode_index") and (spec and not (spec == "d" or
            (spec.endswith("d") and spec[:-1].isdigit()))):
            raise ContractError("Only plain integer formatting accepted")
        if field == "video_key" and spec:
            raise ContractError("No format spec for video_key")
    try:
        rendered = template.format(**kwargs)
    except (IndexError, KeyError, ValueError, TypeError, OverflowError) as exc:
        raise ContractError("Cannot render v2.1 path template") from exc
    path = PurePosixPath(rendered)
    if path.is_absolute() or ".." in path.parts or "." in path.parts or not path.parts or
            rendered.startswith("~") or "//" in rendered or ":" in rendered:
        raise ContractError("Unsafe rendered v2.1 relative path")
    return path.as_posix()


def plan_v21_expert_files(root, episode_indices):
    """Plan exact per-episode payload paths; never fetch or decode bytes."""
    root = Path(root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    if not info_path.is_file() or not episodes_path.is_file():
        raise ContractError("Need real local v2.1 meta/info.json AND meta/episodes.jsonl")
    obj = read_json(info_path)
    if not isinstance(obj, dict) or obj.get("codebase_version") != "v2.1":
        raise ContractError("LIBERO-Recover paper specifies v2.1; observed file disagrees")
    fps = obj.get("fps")
    if type(fps) not in (float, int) or fps <= 0 or fps > 200:
        raise ContractError("Invalid dataset FPS")
    chunk_size = obj.get("chunks_size")
    if type(chunk_size) is not int or chunk_size < 1 or chunk_size > 100000:
        raise ContractError("Need observed chunks_size for v2.1 paths; do not assume 1000")
    features = obj.get("features")
    if not isinstance(features, dict):
        raise ContractError("Missing observed feature definitions")
    cameras = sorted(k for k, v in features.items() if isinstance(v, dict) and v.get("dtype") == "video")
    if len(cameras) != 2:
        raise ContractError("Expected two observed v2.1 video cameras, got " + repr(cameras))
    if not isinstance(episode_indices, list) or not episode_indices or len(set(episode_indices)) != len(episode_indices) or any(
            type(n) is not int or n < 0 for n in episode_indices):
        raise ContractError("Require distinct nonnegative episode IDs")
    if len(episode_indices) > 100:
        raise ContractError("Plan at most 100 episodes per batch")
    index = {}
    for row in read_jsonl(episodes_path, max_records=100000):
        ep = row.get("episode_index")
        if type(ep) is not int or ep < 0 or ep in index:
            raise ContractError("Missing/duplicate episode_index in real v2.1 index")
        index[ep] = row
    if not set(episode_indices).issubset(index):
        raise ContractError("Missing selected Episode IDs from v2.1 metadata")
    mapping, files = {}, set()
    for ep in sorted(episode_indices):
        chunk = ep // chunk_size
        data_file = _resolve_template(obj.get("data_path"), episode_index=ep, episode_chunk=chunk)
        video_files = {key: _resolve_template(obj.get("video_path"), episode_index=ep,
                                              episode_chunk=chunk, video_key=key)
                       for key in cameras}
        if not data_file.startswith("data/") or not data_file.endswith(".parquet"):
            raise ContractError("Unexpected v2.1 Parquet path")
        if any(not p.startswith("videos/") or not p.endswith(".mp4") for p in video_files.values()):
            raise ContractError("Unexpected v2.1 video path")
        files.add(data_file)
        files.update(video_files.values())
        mapping[str(ep)] = {"data_file": data_file, "video_files": video_files,
                            "length": index[ep].get("length"), "tasks": index[ep].get("tasks")}
    return {
        "status": "V21_FILE_PLAN_ONLY_NOT_DOWNLOADED",
        "source_info_sha256": _sha_file(info_path),
        "source_episodes_sha256": _sha_file(episodes_path),
        "observed_dataset_version": "v2.1", "observed_fps": fps,
        "cameras": cameras, "episodes": sorted(episode_indices),
        "episode_files": mapping, "payload_files": sorted(files),
        "payload_file_count": len(files),
        "semantic_checks": {
            "action_coordinates_verified": False,
            "gripper_values_verified": False,
            "camera_orientation_verified": False,
            "source_failure_scene_mapping_verified": False,
            "normalization_verified": False,
        },
        "note": "These are paths from observed v2.1 templates, not proof of existing bytes or usable SmolVLA training data",
    }
