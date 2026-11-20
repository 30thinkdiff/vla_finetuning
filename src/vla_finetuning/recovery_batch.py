"""Build one real, read-only, boundary-safe SmolVLA sample from Expert v2.1.

This is a CPU prototype, not a replacement for the pinned LeRobot v3 DataLoader.
Action windows are [t, t+49] at 20Hz with end-of-episode CLAMP padding
matching pinned LeRobot DatasetReader._get_query_indices. The matching mask
is explicit. No action normalization or model inference occurs.

Generated .npz files contain numeric arrays (no pickle) and a Unicode task
string; source dataset is never modified. Two camera keys are staged for the
PRETRAINED processor's one-time rename:
  image -> camera1; wrist_image -> image2 -> camera2.
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path, PurePosixPath

from .recovery_v21 import plan_v21_expert_files


class RecoveryBatchError(ValueError):
    pass


def require_finite_vector(value, width, label):
    if not isinstance(value, (list, tuple)) or len(value) != width:
        raise RecoveryBatchError(f"{label}: expected {width} components")
    if not all(type(x) in (int, float) and math.isfinite(x) for x in value):
        raise RecoveryBatchError(f"{label}: nonfinite or nonnumeric values")
    return [float(x) for x in value]


def select_action_window(rows, frame, chunk_size):
    """Clamp at ep end and flag padding; do NOT cross an Episode boundary.

    Mirrors LeRobot v3 DatasetReader's clamp indices and action_is_pad. Both
    the actual action and pad mask are needed: padded losses must be ignored.
    """
    if not isinstance(rows, list) or not rows or type(frame) is not int or not 0 <= frame < len(rows):
        raise RecoveryBatchError("Invalid episode rows / frame position")
    if type(chunk_size) is not int or not 1 <= chunk_size <= 100:
        raise RecoveryBatchError("Invalid action chunk size")
    ep = rows[0].get("episode_index")
    task = rows[0].get("task_index")
    for idx, row in enumerate(rows):
        if (row.get("episode_index") != ep or row.get("task_index") != task
                or row.get("frame_index") != idx):
            raise RecoveryBatchError("Episode rows contain mismatched episode/task/frame_index")
    indices = [min(frame + i, len(rows) - 1) for i in range(chunk_size)]
    pad_mask = [(frame + i >= len(rows)) for i in range(chunk_size)]
    vectors = [require_finite_vector(rows[i].get("action"), 7, f"action[{i}]") for i in indices]
    return {"action": vectors, "action_is_pad": pad_mask, "source_frame_indices": indices,
            "valid_action_steps": sum(not x for x in pad_mask)}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for part in iter(lambda: file.read(1048576), b""):
            h.update(part)
    return h.hexdigest()


def verify_manifest_files(root, manifest_path, required):
    manifest_path = Path(manifest_path)
    receipt = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (receipt.get("kind") != "modelscope_recovery_exact_manifest_v1"
            or receipt.get("repo_id") != "ataier/LIBERO_Recovery_Expert"
            or receipt.get("revision") != Path(root).name):
        raise RecoveryBatchError("Expert manifest repo or SHA mismatch")
    known = {}
    for item in receipt.get("files", []):
        rel = item.get("path")
        if not isinstance(rel, str) or rel in known:
            raise RecoveryBatchError("Malformed or duplicate manifest file")
        known[rel] = item
    verified = {}
    for rel in required:
        parsed = PurePosixPath(rel)
        if parsed.is_absolute() or not parsed.parts or ".." in parsed.parts or "\\" in rel:
            raise RecoveryBatchError("Unsafe dataset relative path")
        record = known.get(rel)
        if record is None:
            raise RecoveryBatchError("Manifest omits necessary file: " + rel)
        path = Path(root)/rel
        if (not path.is_file() or path.is_symlink()
                or any(part.is_symlink() for part in path.parents if part == Path(root) or Path(root) in part.parents)
                or not path.resolve().is_relative_to(Path(root))):
            raise RecoveryBatchError("Missing or unsafe Expert payload: " + rel)
        sha = record.get("sha256")
        if not isinstance(sha, str) or len(sha) != 64:
            raise RecoveryBatchError("No usable upstream SHA256 for " + rel)
        if path.stat().st_size != record.get("size_bytes") or sha256_file(path) != sha:
            raise RecoveryBatchError("Remote manifest bytes/hash differ: " + rel)
        verified[rel] = sha
    return verified


def decode_exact_rgb_frame(path, frame, *, width=256, height=256, ffmpeg="ffmpeg"):
    """Decode video by index, NOT approximate -ss: exact RGB bytes expected."""
    if type(frame) is not int or frame < 0:
        raise RecoveryBatchError("Invalid frame number")
    args = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-threads", "2", "-i", str(path),
            "-vf", f"select=eq(n\\,{frame})",
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    try:
        process = subprocess.run(args, capture_output=True, check=False, timeout=90)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RecoveryBatchError("FFmpeg unavailable or timed out") from exc
    if process.returncode != 0 or len(process.stdout) != width * height * 3:
        raise RecoveryBatchError(
            f"Cannot decode exact video frame {frame}: rc={process.returncode}, "
            f"rgb_bytes={len(process.stdout)} expected={width * height * 3} "
            f"stderr={process.stderr[-300:].decode(errors='replace')}"
        )
    return process.stdout


def build_pilot_batch(root, manifest_path, contract_path, *, episode, frame, out_npz, out_report,
                      ffmpeg="ffmpeg"):
    """Materialize one aligned, unnormalized sample using local pilot data.

    All source files remain untouched, and output paths must not exist.
    """
    root = Path(root).expanduser().resolve()
    out_npz, out_report = Path(out_npz), Path(out_report)
    if out_npz == out_report or any(p.exists() or p.is_symlink() for p in (out_npz, out_report)):
        raise RecoveryBatchError("Refuse to overwrite an existing pilot batch/report")
    report_contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    if (report_contract.get("status") != "STATIC_FEATURE_BRIDGE_PROPOSED_NOT_TRAIN_READY"
            or report_contract.get("model_action_dim") != 7
            or report_contract.get("model_state_dim") != 8
            or report_contract.get("expert_fps") != 20
            or report_contract.get("proposed_final_camera_map") != {
                "observation.images.image": "observation.images.camera1",
                "observation.images.wrist_image": "observation.images.camera2"}):
        raise RecoveryBatchError("Static SmolVLA contract is incompatible or absent")
    chunk = report_contract.get("model_chunk_size")
    plan = plan_v21_expert_files(root, [episode])
    descriptor = plan["episode_files"][str(episode)]
    required = [descriptor["data_file"], *descriptor["video_files"].values()]
    manifest_verified = verify_manifest_files(root, manifest_path, required)
    try:
        import numpy as np
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RecoveryBatchError("Require NumPy/PyArrow in dedicated .venvs/data") from exc

    table = pq.read_table(root/descriptor["data_file"], columns=[
        "episode_index", "frame_index", "task_index", "timestamp",
        "observation.state", "action"])
    rows = table.to_pylist()
    if len(rows) != descriptor["length"] or not rows:
        raise RecoveryBatchError("Episode metadata and true Parquet row count differ")
    # Re-check local timestamps and selection for this single sample.
    if not all(isinstance(row.get("timestamp"), (int, float)) and
               math.isfinite(row["timestamp"]) and
               abs(float(row["timestamp"])-(i/20.0)) < 0.003
               for i, row in enumerate(rows)):
        raise RecoveryBatchError("20Hz Parquet frame/timestamp alignment mismatch")
    selected = select_action_window(rows, frame, chunk)
    state = require_finite_vector(rows[frame].get("observation.state"), 8, "observation.state")
    task_values = descriptor.get("tasks")
    if (not isinstance(task_values, list) or len(task_values) != 1
            or not isinstance(task_values[0], str) or not task_values[0].strip()):
        raise RecoveryBatchError("Need exactly one real task instruction")
    task = task_values[0]

    image_paths = {
        "observation.images.image": descriptor["video_files"]["observation.images.image"],
        "observation.images.image2": descriptor["video_files"]["observation.images.wrist_image"],
    }
    arrays = {}
    images = {}
    for target_key, relative in image_paths.items():
        rgb = decode_exact_rgb_frame(root/relative, frame, ffmpeg=ffmpeg)
        arr = np.frombuffer(rgb, dtype=np.uint8).reshape((256, 256, 3))
        # Real LeRobot video frames are float CHW in [0,1], not raw [0,255].
        arrays[target_key] = np.ascontiguousarray(arr.transpose((2, 0, 1)).astype(np.float32) / 255.0)
        images[target_key] = {
            "source_file": relative,
            "source_sha256": manifest_verified[relative],
            "rgb_raw_sha256": hashlib.sha256(rgb).hexdigest(),
            "chw_shape": [3,256,256], "pixel_min": float(arrays[target_key].min()),
            "pixel_max": float(arrays[target_key].max()),
        }
    arrays["observation.state"] = np.asarray(state, dtype=np.float32)
    arrays["action"] = np.asarray(selected["action"], dtype=np.float32)
    arrays["action_is_pad"] = np.asarray(selected["action_is_pad"], dtype=np.bool_)
    arrays["task"] = np.asarray(task)
    if (arrays["action"].shape != (chunk,7) or
            arrays["observation.state"].shape != (8,) or
            arrays["action_is_pad"].shape != (chunk,)):
        raise RecoveryBatchError("Incorrect numeric training sample shape")
    if not all(np.isfinite(array).all() for key, array in arrays.items() if key != "task"):
        raise RecoveryBatchError("Invalid float/image values in materialized arrays")
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creates; no overwrite, even if a prior run partially completed.
    with out_npz.open("xb") as file:
        np.savez_compressed(file, **arrays)
    report = {
        "status": "REAL_RECOVERY_PILOT_SAMPLE_BUILT_NO_TRAINING",
        "ready_to_train": False,
        "dataset_revision": root.name, "episode_index": episode,
        "frame_index": frame, "timestamp": float(rows[frame]["timestamp"]),
        "task": task, "fps": 20, "model_action_chunk": chunk,
        "action_shape": [chunk, 7], "state_shape": [8],
        "action_is_pad_true_count": sum(selected["action_is_pad"]),
        "valid_action_steps": selected["valid_action_steps"],
        "source_frame_indices": selected["source_frame_indices"],
        "camera_preprocessor_keys": list(image_paths),
        "images": images,
        "parquet_source_sha256": manifest_verified[descriptor["data_file"]],
        "contract_report_sha256": sha256_file(contract_path),
        "manifest_sha256": sha256_file(manifest_path),
        "sample_npz_sha256": sha256_file(out_npz),
        "sample_npz_path": str(out_npz.resolve()),
        "outstanding": [
            "No pretrained SmolVLA Processor/tokenizer/model actually loaded",
            "No action/state normalization weights loaded or applied",
            "Camera semantic orientation and action physical units unverified",
            "No LeRobot v2.1 to v3 dataset materialization, GPU gradients or evaluation",
        ]
    }
    with out_report.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return report
