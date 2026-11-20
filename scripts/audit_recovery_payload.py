"""Read-only Recovery Expert payload audit (Parquet + AV1 MP4 + upstream hashes).

Only reads previously downloaded files; requires pyarrow and optionally ffprobe.
Never loads Torch/CUDA, simulator, checkpoints or unsafe pickle data.

Example:
  PYTHONPATH=src .venvs/data/bin/python scripts/audit_recovery_payload.py \
    --dataset-root datasets/recovery_expert/<pinned_commit> \
    --manifest runs/recovery_pilot3_manifest_<pinned_commit>.json \
    --episodes 0,15,30 --out runs/recovery_payload_audit_<pinned_commit>.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import shutil
import statistics
import subprocess
from collections import Counter
from fractions import Fraction
from pathlib import Path

from vla_finetuning.recovery_v21 import plan_v21_expert_files

REQUIRED_PARQUET = (
    "action", "observation.state", "timestamp", "frame_index",
    "episode_index", "task_index",
)


class PayloadAuditError(ValueError):
    pass


def sha256_file(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def numeric_vector(value, dims, label):
    if not isinstance(value, (list, tuple)) or len(value) != dims:
        raise PayloadAuditError(f"{label} must contain exactly {dims} numbers")
    nums = []
    for x in value:
        if type(x) not in (float, int) or not math.isfinite(x):
            raise PayloadAuditError(f"{label} contains nonnumeric/nonfinite values")
        nums.append(float(x))
    return nums


def summarize_vectors(values, dims, label):
    if not values:
        raise PayloadAuditError(f"No {label} vectors")
    columns = [[] for _ in range(dims)]
    for v in values:
        for col, item in zip(columns, numeric_vector(v, dims, label)):
            col.append(item)
    return {
        "count": len(values),
        "dimensions": [
            {"index": i, "min": min(col), "max": max(col),
             "mean": round(statistics.fmean(col), 8),
             "first": col[0], "last": col[-1]}
            for i, col in enumerate(columns)
        ],
        "last_dim_rounded_values_sample": sorted({round(v[-1], 4) for v in values})[:35],
        "first_vector": numeric_vector(values[0], dims, label),
        "last_vector": numeric_vector(values[-1], dims, label),
    }


def frame_integrity(rows, expected_ep, fps):
    if not rows:
        raise PayloadAuditError("Empty parquet rows")
    frames, stamps = [], []
    tasks = set()
    for row in rows:
        ep = row["episode_index"]
        frame = row["frame_index"]
        t = row["timestamp"]
        task = row["task_index"]
        if type(ep) is not int or ep != expected_ep:
            raise PayloadAuditError("Unexpected episode_index inside Parquet")
        if type(frame) is not int or type(task) is not int:
            raise PayloadAuditError("frame_index/task_index must be integers")
        if type(t) not in (int, float) or not math.isfinite(t):
            raise PayloadAuditError("Invalid timestamp")
        frames.append(frame)
        stamps.append(float(t))
        tasks.add(task)
    if frames != list(range(len(rows))):
        raise PayloadAuditError("Noncontiguous or nonzero frame_index sequence")
    if len(tasks) != 1:
        raise PayloadAuditError("Task index varies within one Episode")
    dt = [stamps[i+1]-stamps[i] for i in range(len(stamps)-1)]
    if any(t <= 0 for t in dt):
        raise PayloadAuditError("Nonincreasing timestamps")
    median = statistics.median(dt) if dt else None
    return {
        "rows": len(rows), "task_index": next(iter(tasks)),
        "frame_start_end": [frames[0], frames[-1]],
        "timestamp_start_end": [stamps[0], stamps[-1]],
        "median_dt": median, "nominal_dt": 1.0/fps,
        "max_dt_deviation_from_nominal": max((abs(x-1/fps) for x in dt), default=0.0),
        "timing_within_3ms_of_nominal": all(abs(x-1/fps) < 0.003 for x in dt),
    }


def ffprobe_video(path, expected_frames, *, expected_fps=20, ffprobe="ffprobe"):
    process = subprocess.run(
        [ffprobe, "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
         "-of", "json", str(path)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if process.returncode != 0:
        raise PayloadAuditError(f"ffprobe failed for {path}: {process.stderr[-500:]}")
    try:
        streams = json.loads(process.stdout)["streams"]
        if len(streams) != 1:
            raise PayloadAuditError("Expected exactly one video stream")
        stream = streams[0]
        frames = int(stream["nb_read_frames"])
        fps = float(Fraction(stream["avg_frame_rate"]))
    except (ValueError, TypeError, KeyError, IndexError, ZeroDivisionError) as exc:
        raise PayloadAuditError("Missing valid video codec/fps/decoded frame count") from exc
    result = {
        "codec": stream.get("codec_name"),
        "width": stream.get("width"), "height": stream.get("height"),
        "fps": fps, "decoded_frames": frames, "expected_frames": expected_frames,
    }
    if result["codec"] != "av1" or result["width"] != 256 or result["height"] != 256:
        raise PayloadAuditError("Video codec/resolution differs from observed Expert info.json")
    if abs(fps-expected_fps) > 0.001 or frames != expected_frames:
        raise PayloadAuditError("Video FPS or frame count does not match episode")
    return result


def audit(root, manifest_path, episodes, *, probe_videos=True):
    root = Path(root).expanduser().resolve()
    manifest_path = Path(manifest_path)
    plan = plan_v21_expert_files(root, episodes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("kind") != "modelscope_recovery_exact_manifest_v1"
            or manifest.get("repo_id") != "ataier/LIBERO_Recovery_Expert"
            or manifest.get("revision") != root.name):
        raise PayloadAuditError("Manifest repo/revision does not match real local Expert directory")
    records = {}
    for record in manifest.get("files", []):
        path = record["path"]
        if not isinstance(path, str) or path.startswith("/") or "\\" in path or ".." in Path(path).parts:
            raise PayloadAuditError("Unsafe manifest path")
        if path in records:
            raise PayloadAuditError("Duplicate file in manifest")
        records[path] = record
    required = set(plan["payload_files"])
    if not required.issubset(records):
        raise PayloadAuditError("Manifest does not contain all selected episode files")
    for relative in sorted(required):
        file = root/relative
        if not file.is_file() or file.is_symlink() or file.resolve() != (root/relative).resolve():
            raise PayloadAuditError("Missing/unsafe file: " + relative)
        record = records[relative]
        if file.stat().st_size != record["size_bytes"]:
            raise PayloadAuditError("Size mismatch: " + relative)
        expected_sha = record.get("sha256")
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise PayloadAuditError("Require upstream SHA256 in pilot file manifest")
        if sha256_file(file) != expected_sha:
            raise PayloadAuditError("SHA256 mismatch: " + relative)
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise PayloadAuditError("pyarrow is required in isolated .venvs/data environment") from exc

    ffprobe = shutil.which("ffprobe") if probe_videos else None
    results = {}
    for episode in episodes:
        descriptor = plan["episode_files"][str(episode)]
        path = root/descriptor["data_file"]
        pf = pq.ParquetFile(path)
        if not set(REQUIRED_PARQUET).issubset(pf.schema_arrow.names):
            raise PayloadAuditError("Parquet missing required fields: " + str(path))
        if pf.metadata.num_rows != descriptor["length"]:
            raise PayloadAuditError("Parquet row count differs from episode length")
        table = pq.read_table(path, columns=list(REQUIRED_PARQUET))
        rows = table.to_pylist()
        timeline = frame_integrity(rows, episode, plan["observed_fps"])
        state_summary = summarize_vectors([r["observation.state"] for r in rows], 8, "state")
        action_summary = summarize_vectors([r["action"] for r in rows], 7, "action")
        videos = {}
        for cam, relative in descriptor["video_files"].items():
            videos[cam] = (
                ffprobe_video(root/relative, len(rows), expected_fps=plan["observed_fps"], ffprobe=ffprobe)
                if ffprobe else {"status": "NOT_PROBED", "reason": "ffprobe unavailable or explicitly skipped"}
            )
        results[str(episode)] = {
            "task_text_from_index": descriptor["tasks"],
            "timeline": timeline, "state": state_summary, "action": action_summary,
            "videos": videos,
        }

    return {
        "status": "PAYLOAD_DECODE_CHECKED_SEMANTICS_NOT_VERIFIED" if ffprobe else "VIDEO_PROBE_PENDING",
        "source_revision": manifest["revision"],
        "manifest_sha256": sha256_file(manifest_path),
        "total_episodes": len(episodes), "total_parquet_rows": sum(r["timeline"]["rows"] for r in results.values()),
        "hashes_verified_against_remote_manifest": len(required),
        "videos_probed": bool(ffprobe),
        "episodes": results,
        "training_ready": False,
        "outstanding_semantics": ["action coordinate and rotation conventions", "gripper encoding",
                                  "camera flip/orientation and processor mapping",
                                  "normalization", "source failure-scene provenance"],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--episodes", required=True, help="Comma-separated Episode IDs")
    parser.add_argument("--skip-video-probe", action="store_true")
    parser.add_argument("--out", required=True, help="Write new JSON; never overwrite")
    args = parser.parse_args(argv)
    try:
        ids = [int(x) for x in args.episodes.split(",")]
        data = audit(args.dataset_root, args.manifest, ids, probe_videos=not args.skip_video_probe)
        path = Path(args.out)
        if path.exists() or path.is_symlink():
            raise PayloadAuditError("Refuse to overwrite existing audit JSON")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        parser.exit(2, "STOP: "+str(exc)+"\n")


if __name__ == "__main__":
    main()
