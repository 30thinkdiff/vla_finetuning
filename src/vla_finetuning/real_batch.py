"""Read exactly one real LeRobot sample on CPU; never instantiate a policy.

Requires a separate pinned LeRobot + torch + PyAV environment. The sample may
decode two video frames, so do not run while Corrector I/O is unexamined.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path


def inspect(repo, revision, root, episode):
    if not Path(root).is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {root}")
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        raise RuntimeError("Explicitly set CUDA_VISIBLE_DEVICES='' to ensure CPU-only sample audit")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Refusing GPU-enabled real-batch sample audit")
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    torch.set_num_threads(1)
    dataset = LeRobotDataset(repo_id=repo, root=root, revision=revision,
                             episodes=[episode], video_backend="pyav", token=False)
    if not len(dataset):
        raise RuntimeError("No usable frames in selected episode")
    item = dataset[0]
    fields = {}
    for key, value in item.items():
        if torch.is_tensor(value):
            payload = {"shape": list(value.shape), "dtype": str(value.dtype),
                       "device": str(value.device)}
            if value.is_floating_point():
                payload["finite"] = bool(torch.isfinite(value).all())
                payload["min"] = float(value.min()) if value.numel() else None
                payload["max"] = float(value.max()) if value.numel() else None
            fields[key] = payload
        else:
            fields[key] = {"python_type": type(value).__name__, "preview": str(value)[:120]}
    return {"status": "REAL_CPU_SAMPLE_READ_NOT_TRAINING",
            "repo_id": repo, "revision": revision, "episode": episode,
            "item_keys": sorted(item), "fields": fields,
            "len_selected_dataset": len(dataset), "dataset_fps": dataset.meta.fps,
            "warnings": ["No policy/processor was loaded", "No action chunk supervised yet",
                         "Camera and robot action semantics require a separate provenance audit"]}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--out")
    args = p.parse_args(argv)
    data = inspect(args.repo, args.revision, args.root, args.episode)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", encoding="utf8") as f:
            json.dump(data, f, indent=2)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
