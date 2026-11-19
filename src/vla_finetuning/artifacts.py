"""Constrained, revision-pinned Hugging Face snapshots.

Planner may query Hub metadata only with --allow-network.
Downloader needs --apply, exact SHA, a size budget and an idle GPU. It performs
no training and requests max_workers=1 to reduce I/O pressure.

For v3 data, use dataset_files.plan_v3_episode_shards to generate exact payload
paths; never use wildcard all-data for an unreviewed dataset.
"""
from __future__ import annotations
import fnmatch
import json
from pathlib import Path
from .core import ContractError, GIB
from .guards import (require_sha, require_explicit, require_disk, require_gpu_idle,
                     ensure_cache_on_data_disk)


def hub_sizes(repo, revision, *, repo_type, allow_network=False, api=None):
    require_sha(revision, "Hub revision")
    if not allow_network:
        raise ContractError("Explicit --allow-network required for Hub API")
    if repo_type not in ("dataset", "model"):
        raise ContractError("repo_type must be dataset or model")
    if api is None:
        try:
            from huggingface_hub import HfApi
        except ImportError as exc:
            raise ContractError("huggingface_hub must be installed in a separate tooling venv") from exc
        api = HfApi()
    info = (api.dataset_info(repo_id=repo, revision=revision, files_metadata=True)
            if repo_type == "dataset" else
            api.model_info(repo_id=repo, revision=revision, files_metadata=True))
    if getattr(info, "sha", None) != revision:
        raise ContractError("Hub resolved a different repository revision; stop")
    out = {}
    for x in info.siblings or []:
        name, size = x.rfilename, x.size
        if not name or name.startswith("/") or ".." in Path(name).parts:
            raise ContractError("Unexpected Hub filename")
        if name in out:
            raise ContractError("Duplicate Hub file")
        out[name] = size
    if not out:
        raise ContractError("Hub returned no file inventory")
    return out


def file_plan(repo, revision, *, kind, selected_paths=None, inventory=None):
    """Pure plan. Supply exact relative files for data; model uses recognized artifacts."""
    require_sha(revision, "revision")
    if kind not in ("dataset_metadata", "dataset_payload", "model"):
        raise ContractError("Unsupported artifact kind")
    if inventory is None or not isinstance(inventory, dict):
        raise ContractError("Supply Hub inventory with real per-file sizes")
    if kind == "dataset_metadata":
        paths = sorted(p for p in inventory if p.startswith("meta/"))
    elif kind == "dataset_payload":
        if not isinstance(selected_paths, list) or not selected_paths:
            raise ContractError("Exact v3 shard manifest required; no full dataset wildcard")
        paths = sorted(set(selected_paths))
        if any(not (p.startswith("data/") and p.endswith(".parquet") or
                    p.startswith("videos/") and p.endswith(".mp4")) for p in paths):
            raise ContractError("Payload must only include validated data Parquet / video MP4 paths")
    else:
        paths = sorted(p for p in inventory if
                       p.endswith((".json", ".txt", ".model", ".safetensors", ".bin",
                                   ".tiktoken", ".jsonl", ".py", ".md"))
                       and not p.startswith((".git/", "datasets/", "data/")))
        if not any(p.endswith((".safetensors", ".bin")) for p in paths):
            raise ContractError("No model weight artifacts found in snapshot")
    missing = [p for p in paths if p not in inventory]
    if not paths or missing:
        raise ContractError("Missing requested source files: " + repr(missing[:20]))
    if any(type(inventory[p]) is not int or inventory[p] < 0 for p in paths):
        raise ContractError("Hub did not report exact file sizes; refuse unbounded download")
    total = sum(inventory[p] for p in paths)
    return {"kind": kind, "repo_id": repo, "revision": revision,
            "files": paths, "file_count": len(paths), "total_bytes": total,
            "total_gib": round(total/GIB, 3),
            "warning": "Only an estimate from Hub metadata. Verify disk before and after download.",
            "status": "PLANNED_ONLY_NO_DOWNLOAD"}


def download_plan(plan, target, project_root, *, apply=False, max_gib=1,
                  allow_busy=False, client=None, runner=None):
    """Apply the exact plan only by explicit user command. Returns local path."""
    require_explicit(apply, "download")
    for key in ("repo_id", "revision", "kind", "files", "total_bytes"):
        if key not in plan:
            raise ContractError("Incomplete downloaded plan: " + key)
    require_sha(plan["revision"], "download revision")
    if type(max_gib) not in (int, float) or not (0 < max_gib <= 50):
        raise ContractError("A strict 0<max_gib<=50 size budget is mandatory")
    if plan["total_bytes"] > max_gib*GIB:
        raise ContractError("Planned download exceeds user-specified GiB budget")
    if not allow_busy:
        require_gpu_idle(runner=runner)
    else:
        # This is an explicit override after the user has examined I/O resources.
        # Never invoked as an automatic fallback.
        pass
    require_disk(target, free_gib=max(30, max_gib+15))
    env = ensure_cache_on_data_disk(project_root)
    if client is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ContractError("Install huggingface_hub in independent download venv") from exc
        client = snapshot_download
    target = Path(target).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    # Hugging Face SDK uses process env for cache; use explicit local_dir/cache_dir
    # instead of mutating os.environ of the caller.
    result = client(repo_id=plan["repo_id"], repo_type="dataset" if plan["kind"] != "model" else "model",
                    revision=plan["revision"], local_dir=str(target),
                    cache_dir=env["HF_HUB_CACHE"], allow_patterns=list(plan["files"]),
                    max_workers=1)
    missing = [name for name in plan["files"] if not (target / name).is_file()]
    if missing:
        raise ContractError("Incomplete artifact snapshot: " + repr(missing[:10]))
    receipt = target / (".vla_receipt_" + plan["kind"] + ".json")
    if receipt.exists():
        raise ContractError("Download receipt already exists; refusing overwrite: " + str(receipt))
    import json
    import hashlib
    records = {name: {"size": (target/name).stat().st_size} for name in plan["files"]}
    receipt_data = {"kind": plan["kind"], "repo_id": plan["repo_id"],
                    "revision": plan["revision"], "files": records,
                    "source_plan_status": plan.get("status"), "bytes_planned": plan["total_bytes"],
                    "sha256_note": "Remote revision pinned; file content hashes not independently computed"}
    with receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt_data, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return {"status": "DOWNLOAD_COMPLETE_FILES_PRESENT", "path": str(result),
            "file_count": len(plan["files"]), "missing_count": 0,
            "receipt": str(receipt),
            "note": "File presence/size checked; semantic and full hash verification remain pending."}
