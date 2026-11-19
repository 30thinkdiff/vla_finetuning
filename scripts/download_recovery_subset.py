"""ModelScope Recovery Expert subset downloader from an independently reviewed inventory.

Explicit per-file entries; no wildcard or whole-dataset implicit download.
The downloader validates sizes after transfers; check source file sizes and
CPU/IO constraints BEFORE applying. No policy training.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def plan_inventory(manifest, max_gib):
    if not isinstance(manifest, dict) or manifest.get("repo_id") != "ataier/LIBERO_Recovery_Expert":
        raise ValueError("Unexpected dataset; audit repository identity first")
    revision = manifest.get("revision")
    if not isinstance(revision, str) or not revision or revision in ("main", "master", "latest"):
        raise ValueError("Use an explicit reviewed, immutable ModelScope revision")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Provide nonempty reviewed list of exact file names and sizes")
    seen = set()
    total = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Malformed file entry")
        name, size = entry.get("path"), entry.get("size_bytes")
        if not isinstance(name, str):
            raise ValueError("Missing exact file path")
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or name in seen:
            raise ValueError("Unsafe/duplicate file path")
        if not name.startswith(("meta/", "data/", "videos/")):
            raise ValueError("Only audited LeRobot metadata and data/video paths")
        seen.add(name)
        if type(size) is not int or size < 0:
            raise ValueError("Review actual source byte sizes first")
        total += size
    if not 0 < max_gib <= 50 or total > max_gib * 1024**3:
        raise ValueError("Inventory exceeds explicit max GiB budget")
    return {"repo_id": manifest["repo_id"], "revision": revision,
            "files": entries, "bytes": total,
            "gib": round(total/1024**3, 2), "status": "PLAN_ONLY"}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path,
                   default=Path("/root/autodl-tmp/vla_finetuning/datasets/recovery_subset"))
    p.add_argument("--max-gib", type=float, default=5)
    p.add_argument("--apply", action="store_true")
    a=p.parse_args(argv)
    plan=plan_inventory(json.loads(a.manifest.read_text()), a.max_gib)
    if not a.apply:
        print(json.dumps(plan, indent=2))
        return
    from vla_finetuning.guards import require_gpu_idle, require_disk
    require_gpu_idle()
    require_disk(a.out_dir, free_gib=max(30, a.max_gib+15))
    from modelscope.hub.file_download import dataset_file_download
    out=a.out_dir.resolve()
    receipt=out/".vla_recovery_download_receipt.json"
    if receipt.exists():
        raise RuntimeError("Receipt exists: refuse to overwrite old experiment download")
    out.mkdir(parents=True, exist_ok=True)
    downloaded=[]
    for entry in plan["files"]:
        dest=out/entry["path"]
        if dest.exists():
            raise RuntimeError("Refuse to overwrite file: "+str(dest))
        result=Path(dataset_file_download(dataset_id=plan["repo_id"],
                                          file_path=entry["path"],
                                          revision=plan["revision"],
                                          local_dir=str(out),
                                          cache_dir=str(out/".cache")))
        if not result.is_file() or result.stat().st_size != entry["size_bytes"]:
            raise RuntimeError("File size mismatch: "+entry["path"])
        downloaded.append({"path":entry["path"], "size_bytes":entry["size_bytes"]})
    receipt.write_text(json.dumps({
        "status":"SUBSET_DOWNLOADED_SIZE_CHECKED_ONLY",
        "repo_id":plan["repo_id"],"revision":plan["revision"],
        "files":downloaded, "origin_manifest_sha256":
            hashlib.sha256(a.manifest.read_bytes()).hexdigest()},indent=2))
    print("SUBSET_DOWNLOADED_SIZE_CHECKED_ONLY; action/observation semantics unverified")


if __name__=="__main__":
    main()
