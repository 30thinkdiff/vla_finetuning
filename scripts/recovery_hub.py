"""Opt-in ModelScope Recovery inventory, exact bounded downloads; stdlib until network.

External dependency for network operations: modelscope-hub>=0.4.5.
No GPU/model/simulator use. Remote sizes are budget estimates, NOT hard
streaming caps; SHA256 receipt and size verified after each file transfer.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath

REPOS = {"ataier/LIBERO_Recovery_Expert", "ataier/LIBERO_Recovery_FailureScenes"}


def safe_path(p):
    if not isinstance(p, str) or not p or p.startswith("/") or "\\" in p or "//" in p or ":" in p:
        raise ValueError("Unsafe path: " + repr(p))
    if any(x in (".", "..") for x in PurePosixPath(p).parts):
        raise ValueError("Unsafe traversal: " + repr(p))
    return p


def write_new(path, obj):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Refuse symlink output")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(obj, file, indent=2, ensure_ascii=False, sort_keys=True)
        file.write("\n")


def inventory_from_remote(repo, rev, api):
    if repo not in REPOS or not rev or rev in ("master", "main", "latest"):
        raise ValueError("Use reviewed non-default revision")
    files = {}
    for info in api.list_repo_files(repo, "dataset", revision=rev, recursive=True):
        if getattr(info, "type", "blob") not in ("blob", "file"):
            continue
        name = safe_path(info.path)
        size = info.size
        if name in files or type(size) is not int or size < 0:
            raise ValueError("Duplicate/invalid remote file or size")
        sha = getattr(info, "sha256", None)
        if sha and not re.fullmatch("[0-9a-f]{64}", sha):
            raise ValueError("Invalid remote file digest")
        files[name] = {"path": name, "size_bytes": size, "sha256": sha}
    if not files:
        raise ValueError("Empty remote inventory")
    return {"repo_id": repo, "revision": rev, "kind": "modelscope_recovery_inventory_v1",
            "files": [files[k] for k in sorted(files)],
            "total_bytes": sum(v["size_bytes"] for v in files.values())}


def exact_manifest(inv, paths, max_gib):
    if inv.get("repo_id") not in REPOS or not inv.get("revision") or inv["revision"] in ("master", "main", "latest"):
        raise ValueError("Unknown repo or mutable revision")
    if type(max_gib) not in (int, float) or not math.isfinite(max_gib) or not (0 < max_gib <= 50):
        raise ValueError("Require max-gib in (0,50]")
    lookup = {}
    for x in inv.get("files", []):
        p = safe_path(x["path"])
        if p in lookup or type(x["size_bytes"]) is not int or x["size_bytes"] < 0:
            raise ValueError("Unknown/negative/duplicate byte size")
        lookup[p] = x
    if not paths:
        raise ValueError("No selected files")
    selected = sorted({safe_path(p) for p in paths})
    if any(p not in lookup for p in selected):
        raise ValueError("Selected file absent from published inventory")
    if any(lookup[p]["size_bytes"] <= 0 for p in selected):
        raise ValueError("Selected file has missing or zero byte size")
    total = sum(lookup[p]["size_bytes"] for p in selected)
    if total > max_gib * 1024**3:
        raise ValueError("Download exceeds budget")
    return {"kind": "modelscope_recovery_exact_manifest_v1",
            "repo_id": inv["repo_id"], "revision": inv["revision"],
            "files": [lookup[p] for p in selected], "total_bytes": total,
            "status": "PLAN_ONLY_NOT_DOWNLOADED",
            "revision_sha40": bool(re.fullmatch("[0-9a-f]{40}", inv["revision"]))}


def sha256(path):
    dig = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            dig.update(chunk)
    return dig.hexdigest()


def sdk_download_exact(api, *, repo_id, revision, path, target, upstream_sha256=None):
    """Call the *public* modelscope-hub 0.4.5 API (no file_size argument)."""
    return Path(api.download_file(
        repo_id, "dataset", path, revision=revision,
        local_dir=target, expected_sha256=upstream_sha256
    ))


def download(manifest, target, max_gib, api, receipt_id="snapshot"):
    from vla_finetuning.guards import require_gpu_idle, require_disk
    if manifest.get("kind") != "modelscope_recovery_exact_manifest_v1":
        raise ValueError("Wrong manifest kind")
    checked = exact_manifest(manifest, [x["path"] for x in manifest["files"]], max_gib)
    require_gpu_idle()
    target = Path(target).expanduser().resolve()
    if not str(target).startswith("/root/autodl-tmp/"):
        raise ValueError("Require persistent AutoDL data disk")
    require_disk(target, free_gib=max(30, 30+2*checked["total_bytes"]/1024**3))
    if not re.fullmatch("[a-z0-9][a-z0-9_-]{0,31}", receipt_id):
        raise ValueError("Unsafe receipt ID")
    receipt = target/(".vla_recovery_verified_"+receipt_id+".json")
    if receipt.exists() or receipt.is_symlink():
        raise ValueError("Existing receipt; refusing to overwrite source snapshot")
    for x in checked["files"]:
        p = target/x["path"]
        if p.is_symlink() or any(a.is_symlink() for a in p.parents if a != target and target in a.parents):
            raise ValueError("Symlink destination detected: "+str(p))
        if p.exists() and (not x.get("sha256") or p.stat().st_size != x["size_bytes"] or
                           sha256(p) != x["sha256"]):
            raise ValueError("Cannot safely reuse existing file without matching remote SHA256: "+str(p))
    target.mkdir(parents=True, exist_ok=True)
    results = []
    for x in checked["files"]:
        p = target/x["path"]
        if not p.is_file():
            downloaded = sdk_download_exact(
                api, repo_id=manifest["repo_id"], revision=manifest["revision"],
                path=x["path"], target=target, upstream_sha256=x.get("sha256")
            )
            if downloaded.resolve() != p.resolve():
                raise ValueError("Unexpected SDK file destination")
        if not p.is_file() or p.stat().st_size != x["size_bytes"]:
            raise ValueError("Downloaded file size mismatch: "+str(p))
        digest = sha256(p)
        if x.get("sha256") and x["sha256"] != digest:
            raise ValueError("Downloaded SHA256 mismatch: "+str(p))
        results.append({"path": x["path"], "size_bytes": p.stat().st_size, "sha256": digest})
        print("VERIFIED", x["path"], flush=True)
    receipt_obj = {"repo_id": manifest["repo_id"], "revision": manifest["revision"],
                   "status": "LOCAL_SIZE_CHECKED_SHA_RECORDED_SEMANTICS_UNVERIFIED",
                   "files": results}
    write_new(receipt, receipt_obj)
    return {"receipt": str(receipt), "files": len(results), "status": receipt_obj["status"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="operation", required=True)
    x = sub.add_parser("revisions")
    x.add_argument("--repo", choices=sorted(REPOS), required=True)
    x.add_argument("--allow-network", action="store_true")
    x = sub.add_parser("inventory")
    x.add_argument("--repo", choices=sorted(REPOS), required=True)
    x.add_argument("--revision", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--allow-network", action="store_true")
    x = sub.add_parser("manifest")
    x.add_argument("--inventory", required=True)
    source = x.add_mutually_exclusive_group(required=True)
    source.add_argument("--paths", help="Comma-separated exact file paths")
    source.add_argument("--episode-plan", help="JSON result from recovery_cli plan-v21-files")
    x.add_argument("--max-gib", type=float, required=True)
    x.add_argument("--out", required=True)
    x = sub.add_parser("download")
    x.add_argument("--manifest", required=True)
    x.add_argument("--target", required=True)
    x.add_argument("--max-gib", type=float, required=True)
    x.add_argument("--apply", action="store_true")
    x.add_argument("--receipt-id", default="snapshot", help="Unique identifier for incremental stage")
    a = p.parse_args(argv)
    try:
        if a.operation in ("revisions", "inventory") and not a.allow_network:
            raise ValueError("Network requires explicit --allow-network")
        if a.operation == "revisions":
            from modelscope_hub import HubApi
            out = {"repo": a.repo, "revisions": HubApi().list_repo_revisions(a.repo, "dataset")}
        elif a.operation == "inventory":
            from modelscope_hub import HubApi
            inv = inventory_from_remote(a.repo, a.revision, HubApi())
            write_new(a.out, inv)
            out = {"inventory": a.out, "files": len(inv["files"]),
                   "total_gib": round(inv["total_bytes"]/1024**3, 3),
                   "metadata": [x for x in inv["files"] if x["path"].startswith("meta/")][:20]}
        elif a.operation == "manifest":
            inv = json.loads(Path(a.inventory).read_text(encoding="utf-8"))
            if a.episode_plan:
                plan = json.loads(Path(a.episode_plan).read_text(encoding="utf-8"))
                if plan.get("status") != "V21_FILE_PLAN_ONLY_NOT_DOWNLOADED":
                    raise ValueError("Invalid v2.1 Episode plan")
                paths = plan["payload_files"]
            else:
                paths = a.paths.split(",")
            out = exact_manifest(inv, paths, a.max_gib)
            write_new(a.out, out)
        elif a.operation == "download":
            manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
            if not a.apply:
                out = exact_manifest(manifest, [x["path"] for x in manifest["files"]], a.max_gib)
            else:
                from modelscope_hub import HubApi
                out = download(manifest, a.target, a.max_gib, HubApi(), receipt_id=a.receipt_id)
        else:
            raise AssertionError("unreachable")
        print(json.dumps(out, indent=2, ensure_ascii=False, default=str))
    except (ImportError, KeyError, TypeError, OSError, ValueError) as exc:
        p.exit(2, "STOP: "+str(exc)+"\n")


if __name__ == "__main__":
    main()
