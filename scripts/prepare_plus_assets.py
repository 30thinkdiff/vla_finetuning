"""Pinned, resource-gated LIBERO-plus assets ZIP downloader and safe extractor.

Never mutates LIBERO-plus source tree. Extraction goes to a new staging folder;
the user must inspect the internal tree before installing into simulator.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath


def audit_archive(path, *, max_expanded_gib=16):
    entries, total = [], 0
    with zipfile.ZipFile(path) as z:
        seen = set()
        for info in z.infolist():
            rel = PurePosixPath(info.filename)
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError("ZIP path traversal")
            if info.filename in seen:
                raise ValueError("Duplicate ZIP entry")
            seen.add(info.filename)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("No ZIP symlinks permitted")
            total += info.file_size
            if total > max_expanded_gib * 1024**3:
                raise ValueError("Archive uncompressed size exceeds reviewed budget")
            if info.compress_size and info.file_size > info.compress_size * 1000:
                raise ValueError("Suspicious expansion ratio")
            entries.append(info.filename)
    return {"files": len(entries), "expanded_bytes": total, "prefixes": sorted({x.split("/")[0] for x in entries})}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--revision", required=True, help="Fixed Hugging Face dataset 40-hex SHA")
    p.add_argument("--repo", default="Sylvest/LIBERO-plus")
    p.add_argument("--dest", default="/root/autodl-tmp/vla_finetuning/datasets/plus_assets_staging")
    p.add_argument("--max-zip-gib", type=float, default=8)
    p.add_argument("--max-expanded-gib", type=float, default=16)
    p.add_argument("--allow-network", action="store_true")
    p.add_argument("--apply", action="store_true")
    p.add_argument("--extract", action="store_true")
    a = p.parse_args()
    from vla_finetuning.guards import require_sha, require_gpu_idle, require_disk
    require_sha(a.revision, "assets repo revision")
    if not a.allow_network:
        p.error("Explicit --allow-network required to inspect remote assets")
    from huggingface_hub import HfApi
    api = HfApi()
    info = api.dataset_info(a.repo, revision=a.revision, files_metadata=True)
    if info.sha != a.revision:
        raise RuntimeError("Asset SHA mismatch")
    match = [f for f in info.siblings if f.rfilename == "assets.zip"]
    if len(match) != 1 or type(match[0].size) is not int:
        raise RuntimeError("Expected exact assets.zip with reported byte size")
    size = match[0].size
    if size > a.max_zip_gib * 1024**3:
        raise RuntimeError("ZIP exceeds requested budget")
    plan = {"repo": a.repo, "revision": a.revision, "filename": "assets.zip",
            "zip_bytes": size, "download_requested": a.apply, "extract_requested": a.extract}
    if not a.apply:
        print(json.dumps({"status":"PLAN_ONLY",**plan}, indent=2))
        return
    require_gpu_idle()
    require_disk(a.dest, free_gib=max(30, a.max_zip_gib+a.max_expanded_gib+10))
    from huggingface_hub import hf_hub_download
    dest = Path(a.dest)
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "assets.zip"
    if zip_path.exists():
        raise RuntimeError("Existing assets.zip: review instead of overwriting")
    downloaded = Path(hf_hub_download(repo_id=a.repo, repo_type="dataset",
                                     revision=a.revision, filename="assets.zip",
                                     local_dir=str(dest), cache_dir=str(dest / ".cache")))
    if downloaded.stat().st_size != size:
        raise RuntimeError("Downloaded ZIP size differs from Hub inventory")
    audit = audit_archive(downloaded, max_expanded_gib=a.max_expanded_gib)
    sha256 = hashlib.sha256()
    with downloaded.open("rb") as inp:
        for blob in iter(lambda: inp.read(1024*1024), b""):
            sha256.update(blob)
    # Extract only into a *new* stage; existing vendor/simulator files are untouched.
    if a.extract:
        stage = dest / "extracted"
        if stage.exists():
            raise RuntimeError("Refuse to overwrite existing extracted assets folder")
        stage.mkdir()
        with zipfile.ZipFile(downloaded) as archive:
            for member in archive.infolist():
                rel = PurePosixPath(member.filename)
                target = stage.joinpath(*rel.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    with archive.open(member) as src, target.open("xb") as out:
                        import shutil
                        shutil.copyfileobj(src, out, 1024*1024)
    print(json.dumps({"status":"ZIP_STAGED_NOT_INSTALLED",**plan,**audit,
                      "sha256":sha256.hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
