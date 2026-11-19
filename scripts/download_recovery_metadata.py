"""Explicit one-file ModelScope Recovery metadata download, in a separate tooling venv.

No bulk Recovery Expert/Rollouts download. Note that the legacy ModelScope SDK
does not expose a streamed size cap through dataset_file_download; use only
after independent human verification of remote file size and server resources.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", default="ataier/LIBERO_Recovery_Expert")
    p.add_argument("--revision", required=True)
    p.add_argument("--file", default="meta/info.json",
                   choices=["meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"])
    p.add_argument("--out-root", default="/root/autodl-tmp/vla_finetuning/datasets/recovery_metadata")
    p.add_argument("--max-mib", type=int, default=4)
    p.add_argument("--apply", action="store_true")
    a = p.parse_args()
    if not 1 <= a.max_mib <= 8:
        p.error("max-mib must be in 1..8")
    print("MODEL SCOPE PLAN:", a.repo, a.revision, a.file)
    print("WARNING: size is validated AFTER SDK transfer, not bounded during download.")
    if not a.apply:
        print("DRY RUN. No ModelScope API accessed.")
        return
    from vla_finetuning.guards import require_gpu_idle, require_disk
    require_gpu_idle()
    require_disk(a.out_root, free_gib=30)
    from modelscope.hub.file_download import dataset_file_download
    dest = Path(a.out_root) / a.repo.replace("/", "--") / a.revision
    if (dest / a.file).exists():
        p.error("Refuse to overwrite existing file")
    dest.mkdir(parents=True, exist_ok=True)
    path = Path(dataset_file_download(dataset_id=a.repo, file_path=a.file,
                                      revision=a.revision, local_dir=str(dest),
                                      cache_dir=str(Path(a.out_root) / ".cache")))
    if path.stat().st_size > a.max_mib * 1024 * 1024:
        raise RuntimeError("File larger than post-download limit; inspect safely")
    if a.file.endswith(".json"):
        json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps({"status": "MODEL_SCOPE_METADATA_ONLY", "path": str(path),
                      "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
