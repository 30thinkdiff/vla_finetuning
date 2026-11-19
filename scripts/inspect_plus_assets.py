"""Inspect an existing LIBERO-plus assets ZIP before extraction.

Does not download or extract. Rejects traversal, symlinks, huge expansion,
duplicates and suspicious compression ratios. Use a separate reviewed manual
extraction step AFTER simulation environment compatibility checks.
"""
import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import PurePosixPath, Path


def audit_zip(path, max_uncompressed_gib=12):
    if not 0 < max_uncompressed_gib <= 40:
        raise ValueError("Use a bounded 0<max_uncompressed_gib<=40")
    path = Path(path)
    seen = set()
    count = 0
    total = 0
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            name = PurePosixPath(info.filename)
            if name.is_absolute() or any(part in ("..", "") for part in name.parts):
                raise ValueError("Unsafe zip path")
            if info.filename in seen:
                raise ValueError("Duplicate zip entry")
            seen.add(info.filename)
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("Symlinks are not allowed in assets zip")
            if info.is_dir():
                continue
            count += 1
            total += info.file_size
            if total > max_uncompressed_gib * 1024**3:
                raise ValueError("ZIP would expand beyond disk budget")
            if info.compress_size and info.file_size / info.compress_size > 1000:
                raise ValueError("Suspicious individual ZIP compression ratio")
    return {"status": "ZIP_AUDITED_NOT_EXTRACTED", "zip": str(path.resolve()),
            "entries": count, "uncompressed_gib": round(total / 1024**3, 2),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.stat().st_size < 256*1024**2 else "NOT_HASHED_LARGE_ZIP"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--max-gib", type=float, default=12)
    a = p.parse_args()
    print(json.dumps(audit_zip(a.zip, a.max_gib), indent=2))


if __name__ == "__main__":
    main()
