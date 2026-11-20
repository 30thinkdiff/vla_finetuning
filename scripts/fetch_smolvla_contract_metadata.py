"""Fetch only pinned, tiny public SmolVLA JSON contracts; never model weights.

Network opt-in required. First resolves the current Hugging Face model revision
SHA, then downloads config.json and policy_preprocessor.json using THAT SHA.
A manifest records URL/SHA256 for both files. No Torch or GPU is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path


MODEL = "lerobot/smolvla_libero"
JSON_FILES = ("config.json", "policy_preprocessor.json")
MAX_METADATA_BYTES = 2 * 1024 * 1024


def get_bounded(url, *, opener=None, max_bytes=MAX_METADATA_BYTES):
    opener = opener or urllib.request.urlopen
    request = urllib.request.Request(url, headers={"User-Agent": "Recovery-SmolVLA-contract/1.0"})
    with opener(request, timeout=30) as response:
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > max_bytes:
            raise ValueError("Remote JSON exceeds metadata size limit")
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("Downloaded JSON exceeds metadata size limit")
    return data


def gather_contract(*, opener=None):
    base = "https://huggingface.co"
    info = json.loads(get_bounded(f"{base}/api/models/{MODEL}", opener=opener))
    sha = info.get("sha")
    if not isinstance(sha, str) or not re.fullmatch("[0-9a-f]{40}", sha):
        raise ValueError("Hugging Face API returned no usable 40-character commit SHA")
    files = {}
    for name in JSON_FILES:
        url = f"{base}/{MODEL}/resolve/{sha}/{name}"
        blob = get_bounded(url, opener=opener)
        parsed = json.loads(blob)
        if not isinstance(parsed, dict):
            raise ValueError("Malformed model contract JSON: " + name)
        files[name] = {"body": blob, "url": url,
                       "sha256": hashlib.sha256(blob).hexdigest()}
    return sha, files


def materialize(out_root, sha, files):
    root = Path(out_root).expanduser().resolve()/sha
    if root.exists() or root.is_symlink():
        raise ValueError("Destination revision already exists; refuse overwrite: "+str(root))
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "PINNED_MODEL_JSON_ONLY_NOT_MODEL_WEIGHTS",
                "model_repo": MODEL, "revision": sha, "files": {}}
    for name in JSON_FILES:
        item = files[name]
        with (root/name).open("xb") as fp:
            fp.write(item["body"])
        manifest["files"][name] = {
            "url": item["url"], "sha256": item["sha256"],
            "bytes": len(item["body"]),
        }
    with (root/"receipt.json").open("x", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, ensure_ascii=False)
        fp.write("\n")
    return root, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args(argv)
    if not args.allow_network:
        parser.exit(2, "STOP: explicit --allow-network required\n")
    try:
        sha, files = gather_contract()
        path, manifest = materialize(args.out_root, sha, files)
        print(json.dumps({"path":str(path), **manifest}, indent=2, ensure_ascii=False))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, "STOP: "+str(exc)+"\n")


if __name__ == "__main__":
    main()
