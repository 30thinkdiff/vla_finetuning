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
import shutil
import subprocess
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit


MODEL = "lerobot/smolvla_libero"
ALLOWED_ENDPOINTS = ("https://huggingface.co", "https://hf-mirror.com")
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


def get_bounded_curl_ipv4(url, *, direct=False, max_bytes=MAX_METADATA_BYTES):
    """Use curl's IPv4 transport when urllib's connection fails in some containers.

    Honors environment proxy unless --curl-direct is selected. Only accepts
    HTTPS requests to Hugging Face and retains the same byte cap as urllib.
    """
    parsed = urlsplit(url)
    if (parsed.scheme != "https" or parsed.netloc not in ("huggingface.co", "hf-mirror.com")
            or parsed.username or parsed.password):
        raise ValueError("Unexpected contract endpoint")
    if not shutil.which("curl"):
        raise ValueError("curl is not installed; cannot use IPv4 fallback")
    args = [
        "curl", "--ipv4", "--fail", "--location", "--silent", "--show-error",
        "--connect-timeout", "10", "--max-time", "35", "--retry", "2",
        "--proto", "=https", "--proto-redir", "=https", "--max-redirs", "4",
        "--max-filesize", str(max_bytes),
    ]
    if direct:
        args.extend(["--noproxy", "*"])
    args.append(url)
    try:
        output = subprocess.run(args, capture_output=True, check=True, timeout=90)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode("utf-8", errors="replace")[-400:]
        raise OSError("Hugging Face curl IPv4 request failed: " + detail) from exc
    except subprocess.TimeoutExpired as exc:
        raise OSError("Hugging Face curl IPv4 request timed out") from exc
    if len(output.stdout) > max_bytes:
        raise ValueError("Downloaded JSON exceeds metadata size limit")
    return output.stdout


def gather_contract(*, opener=None, getter=None, endpoint="https://huggingface.co"):
    if getter is None:
        getter = lambda url: get_bounded(url, opener=opener)
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError("Unapproved Hugging Face metadata endpoint")
    base = endpoint
    info = json.loads(getter(f"{base}/api/models/{MODEL}"))
    sha = info.get("sha")
    if not isinstance(sha, str) or not re.fullmatch("[0-9a-f]{40}", sha):
        raise ValueError("Hugging Face API returned no usable 40-character commit SHA")
    files = {}
    for name in JSON_FILES:
        url = f"{base}/{MODEL}/resolve/{sha}/{name}"
        blob = getter(url)
        parsed = json.loads(blob)
        if not isinstance(parsed, dict):
            raise ValueError("Malformed model contract JSON: " + name)
        files[name] = {"body": blob, "url": url,
                       "sha256": hashlib.sha256(blob).hexdigest()}
    return sha, files


def materialize(out_root, sha, files, *, endpoint="https://huggingface.co"):
    root = Path(out_root).expanduser().resolve()/sha
    if root.exists() or root.is_symlink():
        raise ValueError("Destination revision already exists; refuse overwrite: "+str(root))
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "PINNED_MODEL_JSON_ONLY_NOT_MODEL_WEIGHTS",
                "model_repo": MODEL, "revision": sha, "source_endpoint": endpoint,
                "source_is_third_party_mirror": endpoint != "https://huggingface.co",
                "note": ("Third-party mirror metadata/commit is not independent proof of upstream parity"
                         if endpoint != "https://huggingface.co" else "Official Hugging Face metadata"),
                "files": {}}
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
    parser.add_argument("--endpoint", choices=ALLOWED_ENDPOINTS,
                        default="https://huggingface.co", help="Exact approved Hub endpoint; mirror is third-party")
    parser.add_argument("--transport", choices=["urllib", "curl-ipv4", "curl-ipv4-direct"],
                        default="urllib", help="Use curl IPv4 if urllib fails; -direct bypasses proxies")
    args = parser.parse_args(argv)
    if not args.allow_network:
        parser.exit(2, "STOP: explicit --allow-network required\n")
    try:
        if args.transport == "urllib":
            sha, files = gather_contract(endpoint=args.endpoint)
        else:
            sha, files = gather_contract(
                endpoint=args.endpoint,
                getter=lambda url: get_bounded_curl_ipv4(
                    url, direct=args.transport == "curl-ipv4-direct"))
        path, manifest = materialize(args.out_root, sha, files, endpoint=args.endpoint)
        print(json.dumps({"path":str(path), **manifest}, indent=2, ensure_ascii=False))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        parser.exit(2, "STOP: "+str(exc)+"\n")


if __name__ == "__main__":
    main()
