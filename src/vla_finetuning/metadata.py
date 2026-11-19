"""LeRobot v2/v3 metadata audit and explicit, bounded Hugging Face metadata fetch.

Never downloads model weights, videos, or training parquet. No GPU/PyTorch.
"""
import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from .core import ContractError, audit_lerobot_info

REPO = re.compile(r"^[\w.-]+/[\w.-]+$", re.ASCII)
SHA = re.compile(r"^[0-9a-fA-F]{40}$")
ALLOWED = {"meta/info.json", "meta/tasks.parquet", "meta/tasks.jsonl",
           "meta/episodes.jsonl", "meta/stats.json"}


def check_repo(repo):
    if not isinstance(repo, str) or not REPO.fullmatch(repo) or any(
        x in (".", "..") for x in repo.split("/")
    ):
        raise ContractError("Expected repo in owner/name format")
    return repo


def bounded_read(url, *, max_bytes, opener=None):
    if not isinstance(max_bytes, int) or not (1024 <= max_bytes <= 2 * 1024 * 1024):
        raise ContractError("File limit must be 1KiB–2MiB")
    opener = opener or urllib.request.urlopen
    req = urllib.request.Request(url, headers={"User-Agent": "VLA-metadata-audit/0.1"})
    try:
        with opener(req, timeout=15) as resp:
            declared = resp.headers.get("Content-Length")
            if declared and int(declared) > max_bytes:
                raise ContractError("Metadata exceeds declared size cap")
            blob = resp.read(max_bytes + 1)
    except urllib.error.URLError as exc:
        raise ContractError("Metadata request failed: " + str(exc)) from exc
    if len(blob) > max_bytes:
        raise ContractError("Metadata exceeded streamed size cap")
    return blob


def resolve_revision(repo, *, allow_network=False, opener=None):
    repo = check_repo(repo)
    if not allow_network:
        raise ContractError("Explicit --allow-network is required")
    data = bounded_read("https://huggingface.co/api/datasets/" + repo,
                        max_bytes=2 * 1024 * 1024, opener=opener)
    try:
        sha = json.loads(data)["sha"]
    except (KeyError, ValueError, TypeError) as exc:
        raise ContractError("HF API did not return dataset SHA") from exc
    if not isinstance(sha, str) or not SHA.fullmatch(sha):
        raise ContractError("Dataset revision SHA must be 40 hexadecimal characters")
    return {"repo_id": repo, "revision": sha.lower(), "downloaded_files": 0,
            "note": "Pin this SHA in your experiment; HEAD may change."}


def fetch_metadata(repo, revision, filename, dest_root, *,
                   allow_network=False, max_bytes=512*1024, opener=None):
    check_repo(repo)
    if not allow_network:
        raise ContractError("Explicit --allow-network is required")
    if not isinstance(revision, str) or not SHA.fullmatch(revision):
        raise ContractError("Exact 40-character dataset revision is required")
    if filename not in ALLOWED:
        raise ContractError("Only explicitly whitelisted metadata files allowed")
    dest = Path(dest_root) / repo.replace("/", "--") / revision.lower() / filename
    if dest.exists() or dest.is_symlink():
        raise ContractError("Refuse to overwrite " + str(dest))
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}"
    blob = bounded_read(url, max_bytes=max_bytes, opener=opener)
    if filename.endswith(".json") or filename.endswith(".jsonl"):
        try:
            for line in (blob.splitlines() if filename.endswith(".jsonl") else [blob]):
                if line.strip():
                    json.loads(line)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ContractError("Downloaded metadata is not valid JSON") from exc
    if filename.endswith(".parquet") and (blob[:4] != b"PAR1" or blob[-4:] != b"PAR1"):
        raise ContractError("Downloaded metadata is not a complete Parquet file")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("xb") as f:
        f.write(blob)
    return {"status": "DOWNLOADED_SINGLE_METADATA", "path": str(dest),
            "bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
            "revision": revision.lower(), "training_data_downloaded": False}


def inspect_layout(root, *, sample_rows=0):
    """Default only reads info.json and lists index files. Sampling needs optional pyarrow."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ContractError("Dataset root absent: " + str(root))
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise ContractError("Missing meta/info.json at " + str(root))
    info = audit_lerobot_info(info_path)
    meta = root / "meta"
    v3_tasks = (meta / "tasks.parquet").is_file()
    v2_tasks = (meta / "tasks.jsonl").is_file()
    v2_episodes = (meta / "episodes.jsonl").is_file()
    shards = sorted((meta / "episodes").glob("**/*.parquet")) if (meta/"episodes").is_dir() else []
    result = {"meta": info, "tasks_index": "tasks.parquet" if v3_tasks else
              ("tasks.jsonl" if v2_tasks else None),
              "episodes_index": "episodes/*.parquet" if shards else
              ("episodes.jsonl" if v2_episodes else None),
              "episode_shard_count": len(shards), "sample": [],
              "sample_status": "NOT_REQUESTED", "source_group_verified": False,
              "safe_to_train": False,
              "note": "Task text/index cannot establish camera changes, source demo, or recovery scene."}
    if not isinstance(sample_rows, int) or not 0 <= sample_rows <= 10:
        raise ContractError("sample_rows must be in 0..10")
    if sample_rows == 0:
        return result
    if not shards:
        result["sample_status"] = "NO_LOCAL_V3_EPISODE_SHARD"
        return result
    try:
        import pyarrow.parquet as pq
    except ImportError:
        result["sample_status"] = "PYARROW_OPTIONAL_NOT_INSTALLED"
        return result
    pf = pq.ParquetFile(shards[0])
    columns = [x for x in ("episode_index", "length", "tasks", "task_index")
               if x in pf.schema_arrow.names]
    if not columns:
        result["sample_status"] = "NO_RECOGNIZED_EPISODE_COLUMNS"
        return result
    rows = next(pf.iter_batches(batch_size=sample_rows, columns=columns), None)
    result["sample"] = rows.to_pylist() if rows is not None else []
    result["sample_status"] = "READ_UP_TO_10_ROWS_OF_METADATA_ONLY"
    return result
