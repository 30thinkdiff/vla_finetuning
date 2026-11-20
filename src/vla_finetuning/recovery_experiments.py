"""CPU-only frozen Recovery experiment contracts; no trainer or GPU import."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .core import ContractError, read_json

EXPERT = "ataier/LIBERO_Recovery_Expert"
NORMAL = "lerobot/libero"


def validate_split(path):
    """Validate manually reviewed *global* source-group split, not random episodes."""
    obj = read_json(path)
    if not isinstance(obj, dict) or obj.get("kind") != "recovery_source_split_v1":
        raise ContractError("Expected recovery_source_split_v1")
    if obj.get("real_evidence_reviewed") is not True:
        raise ContractError("Split requires explicit human provenance review")
    datasets = obj.get("datasets")
    if not isinstance(datasets, dict) or EXPERT not in datasets or NORMAL not in datasets:
        raise ContractError("Freeze Expert and normal-data versions separately")
    for source in (EXPERT, NORMAL):
        v = datasets[source]
        if not isinstance(v, dict) or not isinstance(v.get("revision"), str) or not v["revision"].strip() or v["revision"] in ("main", "master", "latest"):
            raise ContractError("Missing explicit reviewed dataset revision for " + source)
        if v.get("source_identity_checked") is not True:
            raise ContractError("Dataset origin not verified: " + source)
    splits = obj.get("splits")
    if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
        raise ContractError("Exactly train/validation/test required")
    seen_keys, group_locations = set(), {}
    counts = Counter()
    for split_name, rows in splits.items():
        if not isinstance(rows, list) or not rows:
            raise ContractError("Empty split: " + split_name)
        for row in rows:
            if not isinstance(row, dict):
                raise ContractError("Invalid row in split")
            dataset, episode, group = row.get("dataset_id"), row.get("episode_id"), row.get("source_group")
            if dataset not in (EXPERT, NORMAL) or type(episode) is not int or episode < 0:
                raise ContractError("Invalid dataset or Episode identity")
            if not isinstance(group, str) or not group.strip():
                raise ContractError("Unknown source group")
            if row.get("review_status") != "verified" or not isinstance(row.get("source_evidence"), str) or not row["source_evidence"].strip():
                raise ContractError("Unverified source evidence")
            if "SYNTHETIC" in json.dumps(row).upper():
                raise ContractError("Synthetic source cannot be used as real train/test")
            key = (dataset, episode)
            if key in seen_keys:
                raise ContractError("Duplicate episode across splits: " + repr(key))
            seen_keys.add(key)
            if group in group_locations and group_locations[group] != split_name:
                raise ContractError("Source-group leakage: " + group)
            group_locations[group] = split_name
            counts[f"{split_name}:{dataset}"] += 1
    if counts[f"train:{EXPERT}"] < 1 or counts[f"train:{NORMAL}"] < 1:
        raise ContractError("Both Expert and normal training samples required")
    bench_groups = obj.get("reserved_benchmark_source_groups")
    if not isinstance(bench_groups, list) or not bench_groups or any(not isinstance(g, str) or not g.strip() for g in bench_groups):
        raise ContractError("Explicit official recovery benchmark holdout groups required")
    if len(set(bench_groups)) != len(bench_groups):
        raise ContractError("Duplicate reserved benchmark group")
    if set(bench_groups) & {r["source_group"] for r in splits["train"] + splits["validation"]}:
        raise ContractError("Official benchmark source in train/validation")
    return {"status": "REVIEWED_SPLIT_SCHEMA_ONLY", "groups": len(group_locations),
            "counts": dict(sorted(counts.items())), "reserved_benchmark_groups": len(bench_groups),
            "split_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "warning": "Review flags and evidence strings need external independent verification"}


def plan_mixture(split_path, *, method, steps, batch_size, seed, recovery_fraction=None):
    """Plan episode-ID draws ONLY; not a real LeRobot batch or gradient step.

    Sampler is episode-uniform, not frame-uniform. Production DataLoader still
    needs separately verified frame/time alignment and normalization.
    """
    info = validate_split(split_path)
    if method not in ("M0", "M1", "M2"):
        raise ContractError("M3 requires validated algorithm and must not be silently implemented")
    if type(steps) is not int or steps < 0 or type(batch_size) is not int or batch_size < 1 or type(seed) is not int:
        raise ContractError("Invalid steps/batch_size/seed")
    if method == "M0":
        if steps != 0:
            raise ContractError("M0 is unmodified baseline: 0 training steps")
        return {"status": "BASELINE_NO_TRAINING", "method": "M0", "steps": 0,
                "split_sha256": info["split_sha256"], "draws": []}
    if steps < 1 or steps * batch_size > 100000:
        raise ContractError("Explicit bounded total draws 1..100000 required")
    if method == "M1":
        if recovery_fraction not in (None, 1, 1.0):
            raise ContractError("M1 uses 100% recovery episodes")
        fraction = 1.0
    else:
        if type(recovery_fraction) not in (int, float) or not 0 < recovery_fraction < 1:
            raise ContractError("M2 recovery_fraction must be strictly between 0 and 1")
        fraction = float(recovery_fraction)
    train = read_json(split_path)["splits"]["train"]
    pools = {source: sorted((r["episode_id"] for r in train if r["dataset_id"] == source)) for source in (EXPERT, NORMAL)}
    if not all(pools.values()):
        raise ContractError("Empty training pool")
    n = steps * batch_size
    rng = random.Random(seed)
    recovery_count = n if method == "M1" else round(n * fraction)
    if method == "M2" and not 0 < recovery_count < n:
        raise ContractError("Batch budget too small to represent both data sources")
    sources = [EXPERT] * recovery_count + [NORMAL] * (n - recovery_count)
    rng.shuffle(sources)
    draws = [{"dataset_id": s, "episode_id": rng.choice(pools[s])} for s in sources]
    return {"status": "SCHEDULE_ONLY_NOT_LOADED_NOT_TRAINED", "method": method,
            "unit": "episode_uniform_draw_NOT_frame_uniform", "steps": steps,
            "batch_size": batch_size, "seed": seed, "requested_recovery_fraction": fraction,
            "realized_recovery_draws": recovery_count, "total_draws": n,
            "split_sha256": info["split_sha256"], "draws": draws,
            "blocking_notes": ["Episode sampling is not a verified production DataLoader",
                               "Freeze action/normalization, LoRA targets and evaluation protocol",
                               "No claim of fairness without equal optimizer steps, tokens and GPU cost"]}
