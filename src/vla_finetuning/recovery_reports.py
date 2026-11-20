"""Strict paired LIBERO-Recovery accounting with source-group bootstrap.

Descriptive from supplied records only. Does not authenticate videos or
assert benchmark/generalization significance on its own.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from .core import ContractError
from .recovery_contract import LEVELS, SUITES

REQUIRED = {"suite", "level", "scene_id", "source_group", "gripper_init",
            "seed", "method", "success", "initial_state_sha256",
            "protocol_sha256", "rollout_evidence"}


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _parse(path):
    with Path(path).open(encoding="utf-8", newline="") as fp:
        rd = csv.DictReader(fp)
        if not REQUIRED.issubset(rd.fieldnames or []):
            raise ContractError("Missing recovery report columns: " + repr(sorted(REQUIRED-set(rd.fieldnames or []))))
        rows = list(rd)
    if not rows:
        raise ContractError("Empty recovery eval CSV")
    seen = set()
    for row in rows:
        if row["suite"] not in SUITES or row["level"] not in LEVELS or row["gripper_init"] not in ("open", "close"):
            raise ContractError("Invalid recovery suite, level or gripper_init")
        for field in ("scene_id", "source_group", "method", "rollout_evidence"):
            if not row[field].strip():
                raise ContractError("Missing " + field)
        if "SYNTHETIC" in json.dumps(row).upper():
            raise ContractError("Synthetic rows cannot become formal recovery results")
        if row["success"] not in ("true", "false", "1", "0"):
            raise ContractError("Explicit success boolean required")
        try:
            row["seed"] = int(row["seed"])
        except ValueError as exc:
            raise ContractError("Invalid seed") from exc
        for field in ("initial_state_sha256", "protocol_sha256"):
            value = row[field]
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ContractError("Invalid lowercase SHA256: " + field)
        row["success"] = row["success"] in ("true", "1")
        key = (row["scene_id"], row["seed"], row["gripper_init"], row["method"])
        if key in seen:
            raise ContractError("Duplicate method on a scene/seed/gripper case")
        seen.add(key)
    return rows


def analyze_recovery_csv(path, *, baseline="M0", candidate="M2", bootstrap=1000, seed=42):
    if not baseline or not candidate or baseline == candidate:
        raise ContractError("Two distinct method names required")
    if type(bootstrap) is not int or not 100 <= bootstrap <= 10000:
        raise ContractError("Bootstrap repetitions must be between 100 and 10000")
    rows = _parse(path)
    pairs = defaultdict(dict)
    scene_info = {}
    for row in rows:
        if row["method"] not in (baseline, candidate):
            raise ContractError("Unexpected method: " + row["method"])
        scene = row["scene_id"]
        fixed = (row["suite"], row["level"], row["source_group"])
        if scene in scene_info and scene_info[scene] != fixed:
            raise ContractError("Inconsistent metadata for scene_id: " + scene)
        scene_info[scene] = fixed
        key = (scene, row["seed"], row["gripper_init"])
        pairs[key][row["method"]] = row
    grouped = defaultdict(list)
    by_level = defaultdict(list)
    by_gripper = defaultdict(list)
    table = CounterDict()
    for key, methods in pairs.items():
        if set(methods) != {baseline, candidate}:
            raise ContractError("Unmatched paired methods: " + repr(key))
        b, c = methods[baseline], methods[candidate]
        for field in ("suite", "level", "source_group", "initial_state_sha256", "protocol_sha256"):
            if b[field] != c[field]:
                raise ContractError("Paired protocol mismatch at " + field)
        k = (b["success"], c["success"])
        table.add(k)
        item = (int(c["success"]) - int(b["success"]))
        grouped[b["source_group"]].append(item)
        by_level[b["level"]].append((b["success"], c["success"]))
        by_gripper[b["gripper_init"]].append((b["success"], c["success"]))
    differences = [v for values in grouped.values() for v in values]
    n = len(differences)
    groups = sorted(grouped)
    rng = random.Random(seed)
    sampled = []
    for _ in range(bootstrap):
        picked = [rng.choice(groups) for _ in groups]
        values = [v for g in picked for v in grouped[g]]
        sampled.append(100 * sum(values) / len(values))
    sampled.sort()
    low = sampled[int(0.025 * bootstrap)]
    high = sampled[min(bootstrap-1, int(0.975 * bootstrap))]
    def stratified(rows):
        return {k: {"n": len(p), "baseline_success": sum(int(b) for b, c in p),
                    "candidate_success": sum(int(c) for b, c in p),
                    "delta_pp": round(100 * sum(int(c)-int(b) for b, c in p) / len(p), 3)}
                for k, p in sorted(rows.items())}
    return {"status": "PAIRED_RECOVERY_CSV_DESCRIPTIVE_ONLY",
            "csv_sha256": _digest(path), "baseline": baseline, "candidate": candidate,
            "pairs": n, "independent_source_groups": len(groups),
            "baseline_success": table.tt + table.tf,
            "candidate_success": table.tt + table.ft,
            "both_success": table.tt, "candidate_only_success": table.ft,
            "baseline_only_success": table.tf, "both_failed": table.ff,
            "delta_percentage_points": round(100*sum(differences)/n, 3),
            "source_group_cluster_bootstrap_95pct_pp": [round(low, 3), round(high, 3)],
            "bootstrap_seed": seed, "bootstrap_replicates": bootstrap,
            "by_level": stratified(by_level), "by_gripper_init": stratified(by_gripper),
            "critical_warnings": ["CSV self-consistency does not authenticate rollouts, camera/action contract, or videos",
                                  "Source groups require external provenance audit",
                                  "Bootstrap is descriptive; repeated runs on one scene are not independent scenes"]}


class CounterDict:
    def __init__(self):
        self.tt = self.tf = self.ft = self.ff = 0

    def add(self, pair):
        b, c = pair
        if b and c:
            self.tt += 1
        elif b:
            self.tf += 1
        elif c:
            self.ft += 1
        else:
            self.ff += 1
