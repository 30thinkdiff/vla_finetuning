"""Production-quality reference solutions for the three student exercises.

These are independent reference implementations and must NOT be silently
substituted for official LeRobot preprocessing or simulator success metrics.
"""
from __future__ import annotations
import math


def extract_task_labels(episode_record):
    """Return ordered unique task strings, or explicit legacy task-index markers.

    A dataset task index is not the benchmark's perturbation/variant task ID.
    """
    if not isinstance(episode_record, dict):
        raise ValueError("Episode metadata must be a dict")
    if "tasks" in episode_record and episode_record["tasks"] is not None:
        raw = episode_record["tasks"]
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            raise ValueError("tasks must be a list of strings or one string")
        labels = []
        for value in raw:
            if not isinstance(value, str):
                raise ValueError("Task label is not a string")
            label = value.strip()
            if label and label not in labels:
                labels.append(label)
        if labels:
            return labels
    idx = episode_record.get("task_index")
    if type(idx) is int and idx >= 0:
        return [f"task_index:{idx}"]
    raise ValueError("No usable task labels or task_index in Episode metadata")


def make_action_chunk(episode_actions, start, horizon):
    """Make [H,7] action chunk and True=padding time mask for a single episode.

    Tail padding copies the last valid action (rather than zeroing gripper state).
    The training loss must mask the padded positions.
    """
    if not isinstance(episode_actions, (list, tuple)) or not episode_actions:
        raise ValueError("Episode actions must be a nonempty list")
    if type(start) is not int or type(horizon) is not int or horizon <= 0:
        raise ValueError("start must be int and horizon positive int")
    if not 0 <= start < len(episode_actions):
        raise ValueError("start index outside this Episode")
    validated = []
    for row in episode_actions:
        if not isinstance(row, (list, tuple)) or len(row) != 7:
            raise ValueError("Each action must contain exactly seven scalars")
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in row):
            raise ValueError("Action contains nonfinite/non-numeric values")
        validated.append([float(x) for x in row])
    chunk = [row.copy() for row in validated[start:start+horizon]]
    real = len(chunk)
    last = chunk[-1].copy()
    while len(chunk) < horizon:
        chunk.append(last.copy())
    mask = [False] * real + [True] * (horizon - real)
    return chunk, mask


def paired_outcomes(rows, baseline="B0", candidate="L1"):
    """Paired success/decline counts from identical domain,condition_id,seed keys.

    Denominators and identity keys are checked; not statistical significance.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError("Rows must be nonempty list")
    if not isinstance(baseline, str) or not isinstance(candidate, str) or baseline == candidate:
        raise ValueError("Distinct comparison methods required")
    by_method = {baseline: {}, candidate: {}}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each row must be a dictionary")
        method = row.get("method")
        if method not in by_method:
            raise ValueError("Unexpected evaluation method")
        if type(row.get("success")) is not bool:
            raise ValueError("success must be Python bool")
        if any(not isinstance(row.get(k), str) or not row[k] for k in ("domain", "condition_id")):
            raise ValueError("Missing domain/condition_id")
        if type(row.get("seed")) is not int:
            raise ValueError("Seed must be integer")
        key = (row["domain"], row["condition_id"], row["seed"])
        if key in by_method[method]:
            raise ValueError("Duplicate method/condition/seed result")
        by_method[method][key] = row["success"]
    b, c = by_method[baseline], by_method[candidate]
    if not b or set(b) != set(c):
        raise ValueError("Missing or unmatched evaluation cases")
    gained = sum(not b[k] and c[k] for k in b)
    lost = sum(b[k] and not c[k] for k in b)
    both_success = sum(b[k] and c[k] for k in b)
    both_failed = sum(not b[k] and not c[k] for k in b)
    assert gained + lost + both_success + both_failed == len(b)
    return {"n_pairs": len(b), "gained": gained, "lost": lost,
            "both_success": both_success, "both_failed": both_failed}
