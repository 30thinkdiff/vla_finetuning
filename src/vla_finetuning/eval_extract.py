"""Turn official LeRobot eval_info.json into paired outcome CSV.

Uses eval_policy_all per_task/task_group/task_id/metrics.successes schema.
Does not verify matching camera, physical initial states or checkpoints.
"""
from __future__ import annotations
import csv
from pathlib import Path
from .core import ContractError, read_json


def eval_rows(info_file, method, start_seed):
    if type(start_seed) is not int:
        raise ContractError("start_seed must be integer")
    obj = read_json(info_file)
    if not isinstance(obj, dict) or not isinstance(obj.get("per_task"), list):
        raise ContractError("Expected official eval_info.json per_task list")
    seen, rows = set(), []
    for task in obj["per_task"]:
        suite, variant = task.get("task_group"), task.get("task_id")
        successes = task.get("metrics", {}).get("successes")
        if not isinstance(suite, str) or not suite or type(variant) is not int:
            raise ContractError("Missing registered suite/variant identity")
        if not isinstance(successes, list) or not successes:
            raise ContractError("Need per-episode successes, not aggregate rate")
        for i, success in enumerate(successes):
            if type(success) is not bool:
                raise ContractError("Each success must be bool")
            key = (suite, variant, start_seed+i)
            if key in seen:
                raise ContractError("Duplicate evaluation condition/seed")
            seen.add(key)
            rows.append({"domain": suite, "condition_id": f"{suite}:variant_{variant}",
                         "seed": start_seed+i, "method": method, "success": success})
    if not rows:
        raise ContractError("No evaluation cases")
    return rows


def merge_official_eval(baseline_info, candidate_info, *, baseline="B0",
                        candidate="L1", start_seed=1000):
    if baseline == candidate:
        raise ContractError("Different methods required")
    b = eval_rows(baseline_info, baseline, start_seed)
    c = eval_rows(candidate_info, candidate, start_seed)
    keys = lambda records: {(x["domain"], x["condition_id"], x["seed"]) for x in records}
    if keys(b) != keys(c):
        raise ContractError("Baseline/candidate registered task and seed sets mismatch")
    return sorted(b+c, key=lambda x: (x["domain"], x["condition_id"], x["seed"], x["method"]))


def save_csv_new(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=["domain", "condition_id", "seed", "method", "success"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "success": str(row["success"]).lower()})
    return {"status": "PAIRED_CSV_FROM_OFFICIAL_EVAL_INFO",
            "path": str(path.resolve()), "rows": len(rows),
            "warning": "Check physical initial states, cameras and checkpoint identities separately."}
