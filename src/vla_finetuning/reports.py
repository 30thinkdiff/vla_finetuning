"""Strict paired-success result audit; no invented rollouts or statistics claims."""
from __future__ import annotations
import csv
import json
from collections import defaultdict
from pathlib import Path
from .core import ContractError, write_new_json
from .reference import paired_outcomes


def load_eval_rows(csv_file):
    with Path(csv_file).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"domain", "condition_id", "seed", "method", "success"}
        if not required.issubset(set(reader.fieldnames or ())):
            raise ContractError("Missing evaluation CSV columns")
        rows = []
        for raw in reader:
            if raw["success"] not in ("true", "false", "True", "False", "1", "0"):
                raise ContractError("success must be explicit boolean, not unparsed text")
            try:
                seed = int(raw["seed"])
            except (ValueError, TypeError) as exc:
                raise ContractError("Invalid evaluation seed") from exc
            rows.append({"domain": raw["domain"].strip(),
                         "condition_id": raw["condition_id"].strip(),
                         "seed": seed,
                         "method": raw["method"].strip(),
                         "success": raw["success"].lower() in ("true", "1")})
    return rows


def analyze_results(csv_file, baseline="B0", candidate="L1"):
    rows = load_eval_rows(csv_file)
    try:
        overall = paired_outcomes(rows, baseline, candidate)
    except ValueError as exc:
        raise ContractError("Unmatched/duplicate paired results: " + str(exc)) from exc
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["domain"]].append(row)
    domains = {}
    for domain, block in sorted(grouped.items()):
        values = paired_outcomes(block, baseline, candidate)
        b_success = values["both_success"] + values["lost"]
        c_success = values["both_success"] + values["gained"]
        domains[domain] = {**values,
                           "baseline_success": b_success, "candidate_success": c_success,
                           "baseline_rate": b_success / values["n_pairs"],
                           "candidate_rate": c_success / values["n_pairs"],
                           "delta_percentage_points": 100*(c_success-b_success)/values["n_pairs"]}
    return {"status": "DESCRIPTIVE_MATCHED_ROLLOUT_AUDIT", "input_csv": str(Path(csv_file).resolve()),
            "baseline": baseline, "candidate": candidate, "overall": overall,
            "domains": domains,
            "warnings": ["This does not establish statistical significance.",
                         "CSV integrity does not establish policy, simulator, camera or task provenance.",
                         "Always verify videos and environment traces for reported rollouts."]}
