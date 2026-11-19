"""Build deterministic *selection* of released failure scenes, without simulator.

Sampling seed is NOT a proven equivalent of the original benchmark launcher
RNG nor an already observed physical initial state.
"""
from __future__ import annotations

import hashlib
import random
from collections import defaultdict

from .core import ContractError
from .recovery_contract import SUITES, audit_scene_index, read_jsonl


def plan_recovery_cases(index_file, *, methods, seed=42, per_suite_level=2,
                        open_trials=1, close_trials=1, suites=None):
    audit = audit_scene_index(index_file)
    if not isinstance(methods, list) or len(methods) < 2 or len(methods) > 5 or not all(
            isinstance(x, str) and x.strip() for x in methods) or len(set(methods)) != len(methods):
        raise ContractError("Provide 2-5 distinct method identifiers")
    for name, value, low, high in (("per_suite_level", per_suite_level, 1, 20),
                                   ("open_trials", open_trials, 0, 20),
                                   ("close_trials", close_trials, 0, 20)):
        if type(value) is not int or not low <= value <= high:
            raise ContractError(f"Invalid {name}")
    if open_trials + close_trials < 1 or type(seed) is not int:
        raise ContractError("At least one trial and integer seed required")
    if suites is None:
        suites = sorted(SUITES)
    if not isinstance(suites, list) or not suites or any(s not in SUITES for s in suites):
        raise ContractError("Invalid suite subset")
    by_bucket = defaultdict(list)
    for row in read_jsonl(index_file, max_records=20000):
        if row["source"] in suites:
            by_bucket[(row["source"], row["level"])].append(row)
    selected = []
    for bucket, records in sorted(by_bucket.items()):
        digest = hashlib.sha256(f"{seed}/{bucket[0]}/{bucket[1]}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        entries = sorted(records, key=lambda r: r["scene_id"])
        rng.shuffle(entries)
        selected.extend(entries[:per_suite_level])
    if not selected:
        raise ContractError("No scenes matched requested suite(s)")
    planned = []
    for scene in sorted(selected, key=lambda s: (s["source"], s["level"], s["scene_id"])):
        for grip, count in (("open", open_trials), ("close", close_trials)):
            for repeat in range(count):
                planned.append({"suite": scene["source"], "level": scene["level"],
                                "scene_id": scene["scene_id"], "gripper_init": grip,
                                "repeat_index": repeat, "master_seed": seed,
                                "perturbed_state_sha256": None,
                                "source_group": None})
    return {"status": "PLAN_ONLY_UNRESOLVED_SCENE_ASSETS", "index_sha256": audit["sha256"],
            "methods": methods, "sampling_seed": seed,
            "selection_strategy": "bucketed_suite_level_deterministic_NOT_official_sampler",
            "scenes": len(selected), "trial_conditions": len(planned),
            "expected_policy_rollouts": len(planned)*len(methods),
            "trials": planned, "simulator_loaded": False, "model_loaded": False,
            "mandatory_before_eval": ["Resolve BDDL and sibling sim_state.npy in exact pinned scene release",
                                      "Use official evaluator's seeded perturbation+settle protocol",
                                      "Capture *actual* post-perturb initial-state hashes for method pairs",
                                      "Verify model processors, camera transforms, control and action semantics",
                                      "Keep official test scenes disjoint from Expert training source groups"]}
