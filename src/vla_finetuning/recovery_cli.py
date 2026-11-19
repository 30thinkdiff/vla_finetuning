"""Offline-only Recovery preparation. All commands are read-only unless --out (new JSON)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import ContractError, write_new_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="operation", required=True)
    def inp(name):
        s = sub.add_parser(name)
        s.add_argument("--out", type=Path, help="Create new JSON only; never overwrite")
        return s
    a = inp("audit-expert")
    a.add_argument("--info", type=Path, required=True)
    a = inp("audit-scenes")
    a.add_argument("--index", type=Path, required=True)
    a = inp("audit-linkage")
    a.add_argument("--jsonl", type=Path, required=True)
    a = inp("audit-split")
    a.add_argument("--split", type=Path, required=True)
    a = inp("plan-mixture")
    a.add_argument("--split", type=Path, required=True)
    a.add_argument("--method", choices=["M0", "M1", "M2", "M3"], required=True)
    a.add_argument("--steps", type=int, required=True)
    a.add_argument("--batch-size", type=int, default=1)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--recovery-fraction", type=float)
    a = inp("plan-v21-files")
    a.add_argument("--dataset-root", type=Path, required=True)
    a.add_argument("--episodes", required=True, help="Comma-separated nonnegative Episode indices")
    a = inp("plan-materialization")
    a.add_argument("--split", type=Path, required=True)
    a.add_argument("--alignment", type=Path, required=True)
    a = inp("plan-eval")
    a.add_argument("--index", type=Path, required=True)
    a.add_argument("--methods", default="M0,M1,M2", help="Comma-separated identifiers")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--per-suite-level", type=int, default=2)
    a.add_argument("--open-trials", type=int, default=1)
    a.add_argument("--close-trials", type=int, default=1)
    a.add_argument("--suites", help="Comma-separated suite subset; default all five")
    a = inp("report")
    a.add_argument("--csv", type=Path, required=True)
    a.add_argument("--baseline", default="M0")
    a.add_argument("--candidate", default="M2")
    a.add_argument("--bootstrap", type=int, default=1000)
    a.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    try:
        if args.operation == "audit-expert":
            from .recovery_contract import audit_expert_info
            result = audit_expert_info(args.info)
        elif args.operation == "audit-scenes":
            from .recovery_contract import audit_scene_index
            result = audit_scene_index(args.index)
        elif args.operation == "audit-linkage":
            from .recovery_contract import audit_manual_linkage
            result = audit_manual_linkage(args.jsonl)
        elif args.operation == "audit-split":
            from .recovery_experiments import validate_split
            result = validate_split(args.split)
        elif args.operation == "plan-mixture":
            from .recovery_experiments import plan_mixture
            result = plan_mixture(args.split, method=args.method, steps=args.steps,
                                  batch_size=args.batch_size, seed=args.seed,
                                  recovery_fraction=args.recovery_fraction)
        elif args.operation == "plan-v21-files":
            from .recovery_v21 import plan_v21_expert_files
            try:
                indices = [int(x) for x in args.episodes.split(",")]
            except ValueError as exc:
                raise ContractError("Expected comma-separated integer Episode IDs") from exc
            result = plan_v21_expert_files(args.dataset_root, indices)
        elif args.operation == "plan-materialization":
            from .recovery_merge_plan import plan_mixed_v3_dataset
            result = plan_mixed_v3_dataset(args.split, args.alignment)
        elif args.operation == "plan-eval":
            from .recovery_eval_plan import plan_recovery_cases
            result = plan_recovery_cases(args.index, methods=args.methods.split(","),
                                         seed=args.seed, per_suite_level=args.per_suite_level,
                                         open_trials=args.open_trials, close_trials=args.close_trials,
                                         suites=args.suites.split(",") if args.suites else None)
        elif args.operation == "report":
            from .recovery_reports import analyze_recovery_csv
            result = analyze_recovery_csv(args.csv, baseline=args.baseline,
                                           candidate=args.candidate, bootstrap=args.bootstrap,
                                           seed=args.seed)
        else:
            raise AssertionError("unreachable")
        if args.out:
            write_new_json(args.out, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ContractError, OSError, FileExistsError, ValueError, TypeError, KeyError) as exc:
        p.exit(2, "STOP: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
