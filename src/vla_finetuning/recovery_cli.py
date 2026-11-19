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
