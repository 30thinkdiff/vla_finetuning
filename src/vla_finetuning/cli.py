"""CPU-first CLI. Importing this module does not touch Torch/CUDA/network."""
import argparse
import json
from pathlib import Path

from .core import (ContractError, audit_lerobot_info, guarded_launch, preflight,
                   split_provenance_csv, training_plan, write_new_json)
from .flow_math import two_number_example


def print_json(data):
    print(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description="SmolVLA research: provenance first, training explicitly gated")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight", help="Read-only disk and Python snapshot")
    p.add_argument("--data-root", default="/root/autodl-tmp")

    p = sub.add_parser("audit-metadata", help="Read local LeRobot meta/info.json (no download)")
    p.add_argument("--info", type=Path, required=True)
    p.add_argument("--out", type=Path)

    p = sub.add_parser("split", help="Group-disjoint split from curated provenance CSV")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.2)

    sub.add_parser("flow-demo", help="Pure-Python two-number Flow Matching example")
    sub.add_parser("lora-demo", help="Two CPU PyTorch LoRA optimization steps; no CUDA")

    p = sub.add_parser("plan-train", help="Validate config; print official CLI without execution")
    p.add_argument("--config", type=Path, default=Path("configs/experiment.template.json"))
    p.add_argument("--out", type=Path)

    p = sub.add_parser("launch-train", help="Opt-in only, refuses if GPU compute process exists")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--i-confirm-corrector-finished", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            result = preflight(args.data_root)
        elif args.command == "audit-metadata":
            result = audit_lerobot_info(args.info)
        elif args.command == "split":
            result = split_provenance_csv(args.csv, args.seed, args.train_frac, args.val_frac)
        elif args.command == "flow-demo":
            result = two_number_example()
        elif args.command == "lora-demo":
            from .lora_lab import two_step_lora_demo
            result = two_step_lora_demo()
        elif args.command == "plan-train":
            result = training_plan(args.config)
        elif args.command == "launch-train":
            guarded_launch(args.config, args.i_confirm_corrector_finished)
            result = {"status": "TRAIN_PROCESS_RETURNED_ZERO", "note": "Inspect checkpoint and logs separately."}
        else:
            raise AssertionError("unreachable")
        if getattr(args, "out", None) is not None:
            write_new_json(args.out, result)
        print_json(result)
    except (ContractError, FileExistsError, OSError, ValueError, KeyError) as exc:
        parser.exit(2, "STOP: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
