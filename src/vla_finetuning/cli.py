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

    p = sub.add_parser("inspect-data-layout", help="Read local v2/v3 metadata, no GPU or data downloads")
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--sample-rows", type=int, default=0)
    p.add_argument("--out", type=Path)

    p = sub.add_parser("hf-pin", help="Optional small HF API request: dataset revision SHA")
    p.add_argument("--repo", required=True)
    p.add_argument("--allow-network", action="store_true")

    p = sub.add_parser("fetch-metadata", help="Download one small whitelisted metadata file only")
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--file", required=True, choices=["meta/info.json", "meta/tasks.parquet",
                                                     "meta/tasks.jsonl", "meta/episodes.jsonl", "meta/stats.json"])
    p.add_argument("--output-root", type=Path, default=Path("datasets/metadata"))
    p.add_argument("--allow-network", action="store_true")

    p = sub.add_parser("inspect-model", help="Inspect local model config and weight filenames only; never load model")
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--out", type=Path)

    p = sub.add_parser("validate-eval-plan", help="Validate paired-rollout manifest without loading simulator")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--out", type=Path)

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
        elif args.command == "inspect-data-layout":
            from .metadata import inspect_layout
            result = inspect_layout(args.dataset_root, sample_rows=args.sample_rows)
        elif args.command == "hf-pin":
            from .metadata import resolve_revision
            result = resolve_revision(args.repo, allow_network=args.allow_network)
        elif args.command == "fetch-metadata":
            from .metadata import fetch_metadata
            result = fetch_metadata(args.repo, args.revision, args.file,
                                    args.output_root, allow_network=args.allow_network)
        elif args.command == "inspect-model":
            from .protocols import inspect_model_snapshot
            result = inspect_model_snapshot(args.model_root)
        elif args.command == "validate-eval-plan":
            from .protocols import validate_eval_plan
            result = validate_eval_plan(args.plan)
        elif args.command == "flow-demo":
            result = two_number_example()
        elif args.command == "lora-demo":
            from .lora_lab import two_step_lora_demo
            try:
                result = two_step_lora_demo()
            except ModuleNotFoundError as exc:
                if exc.name == "torch":
                    raise ContractError("Optional PyTorch not installed in this Python. No need to modify the Corrector environment; use a dedicated training venv later.") from exc
                raise
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
