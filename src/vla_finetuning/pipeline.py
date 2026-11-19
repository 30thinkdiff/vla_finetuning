"""Operational entrypoint: all expensive actions are preview-first and opt-in.

Usage: PYTHONPATH=src python -m vla_finetuning.pipeline <command> --help
The default commands never run training, GPU work, downloads or simulations.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .core import ContractError, read_json, write_new_json
from .guards import load_protocol, shell_render


def _display(result, out=None):
    if out:
        write_new_json(out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="operation", required=True)
    def add_output(x):
        x.add_argument("--out", type=Path)

    s = sub.add_parser("validate-config", help="Static protocol audit (no model or data load)")
    s.add_argument("--config", type=Path, required=True)
    add_output(s)

    s = sub.add_parser("shard-plan", help="Resolve exact dataset v3 shards from pinned local metadata")
    s.add_argument("--metadata-root", type=Path, required=True)
    s.add_argument("--split", type=Path, required=True)
    s.add_argument("--repo-id", default="lerobot/libero_plus")
    add_output(s)

    s = sub.add_parser("hub-plan", help="Explicit Hub metadata lookup; produce download budget")
    s.add_argument("--repo", required=True)
    s.add_argument("--revision", required=True)
    s.add_argument("--kind", choices=["dataset_metadata", "dataset_payload", "model"], required=True)
    s.add_argument("--shard-plan", type=Path)
    s.add_argument("--allow-network", action="store_true")
    add_output(s)

    s = sub.add_parser("hub-download", help="MANUAL download exact budgeted files after GPU idle")
    s.add_argument("--plan", type=Path, required=True)
    s.add_argument("--target", type=Path, required=True)
    s.add_argument("--project-root", type=Path, required=True)
    s.add_argument("--max-gib", type=float, required=True)
    s.add_argument("--apply", action="store_true")
    add_output(s)

    s = sub.add_parser("train", help="Review official LeRobot CLI or opt-in launch on idle GPU")
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--binary", default="lerobot-train")
    s.add_argument("--apply", action="store_true")
    s.add_argument("--confirm-gpu-idle", action="store_true")
    add_output(s)

    s = sub.add_parser("evaluate", help="Review official LeRobot LIBERO-plus eval or opt-in")
    s.add_argument("--config", type=Path, required=True)
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument("--output-dir", type=Path, required=True)
    s.add_argument("--binary", default="lerobot-eval")
    s.add_argument("--apply", action="store_true")
    s.add_argument("--confirm-gpu-idle", action="store_true")
    add_output(s)

    s = sub.add_parser("checkpoint-audit", help="List checkpoint files (no deserialization)")
    s.add_argument("--checkpoint", type=Path, required=True)
    add_output(s)

    s = sub.add_parser("merge-eval", help="Convert two official LeRobot eval_info.json files into paired CSV")
    s.add_argument("--baseline-info", type=Path, required=True)
    s.add_argument("--candidate-info", type=Path, required=True)
    s.add_argument("--baseline", default="B0")
    s.add_argument("--candidate", default="L1")
    s.add_argument("--start-seed", type=int, required=True)
    s.add_argument("--out", type=Path, required=True)

    s = sub.add_parser("report", help="Analyze real paired evaluation CSV and refuse mismatches")
    s.add_argument("--csv", type=Path, required=True)
    s.add_argument("--baseline", default="B0")
    s.add_argument("--candidate", default="L1")
    add_output(s)

    s = sub.add_parser("reference-demo", help="Explain finished, correct algorithms before exercises")
    add_output(s)
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    try:
        if a.operation == "validate-config":
            cfg = load_protocol(a.config)
            result = {"status": "STATIC_PROTOCOL_VALID", "sections": sorted(cfg),
                      "warning": "Does not imply real data, model, checkpoint or simulator validation."}
        elif a.operation == "shard-plan":
            from .dataset_files import plan_v3_episode_shards, train_episodes_from_split
            selected = train_episodes_from_split(a.split, a.repo_id)
            result = plan_v3_episode_shards(a.metadata_root, selected)
        elif a.operation == "hub-plan":
            from .artifacts import hub_sizes, file_plan
            if a.kind == "dataset_payload":
                if a.shard_plan is None:
                    raise ContractError("--shard-plan JSON required for curated payload")
                selected = read_json(a.shard_plan)["payload_files"]
            else:
                selected = None
            inventory = hub_sizes(a.repo, a.revision,
                                  repo_type="model" if a.kind == "model" else "dataset",
                                  allow_network=a.allow_network)
            result = file_plan(a.repo, a.revision, kind=a.kind,
                               selected_paths=selected, inventory=inventory)
        elif a.operation == "hub-download":
            from .artifacts import download_plan
            from .guards import require_explicit
            require_explicit(a.apply, "hub-download")
            result = download_plan(read_json(a.plan), a.target, a.project_root,
                                   apply=True, max_gib=a.max_gib)
        elif a.operation == "train":
            from .engine import train_plan, execute_command
            result = train_plan(a.config, bin_path=a.binary)
            if a.apply:
                cfg = load_protocol(a.config)
                return_code = execute_command(result, cfg, apply=True,
                                              confirmed_idle=a.confirm_gpu_idle)
                result = {"status": "OFFICIAL_TRAIN_PROCESS_EXITED", "returncode": return_code,
                          "note": "Inspect checkpoint and logs; do not infer task success."}
        elif a.operation == "evaluate":
            from .engine import eval_plan, execute_command
            result = eval_plan(a.config, a.checkpoint, a.output_dir, bin_path=a.binary)
            if a.apply:
                cfg = load_protocol(a.config)
                return_code = execute_command(result, cfg, apply=True,
                                              confirmed_idle=a.confirm_gpu_idle)
                result = {"status": "OFFICIAL_EVAL_PROCESS_EXITED", "returncode": return_code,
                          "note": "Verify actual eval results and task provenance."}
        elif a.operation == "checkpoint-audit":
            from .engine import checkpoint_inventory
            result = checkpoint_inventory(a.checkpoint)
        elif a.operation == "merge-eval":
            from .eval_extract import merge_official_eval, save_csv_new
            rows = merge_official_eval(a.baseline_info, a.candidate_info,
                                       baseline=a.baseline, candidate=a.candidate,
                                       start_seed=a.start_seed)
            result = save_csv_new(a.out, rows)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        elif a.operation == "report":
            from .reports import analyze_results
            result = analyze_results(a.csv, baseline=a.baseline, candidate=a.candidate)
        elif a.operation == "reference-demo":
            from .reference import extract_task_labels, make_action_chunk, paired_outcomes
            data = extract_task_labels({"tasks": ["pick up soup", "pick up soup", "place soup"]})
            chunk, pad = make_action_chunk([[float(i)]*7 for i in range(3)], 1, 4)
            rows = []
            for i, (b, c) in enumerate(((False, True), (True, False), (True, True))):
                for name, success in (("B0", b), ("L1", c)):
                    rows.append({"domain": "synthetic", "condition_id": str(i),
                                 "seed": 1000, "method": name, "success": success})
            result = {"labels": data, "action_chunk": chunk, "padding_mask": pad,
                      "paired": paired_outcomes(rows),
                      "NOTE": "Synthetic inputs only. No model, dataset, or eval run."}
        else:
            raise AssertionError("unreachable")
        _display(result, getattr(a, "out", None))
    except (ContractError, FileExistsError, OSError, ValueError, KeyError, TypeError) as exc:
        p.exit(2, "STOP: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
