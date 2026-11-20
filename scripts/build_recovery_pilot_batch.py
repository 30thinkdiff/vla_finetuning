"""Materialize one true Recovery Expert sample for later SmolVLA Processor smoke.

Example:
  PYTHONPATH=src .venvs/data/bin/python scripts/build_recovery_pilot_batch.py \
    --dataset-root datasets/recovery_expert/<EXPERT_SHA> \
    --manifest runs/recovery_pilot3_manifest_<EXPERT_SHA>.json \
    --contract runs/recovery_smolvla_contract_<EXPERT_SHA>_<MODEL_SHA>.json \
    --episode 0 --frame 10 \
    --out-npz runs/recovery_batch/ep0_f10.npz \
    --out-report runs/recovery_batch/ep0_f10.json
"""
import argparse
import json
from pathlib import Path

from vla_finetuning.recovery_batch import RecoveryBatchError, build_pilot_batch


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--contract", type=Path, required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--frame", type=int, required=True)
    p.add_argument("--out-npz", type=Path, required=True)
    p.add_argument("--out-report", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        result = build_pilot_batch(
            args.dataset_root, args.manifest, args.contract,
            episode=args.episode, frame=args.frame,
            out_npz=args.out_npz, out_report=args.out_report,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (OSError, RecoveryBatchError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        p.exit(2, "STOP: "+str(exc)+"\n")


if __name__ == "__main__":
    main()
