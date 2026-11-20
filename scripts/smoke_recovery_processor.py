"""True (offline) LeRobot SmolVLA preprocessor smoke using a real Recovery NPZ.

This is NOT a model forward/backward or a train-ready claim. Requires:
  --model-dir       pinned model config + policy_preprocessor.json
                    + the named normalizer .safetensors state file
  --tokenizer-dir   complete local SmolVLM2 tokenizer (not a Hub repo ID)
  --sample          output from build_recovery_pilot_batch.py
  --sample-report   corresponding provenance JSON

No network calls, no GPU, no checkpoint policy weights, no dataset changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


class ProcessorSmokeError(ValueError):
    pass


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_shapes(arrays):
    """Early fail on common silent mis-batching errors, before importing Torch."""
    wanted = {
        "observation.images.image": (3, 256, 256),
        "observation.images.image2": (3, 256, 256),
        "observation.state": (8,),
        "action": (50, 7),
        "action_is_pad": (50,),
    }
    for key, shape in wanted.items():
        if key not in arrays or tuple(arrays[key].shape) != shape:
            got = None if key not in arrays else tuple(arrays[key].shape)
            raise ProcessorSmokeError(f"Bad raw sample {key}: expected {shape}, got {got}")
    if "task" not in arrays or tuple(arrays["task"].shape) != ():
        raise ProcessorSmokeError("Task must be a scalar Unicode array")
    task = arrays["task"].item()
    if not isinstance(task, str) or not task.strip():
        raise ProcessorSmokeError("Task must be a non-empty string")
    return task


def validate_model_files(model_dir, tokenizer_dir):
    """Strict offline readiness check; never download or invent stats."""
    model_dir = Path(model_dir).expanduser().resolve()
    tokenizer_dir = Path(tokenizer_dir).expanduser().resolve()
    for name in ("config.json", "policy_preprocessor.json", "receipt.json"):
        if not (model_dir / name).is_file():
            raise ProcessorSmokeError("Missing local model config: " + str(model_dir / name))
    receipt = json.loads((model_dir/"receipt.json").read_text(encoding="utf-8"))
    if receipt.get("model_repo") != "lerobot/smolvla_libero" or receipt.get("revision") != model_dir.name:
        raise ProcessorSmokeError("Model receipt repo/revision does not match path")
    if len(receipt.get("revision", "")) != 40:
        raise ProcessorSmokeError("Expected model commit SHA")
    for name in ("config.json", "policy_preprocessor.json"):
        expected = receipt["files"][name]["sha256"]
        if file_sha256(model_dir/name) != expected:
            raise ProcessorSmokeError("Model metadata no longer matches pinned receipt: "+name)
    pre = json.loads((model_dir/"policy_preprocessor.json").read_text(encoding="utf-8"))
    norm = [s.get("state_file") for s in pre.get("steps", []) if
            isinstance(s, dict) and s.get("registry_name") == "normalizer_processor"]
    if len(norm) != 1 or not isinstance(norm[0], str) or not norm[0].endswith(".safetensors"):
        raise ProcessorSmokeError("Expected exact preprocessor normalizer state filename")
    state = model_dir/norm[0]
    if not state.is_file() or state.stat().st_size == 0:
        raise ProcessorSmokeError(
            "Missing pinned normalizer stats "+str(state)+"; two JSON files alone are insufficient")
    state_receipt_path = model_dir / "processor_state_receipt.json"
    if not state_receipt_path.is_file():
        raise ProcessorSmokeError("Missing pinned processor_state_receipt.json")
    state_receipt = json.loads(state_receipt_path.read_text(encoding="utf-8"))
    if (state_receipt.get("model_revision") != receipt["revision"]
            or state_receipt.get("state_file") != norm[0]
            or state_receipt.get("state_file_sha256") != file_sha256(state)):
        raise ProcessorSmokeError("Normalizer state does not match pinned asset receipt")
    if not tokenizer_dir.is_dir() or not (tokenizer_dir/"tokenizer_config.json").is_file():
        raise ProcessorSmokeError("Local tokenizer directory with tokenizer_config.json is required")
    tokenizer_receipt_path = tokenizer_dir / "receipt.json"
    if not tokenizer_receipt_path.is_file():
        raise ProcessorSmokeError("Missing pinned tokenizer receipt")
    tokenizer_receipt = json.loads(tokenizer_receipt_path.read_text(encoding="utf-8"))
    if (tokenizer_receipt.get("repo_id") != "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
            or tokenizer_receipt.get("revision") != tokenizer_dir.name):
        raise ProcessorSmokeError("Wrong tokenizer model revision/source")
    for name, info in tokenizer_receipt.get("files", {}).items():
        if name not in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                        "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model",
                        "sentencepiece.bpe.model", "config.json",
                        "preprocessor_config.json", "processor_config.json",
                        "chat_template.json", "generation_config.json"):
            raise ProcessorSmokeError("Unexpected tokenizer receipt file")
        if (not (tokenizer_dir/name).is_file()
                or file_sha256(tokenizer_dir/name) != info.get("sha256")):
            raise ProcessorSmokeError("Tokenizer assets failed receipt checksum: "+name)
    if "tokenizer_config.json" not in tokenizer_receipt.get("files", {}):
        raise ProcessorSmokeError("Tokenizer receipt missing tokenizer_config.json")
    return model_dir, tokenizer_dir, state, receipt


def process_offline(sample, report_path, model_dir, tokenizer_dir, *, return_batch=False):
    # Set before any transformers/huggingface_hub import: fail closed if assets missing.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model_dir, tokenizer_dir, state, receipt = validate_model_files(model_dir, tokenizer_dir)
    path = Path(sample).expanduser().resolve()
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    if (report.get("status") != "REAL_RECOVERY_PILOT_SAMPLE_BUILT_NO_TRAINING"
            or report.get("sample_npz_sha256") != file_sha256(path)
            or report.get("model_action_chunk") != 50):
        raise ProcessorSmokeError("Sample does not match provenance report")
    try:
        import numpy as np
        import torch
        from lerobot.processor import PolicyProcessorPipeline, NormalizerProcessorStep
    except ImportError as exc:
        raise ProcessorSmokeError(
            "Missing torch/NumPy/installed pinned LeRobot. Use dedicated .venvs/train (not .venvs/data)") from exc

    with np.load(path, allow_pickle=False) as file:
        arrays = {k:file[k] for k in file.files}
    task = validate_shapes(arrays)
    if arrays["action_is_pad"].dtype != np.bool_:
        raise ProcessorSmokeError("action_is_pad must be bool")
    pad_count = int(arrays["action_is_pad"].sum())
    if pad_count != report["action_is_pad_true_count"]:
        raise ProcessorSmokeError("NPZ pad mask count disagrees with provenance report")
    if not (not arrays["action_is_pad"].any() or
            np.array_equal(arrays["action_is_pad"], np.r_[np.zeros(50-pad_count, bool), np.ones(pad_count, bool)])):
        raise ProcessorSmokeError("action_is_pad must be a trailing suffix")
    for key in ("action", "observation.state", "observation.images.image", "observation.images.image2"):
        if arrays[key].dtype != np.float32 or not np.isfinite(arrays[key]).all():
            raise ProcessorSmokeError("Required finite float32 sample: "+key)
    for key in ("observation.images.image", "observation.images.image2"):
        if arrays[key].min() < 0 or arrays[key].max() > 1:
            raise ProcessorSmokeError("Image pixels must be CHW in [0,1]")

    # Critical: upstream AddBatchDimensionActionStep only unsqueezes a 1D action.
    # A 50x7 chunk must be explicitly batched here. Images/state are still
    # UNBATCHED; upstream observation step adds their B dimension.
    feed = {
        "observation.images.image": torch.from_numpy(arrays["observation.images.image"].copy()),
        "observation.images.image2": torch.from_numpy(arrays["observation.images.image2"].copy()),
        "observation.state": torch.from_numpy(arrays["observation.state"].copy()),
        "action": torch.from_numpy(arrays["action"].copy()).unsqueeze(0),
        "action_is_pad": torch.from_numpy(arrays["action_is_pad"].copy()).unsqueeze(0),
        "task": task,
    }
    pipeline = PolicyProcessorPipeline.from_pretrained(
        model_dir, config_filename="policy_preprocessor.json", local_files_only=True,
        overrides={
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_dir)},
            "device_processor": {"device": "cpu"},
        },
    )
    norm_steps=[s for s in pipeline.steps if isinstance(s, NormalizerProcessorStep)]
    if len(norm_steps) != 1:
        raise ProcessorSmokeError("Preprocessor missing the true normalizer step")
    stats = norm_steps[0]._tensor_stats
    for key, dim in (("observation.state", 8), ("action", 7)):
        if key not in stats or not {"mean","std"}.issubset(stats[key]):
            raise ProcessorSmokeError("Missing checkpoint stats for "+key)
        for attr in ("mean", "std"):
            v=stats[key][attr]
            if tuple(v.shape) not in ((dim,), (1,dim)) or not torch.isfinite(v).all():
                raise ProcessorSmokeError("Invalid loaded checkpoint "+key+" "+attr)

    result=pipeline(feed)
    for key, shape in (
        ("observation.images.camera1",(1,3,256,256)),
        ("observation.images.camera2",(1,3,256,256)),
        ("observation.state",(1,8)),
        ("action",(1,50,7)),
        ("action_is_pad",(1,50)),
    ):
        if key not in result or tuple(result[key].shape) != shape:
            raise ProcessorSmokeError(f"Unexpected processed {key} shape; got {None if key not in result else tuple(result[key].shape)}")
    for key in ("observation.language.tokens", "observation.language.attention_mask"):
        # Keys match pinned SmolVLA forward's OBS_LANGUAGE_* constants.
        if key not in result or result[key].ndim != 2 or result[key].shape[0] != 1:
            raise ProcessorSmokeError("Missing batched language token feature: "+key)
    if result["action_is_pad"].dtype != torch.bool or \
            not torch.equal(result["action_is_pad"],feed["action_is_pad"]):
        raise ProcessorSmokeError("Padding mask changed during processing")
    for key in ("observation.state", "action", "observation.images.camera1", "observation.images.camera2"):
        if not torch.isfinite(result[key]).all():
            raise ProcessorSmokeError("Nonfinite processed values: "+key)
    summary = {
        "status":"REAL_PREPROCESSOR_SMOKE_PASS_NO_MODEL_WEIGHTS",
        "ready_to_train":False,
        "sample_npz_sha256":file_sha256(path),
        "model_revision":receipt["revision"],
        "model_source_endpoint":receipt["source_endpoint"],
        "normalizer_file":state.name,
        "normalizer_sha256":file_sha256(state),
        "tokenizer_dir":str(tokenizer_dir),
        "tokenizer_files":sorted(p.name for p in tokenizer_dir.iterdir() if p.is_file()),
        "output_shapes": {k:list(result[k].shape) for k in (
            "observation.images.camera1","observation.images.camera2","observation.state","action",
            "action_is_pad","observation.language.tokens","observation.language.attention_mask")},
        "valid_action_steps":50-pad_count,
        "note":"Real LeRobot preprocessor/tokenizer/normalizer exercised, but no SmolVLA policy forward/backward, controller semantics or training.",
    }
    if return_batch:
        # For a single-batch GPU diagnostic only; do not serialize large tensors.
        return summary, result
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample",type=Path,required=True)
    p.add_argument("--sample-report",type=Path,required=True)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--tokenizer-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    try:
        if args.out.exists() or args.out.is_symlink():
            raise ProcessorSmokeError("Refuse to overwrite existing report")
        result=process_offline(args.sample,args.sample_report,args.model_dir,args.tokenizer_dir)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x",encoding="utf-8") as file:
            json.dump(result,file,indent=2,ensure_ascii=False)
            file.write("\n")
        print(json.dumps(result,indent=2,ensure_ascii=False))
    except (ProcessorSmokeError,OSError,ImportError,KeyError,TypeError,ValueError,RuntimeError) as exc:
        p.exit(2,"STOP: "+str(exc)+"\n")


if __name__=="__main__":
    main()
