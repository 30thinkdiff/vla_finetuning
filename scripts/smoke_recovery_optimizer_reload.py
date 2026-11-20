"""One real Recovery Batch -> ONE LoRA optimizer step -> save/reload verification.

This is a bounded engineering test, NOT a fine-tuning run:
- loads the exact pinned pretrained SmolVLA LIBERO checkpoint (strict=True)
- validates original Recovery NPZ + official LeRobot preprocessing/normalizer
- attaches LoRA r=8 alpha=16 to the upstream SmolVLA target modules
- executes exactly ONE AdamW update at lr=1e-4, without scheduler
- verifies adapter parameters changed and remained finite
- saves only the PEFT adapter (<model weights), reloads it onto a freshly
  initialized, strictly loaded pretrained policy, and verifies tensor equality.

Never touches source data or pretrained model weights; requires explicit
--allow-one-step and refuses existing output paths. This test does not
establish correct Recovery control semantics, task success, or convergence.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path

from fetch_smolvla_policy_weights import REVISION
from smoke_recovery_lora_gpu import GPUSmokeError, validate_assets
from smoke_recovery_processor import process_offline, file_sha256, ProcessorSmokeError


def check_output_paths(adapter_dir, report_path):
    adapter_dir, report_path = Path(adapter_dir), Path(report_path)
    if adapter_dir.resolve() == report_path.resolve():
        raise GPUSmokeError("Adapter and report targets cannot be identical")
    for p in (adapter_dir, report_path):
        if p.exists() or p.is_symlink():
            raise GPUSmokeError("Refuse to overwrite prior one-step smoke artifact: " + str(p))
    return adapter_dir, report_path


def make_frozen_pretrained_policy(model_dir, runtime_dir):
    """Use a LOCAL VLM config/processor, then strictly load the full policy."""
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    cfg = SmolVLAConfig.from_pretrained(str(model_dir), local_files_only=True)
    if (cfg.vlm_model_name != "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
            or cfg.chunk_size != 50 or cfg.action_feature.shape != (7,)):
        raise GPUSmokeError("Unexpected pretrained SmolVLA model configuration")
    cfg.vlm_model_name = str(runtime_dir)
    # The full pretrained policy checkpoint already CONTAINS the VLM weights.
    # False skips a redundant VLM download during model construction; the
    # policy parameters are then all restored from model.safetensors.
    # LeRobot's _validate_peft_config prints a misleading warning based only
    # on this flag, not on the checkpoint actually loaded.
    cfg.load_vlm_weights = False
    cfg.device = "cuda"
    cfg.compile_model = False
    cfg.pretrained_path = str(model_dir)
    return SmolVLAPolicy.from_pretrained(
        str(model_dir), config=cfg, strict=True, local_files_only=True)


def one_step(*, sample, sample_report, model_dir, runtime_dir, adapter_dir):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    model_dir, runtime_dir, weight_sha = validate_assets(model_dir, runtime_dir)
    import torch
    from peft import LoraConfig, PeftModel
    from safetensors import safe_open

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise GPUSmokeError("CUDA/BF16 must be available")
    free, total = torch.cuda.mem_get_info(0)
    if free < 12 * 2**30:
        raise GPUSmokeError(f"Requires 12GiB free VRAM; found {free / 2**30:.2f}GiB")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    preflight, processed = process_offline(
        sample, sample_report, model_dir, runtime_dir, return_batch=True)
    if preflight.get("status") != "REAL_PREPROCESSOR_SMOKE_PASS_NO_MODEL_WEIGHTS":
        raise GPUSmokeError("Processor did not pass real offline smoke")

    base = make_frozen_pretrained_policy(model_dir, runtime_dir)
    target = base._get_default_peft_targets()
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=target["target_modules"],
        modules_to_save=target.get("modules_to_save", []))
    model = base.wrap_with_peft(peft_config=lora_cfg)
    model.train()

    trainable = {
        name: p for name, p in model.named_parameters() if p.requires_grad
    }
    if not trainable or any("lora_" not in name for name in trainable):
        raise GPUSmokeError("Only LoRA parameters may be trainable")
    count = sum(p.numel() for p in trainable.values())
    initial = {name: p.detach().cpu().clone() for name, p in trainable.items()}
    batch = {
        key: (val.to("cuda") if isinstance(val, torch.Tensor) else val)
        for key, val in processed.items()
    }
    optimizer = torch.optim.AdamW(
        list(trainable.values()), lr=1e-4, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, loss_details = model(batch)
    if loss.ndim != 0 or not bool(torch.isfinite(loss).item()):
        raise GPUSmokeError("Invalid scalar Loss")
    before_loss = float(loss.detach().float().cpu().item())
    loss.backward()

    grad_present = sum(p.grad is not None for p in trainable.values())
    grad_nonzero = sum(
        p.grad is not None and bool(torch.any(p.grad != 0).item())
        for p in trainable.values()
    )
    if not grad_nonzero or any(
        p.grad is not None and not bool(torch.isfinite(p.grad).all().item())
        for p in trainable.values()
    ):
        raise GPUSmokeError("LoRA gradient missing or nonfinite")
    clip_norm = torch.nn.utils.clip_grad_norm_(
        list(trainable.values()), max_norm=1.0)
    if not bool(torch.isfinite(clip_norm).item()):
        raise GPUSmokeError("Gradient norm invalid")

    optimizer.step()  # Exactly one optimizer update.
    updated = {}
    changed_tensors, changed_elements, max_delta = 0, 0, 0.0
    for name, p in trainable.items():
        new = p.detach().cpu().clone()
        if not bool(torch.isfinite(new).all().item()):
            raise GPUSmokeError("Optimizer created non-finite LoRA parameter")
        diff = (new - initial[name]).abs()
        num = int(torch.count_nonzero(diff).item())
        changed_elements += num
        changed_tensors += num > 0
        max_delta = max(max_delta, float(diff.max().item()))
        updated[name] = new
    if changed_tensors == 0 or changed_elements == 0 or not math.isfinite(max_delta):
        raise GPUSmokeError("No LoRA parameter changed after one AdamW step")

    adapter_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    adapter_weights = adapter_dir / "adapter_model.safetensors"
    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_weights.is_file() or not adapter_config.is_file():
        raise GPUSmokeError("PEFT did not save adapter-only files")
    with safe_open(adapter_weights, framework="pt", device="cpu") as file:
        adapter_tensor_count = len(file.keys())
    if adapter_tensor_count == 0:
        raise GPUSmokeError("Saved PEFT adapter has zero tensors")

    # Release GPU copies before fresh base+adapter reload.
    del model, base, optimizer, trainable, batch, loss, loss_details
    gc.collect()
    torch.cuda.empty_cache()

    base_reloaded = make_frozen_pretrained_policy(model_dir, runtime_dir)
    restored = PeftModel.from_pretrained(
        base_reloaded, str(adapter_dir), is_trainable=False)
    restored_named = dict(restored.named_parameters())
    if not set(updated).issubset(restored_named):
        raise GPUSmokeError("Reloaded adapter is missing trained LoRA tensors")
    max_reload_diff = 0.0
    for name, old in updated.items():
        new = restored_named[name].detach().cpu()
        if new.shape != old.shape or not bool(torch.isfinite(new).all().item()):
            raise GPUSmokeError("Reloaded LoRA tensor shape or finiteness mismatch")
        diff = float((new.float() - old.float()).abs().max().item())
        max_reload_diff = max(max_reload_diff, diff)
    if max_reload_diff > 1e-6:
        raise GPUSmokeError(f"Adapter reload changed weights: {max_reload_diff}")

    return {
        "status": "REAL_SMOLVLA_LORA_ONE_OPTIMIZER_STEP_SAVE_RELOAD_PASS",
        "scientific_training_completed": False,
        "optimizer_steps": 1,
        "model_revision": REVISION,
        "base_weight_sha256": weight_sha,
        "sample_npz_sha256": preflight["sample_npz_sha256"],
        "vlm_runtime_revision": runtime_dir.name,
        "gpu_name": torch.cuda.get_device_name(0),
        "initial_loss": before_loss,
        "loss_is_single_stochastic_batch_not_quality_metric": True,
        "lr": 1e-4,
        "grad_clip_norm_limit": 1.0,
        "actual_preclip_grad_norm": float(clip_norm.detach().float().cpu().item()),
        "trainable_lora_parameters": count,
        "gradient_tensors_present": grad_present,
        "gradient_tensors_nonzero": grad_nonzero,
        "updated_lora_tensors": changed_tensors,
        "updated_lora_elements": changed_elements,
        "max_abs_parameter_update": max_delta,
        "adapter_dir": str(adapter_dir.resolve()),
        "adapter_config_sha256": file_sha256(adapter_config),
        "adapter_safetensors_sha256": file_sha256(adapter_weights),
        "saved_adapter_tensors": adapter_tensor_count,
        "max_abs_reload_parameter_difference": max_reload_diff,
        "cuda_peak_allocated_gib": round(
            torch.cuda.max_memory_allocated(0) / 2**30, 3),
        "note": "Exactly one optimizer step, adapter-only save and fresh-base reload. No long training, validation set or rollout.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--sample-report", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--vlm-runtime-dir", type=Path, required=True)
    p.add_argument("--adapter-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--allow-one-step", action="store_true")
    a = p.parse_args(argv)
    if not a.allow_one_step:
        p.exit(2, "STOP: explicit --allow-one-step required\n")
    try:
        adapter, report = check_output_paths(a.adapter_dir, a.out)
        result = one_step(
            sample=a.sample, sample_report=a.sample_report,
            model_dir=a.model_dir, runtime_dir=a.vlm_runtime_dir,
            adapter_dir=adapter)
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("x", encoding="utf-8") as fp:
            json.dump(result, fp, ensure_ascii=False, indent=2)
            fp.write("\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (GPUSmokeError, ProcessorSmokeError, OSError, ValueError,
            KeyError, RuntimeError, ImportError) as exc:
        p.exit(2, "STOP: " + str(exc) + "\n")


if __name__ == "__main__":
    main()
