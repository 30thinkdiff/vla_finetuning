"""One real SmolVLA + Recovery forward/backward through LoRA on a single CUDA GPU.

Offline and bounded: verifies pinned source + all local asset receipts, loads
the true 907MB pretrained LIBERO checkpoint, wraps with upstream SmolVLA PEFT
targets, runs ONE batch=1 training loss and backward. No optimizer step,
no checkpoint save, no auto-download, no training loop.

Use .venvs/train/bin/python with a CUDA GPU (>=12 GiB free). A successful
forward/backward proves training plumbing, NOT correct action semantics, model
quality, or a scientifically valid Recovery adaptation experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from smoke_recovery_processor import process_offline, file_sha256, ProcessorSmokeError
from fetch_smolvla_policy_weights import (
    MODEL, REVISION, WEIGHTS_SHA256, verify_weight_bytes,
)


class GPUSmokeError(ValueError):
    pass


def validate_assets(model_dir, runtime_dir):
    model_dir=Path(model_dir).expanduser().resolve()
    runtime_dir=Path(runtime_dir).expanduser().resolve()
    model_file=model_dir/"model.safetensors"
    receipt_file=model_dir/"policy_weights_receipt.json"
    if not model_file.is_file() or model_file.is_symlink() or not receipt_file.is_file():
        raise GPUSmokeError("Missing pinned policy weights and policy_weights_receipt.json")
    receipt=json.loads(receipt_file.read_text(encoding="utf-8"))
    if (receipt.get("model_repo")!=MODEL or receipt.get("model_revision")!=REVISION
            or receipt.get("sha256")!=WEIGHTS_SHA256 or model_dir.name!=REVISION):
        raise GPUSmokeError("Model policy receipt does not match expected pretrained revision")
    size,digest=verify_weight_bytes(model_file)
    if size!=receipt.get("size_bytes") or digest!=receipt.get("sha256"):
        raise GPUSmokeError("Weight size or digest differs from receipt")
    runtime_receipt=runtime_dir/"receipt.json"
    if not runtime_receipt.is_file():
        raise GPUSmokeError("Missing complete local SmolVLM2 processor receipt")
    data=json.loads(runtime_receipt.read_text(encoding="utf-8"))
    if (data.get("status")!="PINNED_SMOLVLM2_CONFIG_AND_PROCESSOR_NO_MODEL_WEIGHTS"
            or data.get("revision")!=runtime_dir.name
            or data.get("repo_id")!="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
        raise GPUSmokeError("Unexpected VLM config / processor provenance")
    for name in ("config.json","tokenizer.json","tokenizer_config.json",
                 "preprocessor_config.json","processor_config.json","chat_template.json"):
        item=data.get("files",{}).get(name)
        if not item or file_sha256(runtime_dir/name)!=item.get("sha256"):
            raise GPUSmokeError("Missing or changed VLM processor asset: "+name)
    return model_dir,runtime_dir,digest


def forward_backward(*, sample, sample_report, model_dir, runtime_dir):
    # Never allow implicit HF Hub access: local files are required for everything.
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TRANSFORMERS_OFFLINE"]="1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"]="1"
    model_dir,runtime_dir,weight_sha=validate_assets(model_dir,runtime_dir)
    import torch
    from peft import LoraConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if not torch.cuda.is_available():
        raise GPUSmokeError("CUDA not available in pinned isolated LeRobot environment")
    if not torch.cuda.is_bf16_supported():
        raise GPUSmokeError("BF16 CUDA not supported: abort rather than defaulting to FP32")
    free,total=torch.cuda.mem_get_info(0)
    if free < 12 * 1024**3:
        raise GPUSmokeError(f"GPU free={free/2**30:.1f}GiB; require >=12GiB before smoke")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    preflight,processed=process_offline(
        sample,sample_report,model_dir,runtime_dir,return_batch=True)
    if preflight.get("status")!="REAL_PREPROCESSOR_SMOKE_PASS_NO_MODEL_WEIGHTS":
        raise GPUSmokeError("Did not obtain validated processor tensors")

    cfg=SmolVLAConfig.from_pretrained(str(model_dir),local_files_only=True)
    if cfg.vlm_model_name!="HuggingFaceTB/SmolVLM2-500M-Video-Instruct":
        raise GPUSmokeError("Unexpected pretrained VLM identity")
    # The policy checkpoint contains its VLM parameters: initializing from
    # a *separate* 2GB VLM checkpoint is unnecessary and prohibited.
    cfg.vlm_model_name=str(runtime_dir)
    cfg.load_vlm_weights=False
    cfg.device="cuda"
    cfg.compile_model=False
    cfg.pretrained_path=str(model_dir)
    if cfg.chunk_size!=50 or cfg.action_feature.shape!=(7,):
        raise GPUSmokeError("Pinned checkpoint action shape/chunk mismatch")

    policy=SmolVLAPolicy.from_pretrained(
        str(model_dir),config=cfg,strict=True,local_files_only=True)
    targets=policy._get_default_peft_targets()
    lora_cfg=LoraConfig(
        r=8,lora_alpha=16,lora_dropout=0.0,
        target_modules=targets["target_modules"],
        modules_to_save=targets.get("modules_to_save",[]),
    )
    model=policy.wrap_with_peft(peft_config=lora_cfg)
    model.train()
    trainable = [(name,param) for name,param in model.named_parameters() if param.requires_grad]
    if not trainable or any("lora_" not in name for name,_ in trainable):
        raise GPUSmokeError("Unexpected trainable parameters; require LoRA-only smoke")
    trainable_count=sum(p.numel() for _,p in trainable)
    if trainable_count==0:
        raise GPUSmokeError("No LoRA trainable parameters")

    batch={key:(value.to(device="cuda",non_blocking=False) if
                isinstance(value,torch.Tensor) else value)
           for key,value in processed.items()}
    if not torch.equal(batch["action_is_pad"].cpu(),processed["action_is_pad"]):
        raise GPUSmokeError("Mask changed on GPU")
    model.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        loss,details=model(batch)
    if loss.ndim!=0 or not torch.isfinite(loss):
        raise GPUSmokeError("Model loss is non-scalar or non-finite")
    value=float(loss.detach().float().cpu().item())
    loss.backward()
    grad_tensors=sum(p.grad is not None for _,p in trainable)
    grad_nonzero=sum(
        p.grad is not None and bool(torch.any(p.grad.detach()!=0).item())
        for _,p in trainable)
    if grad_tensors==0 or grad_nonzero==0:
        raise GPUSmokeError("No nonzero gradient reached LoRA parameters")
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all().item())
           for _,p in trainable):
        raise GPUSmokeError("Nonfinite LoRA gradient")

    result={
        "status":"REAL_SMOLVLA_LORA_SINGLE_BATCH_FORWARD_BACKWARD_PASS",
        "training_complete":False,"optimizer_step_executed":False,
        "model_repo":MODEL,"model_revision":REVISION,
        "weight_sha256":weight_sha,
        "vlm_runtime_sha":runtime_dir.name,
        "sample_npz_sha256":preflight["sample_npz_sha256"],
        "torch_version":torch.__version__,
        "gpu_name":torch.cuda.get_device_name(0),
        "cuda_total_gib":round(total/2**30,2),
        "lo_ra_r":8,"lo_ra_alpha":16,
        "trainable_lora_parameters":trainable_count,
        "trainable_gradient_tensors":grad_tensors,
        "nonzero_gradient_tensors":grad_nonzero,
        "loss":value,"cuda_peak_allocated_gib":round(torch.cuda.max_memory_allocated(0)/2**30,3),
        "batch_action_shape":list(batch["action"].shape),
        "batch_language_shape":list(batch["observation.language.tokens"].shape),
        "note":"A single real Recovery batch/backward only. No optimizer update, checkpoint save, evaluation or control semantics validation.",
    }
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample",type=Path,required=True)
    p.add_argument("--sample-report",type=Path,required=True)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--vlm-runtime-dir",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--allow-gpu-smoke",action="store_true")
    args=p.parse_args(argv)
    if not args.allow_gpu_smoke:
        p.exit(2,"STOP: explicit --allow-gpu-smoke required\n")
    if args.out.exists() or args.out.is_symlink():
        p.exit(2,"STOP: report already exists; no overwrite\n")
    try:
        result=forward_backward(
            sample=args.sample,sample_report=args.sample_report,
            model_dir=args.model_dir,runtime_dir=args.vlm_runtime_dir)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x",encoding="utf-8") as fp:
            json.dump(result,fp,indent=2,ensure_ascii=False)
            fp.write("\n")
        print(json.dumps(result,indent=2,ensure_ascii=False))
    except (GPUSmokeError,ProcessorSmokeError,ValueError,OSError,KeyError,RuntimeError) as exc:
        p.exit(2,"STOP: "+str(exc)+"\n")


if __name__=="__main__":
    main()
