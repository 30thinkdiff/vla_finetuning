"""Read-only distribution audit: Recovery pilot3 vs frozen pretrained SmolVLA stats.

Reads the *actual* verified Expert v2.1 per-episode Parquet and the *actual*
loaded pretrained NormalizerProcessorStep state. Reports per-dimension mean,
std, p95 |z| and extreme rates after applying frozen checkpoint MEAN_STD.

This is a statistical mismatch diagnostic ONLY. It does not prove action
coordinate-frame/rotation/gripper semantics or select a training normalizer.
A tiny three-episode pilot, all cabinet tasks, is not a representative
estimate of all 3,184 Expert episodes or unseen FailureScenes.

Uses .venvs/train Python 3.12; CPU only; no model weights/GPU, no writes to
dataset, and output JSON is exclusive-create.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from smoke_recovery_processor import validate_model_files, file_sha256
from vla_finetuning.recovery_v21 import plan_v21_expert_files
from vla_finetuning.recovery_batch import verify_manifest_files, RecoveryBatchError


class DistributionAuditError(ValueError):
    pass


def dimension_report(values, pretrained_mean, pretrained_std, *, np):
    """Simple numeric report; never silently substitute dataset stats."""
    data=np.asarray(values,dtype=np.float64)
    means=np.asarray(pretrained_mean,dtype=np.float64).reshape(-1)
    stds=np.asarray(pretrained_std,dtype=np.float64).reshape(-1)
    if data.ndim != 2 or data.shape[1] != len(means) or len(means) != len(stds):
        raise DistributionAuditError("Feature width disagrees with checkpoint stats")
    if not np.isfinite(data).all() or not np.isfinite(means).all() or not np.isfinite(stds).all():
        raise DistributionAuditError("Nonfinite data/checkpoint normalizer parameters")
    if np.any(stds <= 0):
        raise DistributionAuditError("Zero/negative pretrained std cannot be treated as usable")
    z=(data-means[None,:])/stds[None,:]
    if not np.isfinite(z).all():
        raise DistributionAuditError("Nonfinite normalized features")
    dimensions=[]
    for i in range(data.shape[1]):
        dz=np.abs(z[:,i])
        dimensions.append({
            "component":i,"checkpoint_mean":float(means[i]),
            "checkpoint_std":float(stds[i]),
            "pilot_mean":float(data[:,i].mean()),
            "pilot_std_population":float(data[:,i].std(ddof=0)),
            "pilot_min":float(data[:,i].min()),
            "pilot_max":float(data[:,i].max()),
            "mean_z":float(z[:,i].mean()),
            "p95_abs_z":float(np.quantile(dz,0.95)),
            "max_abs_z":float(dz.max()),
            "fraction_abs_z_gt_3":float(np.mean(dz>3)),
            "fraction_abs_z_gt_5":float(np.mean(dz>5)),
        })
    return {
        "num_frames":int(data.shape[0]),
        "dimensions":dimensions,
        "max_p95_abs_z":float(max(x["p95_abs_z"] for x in dimensions)),
        "any_p95_abs_z_gt_3":any(x["p95_abs_z"]>3 for x in dimensions),
    }


def report_distribution(*, dataset_root, manifest, model_dir, runtime_dir, episodes):
    os.environ["HF_HUB_OFFLINE"]="1"
    os.environ["TRANSFORMERS_OFFLINE"]="1"
    root=Path(dataset_root).expanduser().resolve()
    if root.name!="2804e989adfb49cdde9b5c04bc7e0fb9f4fbfc73":
        raise DistributionAuditError("Unexpected Expert dataset SHA")
    model_dir,runtime_dir,state,receipt=validate_model_files(model_dir,runtime_dir)
    try:
        import numpy as np
        import pyarrow.parquet as pq
        from lerobot.processor import PolicyProcessorPipeline, NormalizerProcessorStep
    except ImportError as exc:
        raise DistributionAuditError("Use .venvs/train with NumPy/PyArrow/LeRobot installed") from exc
    plan=plan_v21_expert_files(root,episodes)
    if plan["observed_fps"]!=20:
        raise DistributionAuditError("Unexpected Expert FPS")
    processor=PolicyProcessorPipeline.from_pretrained(
        model_dir,config_filename="policy_preprocessor.json",local_files_only=True,
        overrides={
            "tokenizer_processor":{"tokenizer_name":str(runtime_dir)},
            "device_processor":{"device":"cpu"},
        },
    )
    norms=[s for s in processor.steps if isinstance(s,NormalizerProcessorStep)]
    if len(norms)!=1:
        raise DistributionAuditError("Expected exactly one pretrained normalizer")
    stats=norms[0]._tensor_stats
    feature_dims={"observation.state":8,"action":7}
    norm_values={}
    for key,width in feature_dims.items():
        if key not in stats or not {"mean","std"}.issubset(stats[key]):
            raise DistributionAuditError("Missing checkpoint stats for "+key)
        mu=stats[key]["mean"].detach().float().cpu().numpy().reshape(-1)
        sd=stats[key]["std"].detach().float().cpu().numpy().reshape(-1)
        if mu.shape!=(width,) or sd.shape!=(width,):
            raise DistributionAuditError("Unexpected pretrained stats vector width "+key)
        norm_values[key]=(mu,sd)

    all_values={key:[] for key in feature_dims}
    per_episode={}
    for episode in episodes:
        desc=plan["episode_files"][str(episode)]
        rel=desc["data_file"]
        verified=verify_manifest_files(root,manifest,[rel])
        table=pq.read_table(root/rel,columns=[
            "episode_index","frame_index","task_index","timestamp",
            "observation.state","action"])
        rows=table.to_pylist()
        if not rows or len(rows)!=desc["length"]:
            raise DistributionAuditError("Episode Parquet row count disagrees with metadata")
        if any(r.get("episode_index")!=episode or r.get("frame_index")!=j or
               not isinstance(r.get("timestamp"),(int,float)) or
               abs(float(r["timestamp"])-j/20)>0.003 for j,r in enumerate(rows)):
            raise DistributionAuditError("Episode/frame/time alignment mismatch")
        vals={}
        for key,width in feature_dims.items():
            try:
                arr=np.asarray([r[key] for r in rows],dtype=np.float64)
            except (KeyError,ValueError,TypeError) as exc:
                raise DistributionAuditError("Missing/invalid numeric vector "+key) from exc
            if arr.shape!=(len(rows),width) or not np.isfinite(arr).all():
                raise DistributionAuditError("Nonfinite/invalid Parquet vector "+key)
            vals[key]=arr
            all_values[key].append(arr)
        per_episode[str(episode)]={
            "frames":len(rows),"task":desc["tasks"],
            "parquet_sha256":verified[rel],
            "state":dimension_report(vals["observation.state"],*norm_values["observation.state"],np=np),
            "action":dimension_report(vals["action"],*norm_values["action"],np=np),
        }

    merged={key:np.concatenate(parts,axis=0) for key,parts in all_values.items()}
    overall={
        "state":dimension_report(merged["observation.state"],*norm_values["observation.state"],np=np),
        "action":dimension_report(merged["action"],*norm_values["action"],np=np),
    }
    return {
        "status":"REAL_EXPERT_PILOT_NORMALIZATION_DISTRIBUTION_AUDITED_NOT_SEMANTICALLY_VERIFIED",
        "ready_for_formal_training":False,
        "dataset_repo":"ataier/LIBERO_Recovery_Expert",
        "dataset_revision":root.name,
        "source_manifest_sha256":file_sha256(manifest),
        "model_revision":receipt["revision"],
        "model_normalizer_sha256":file_sha256(state),
        "episodes":list(episodes),
        "episode_frame_counts":{k:v["frames"] for k,v in per_episode.items()},
        "total_frames":int(overall["action"]["num_frames"]),
        "pooled":overall,
        "per_episode":per_episode,
        "warnings":[
            "Only selected pilot3 Expert episodes audited; these tasks are not representative of full Expert.",
            "This comparison measures pretrained-mean/std distribution shift, NOT coordinate/control semantics.",
            "No decision to use pretrained stats versus train-split stats can be made without frozen full train split and eval inverse normalization.",
            "No camera orientation or held-out Recovery simulator rollout is verified.",
        ],
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--vlm-runtime-dir",type=Path,required=True)
    p.add_argument("--episodes",default="0,15,30")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    try:
        ids=[int(x) for x in a.episodes.split(",")]
        if not ids or len(set(ids))!=len(ids) or len(ids)>20 or any(x<0 for x in ids):
            raise DistributionAuditError("Require 1..20 distinct nonnegative Episode IDs")
        if a.out.exists() or a.out.is_symlink():
            raise DistributionAuditError("Refuse to overwrite prior normalization audit")
        result=report_distribution(
            dataset_root=a.dataset_root,manifest=a.manifest,
            model_dir=a.model_dir,runtime_dir=a.vlm_runtime_dir,episodes=ids)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        with a.out.open("x",encoding="utf-8") as fp:
            json.dump(result,fp,indent=2,ensure_ascii=False)
            fp.write("\n")
        print(json.dumps({
            "status":result["status"],"ready_for_formal_training":False,
            "total_frames":result["total_frames"],
            "episode_frame_counts":result["episode_frame_counts"],
            "pooled_p95_abs_z_state":[round(x["p95_abs_z"],3) for x in result["pooled"]["state"]["dimensions"]],
            "pooled_p95_abs_z_action":[round(x["p95_abs_z"],3) for x in result["pooled"]["action"]["dimensions"]],
            "any_state_p95_abs_z_gt_3":result["pooled"]["state"]["any_p95_abs_z_gt_3"],
            "any_action_p95_abs_z_gt_3":result["pooled"]["action"]["any_p95_abs_z_gt_3"],
            "report":str(a.out)},indent=2,ensure_ascii=False))
    except (ValueError,RuntimeError,OSError,KeyError,TypeError,RecoveryBatchError) as exc:
        p.exit(2,"STOP: "+str(exc)+"\n")


if __name__=="__main__":
    main()
