"""Run one actual LIBERO-Recovery simulator reset and safe control step.

Uses the exact verified on-disk M0 FailureScenes .bddl + _sim_state.npy and
the independently installed official Recovery fork. No policy model and no
success-rate claim. Does NOT alter the original scenario or train env.
Requires MUJOCO_GL=egl, PYTHONPATH pointing to official pinned LIBERO folder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
from pathlib import Path


SCENE_SHA = "95acf98dd41a4965018069e7e4f27aac1fc69328"
RECOVERY_SHA = "832d54dfc43f0c9e4240bb8d92f5fda7001bb093"


class SceneBringupError(ValueError):
    pass


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def check_scene(manifest,scene_root,source):
    root=Path(scene_root).expanduser().resolve()
    source=Path(source).expanduser().resolve()
    if root.name != SCENE_SHA or not root.is_dir():
        raise SceneBringupError("Unexpected/missing pinned FailureScenes revision root")
    if not (source/"LIBERO/setup.py").is_file():
        raise SceneBringupError("Official LIBERO-Recovery fork not installed")
    head=subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True,timeout=8).strip()
    if head != RECOVERY_SHA:
        raise SceneBringupError("Official source SHA does not match frozen benchmark")
    doc=json.loads(Path(manifest).read_text(encoding="utf-8"))
    if doc.get("kind")!="modelscope_recovery_exact_manifest_v1" or doc.get("revision")!=SCENE_SHA:
        raise SceneBringupError("Pinned M0 scene manifest mismatch")
    selection=doc.get("selection",{})
    entries=doc.get("files",[])
    if (selection.get("suite")!="libero_object" or selection.get("seed")!=7 or
        len([x for x in entries if x["path"].endswith(".bddl")])!=1):
        raise SceneBringupError("Expected exactly one frozen LIBERO-Object scene")
    bddl=selection.get("bddl")
    npy=selection.get("sim_state")
    if (not isinstance(bddl,str) or not isinstance(npy,str) or
        not bddl.startswith("scenes/libero_object/") or
        npy != bddl[:-5] + "_sim_state.npy"):
        raise SceneBringupError("Scene pair mismatch")
    receipt_file=root/".vla_recovery_verified_m0_object_s7.json"
    if not receipt_file.is_file():
        raise SceneBringupError("Verified download receipt missing")
    receipt=json.loads(receipt_file.read_text(encoding="utf-8"))
    if receipt.get("revision")!=SCENE_SHA or receipt.get("repo_id")!="ataier/LIBERO_Recovery_FailureScenes":
        raise SceneBringupError("Download receipt source mismatch")
    rec={x["path"]:x for x in receipt.get("files",[])}
    manifest_entries={x["path"]:x for x in entries}
    for rel in (bddl,npy):
        p=(root/rel)
        if not p.is_file() or p.is_symlink() or not p.resolve().is_relative_to(root):
            raise SceneBringupError("Missing or unsafe scenario member: "+rel)
        expected=manifest_entries.get(rel,{}).get("sha256")
        observed=digest(p)
        if (expected != observed or rec.get(rel,{}).get("sha256")!=observed
                or p.stat().st_size!=manifest_entries[rel]["size_bytes"]):
            raise SceneBringupError("Scenario SHA/size not verified: "+rel)
    return root/bddl,root/npy,selection


def bringup(manifest,scene_root,source,seed):
    if type(seed) is not int or seed < 0 or seed > 2**31-1:
        raise SceneBringupError("Invalid seed")
    bddl,npy,selection=check_scene(manifest,scene_root,source)
    # Required for headless EGL; do not silently use X server.
    if os.getenv("MUJOCO_GL")!="egl" or os.getenv("PYOPENGL_PLATFORM")!="egl":
        raise SceneBringupError("Set MUJOCO_GL=egl PYOPENGL_PLATFORM=egl first")
    import numpy as np
    from robosuite.utils.errors import RandomizationError
    from libero.libero.envs import OffScreenRenderEnv

    init_state=np.load(npy,allow_pickle=False)
    if init_state.ndim!=1 or init_state.size < 20 or not np.isfinite(init_state).all():
        raise SceneBringupError("Unexpected flattened simulation state")
    env=None
    try:
        # Match official setup: random placement is overwritten by fixed init.
        for attempt in range(12):
            np.random.seed(seed+1_000_001+attempt)
            try:
                env=OffScreenRenderEnv(
                    bddl_file_name=str(bddl),
                    camera_heights=256,camera_widths=256)
                break
            except RandomizationError:
                continue
        if env is None:
            raise SceneBringupError("Simulator placement failed after 12 attempts")
        env.seed(seed)
        obs=None
        for attempt in range(12):
            np.random.seed(seed+1_000_001+10_001+attempt)
            try:
                env.reset()
                obs=env.set_init_state(init_state.copy())
                break
            except RandomizationError:
                continue
        if obs is None:
            raise SceneBringupError("Simulator reset failed after 12 attempts")
        for robot in env.robots:
            robot.controller.update()
            robot.controller.update_initial_joints(robot.controller.joint_pos)
            robot.controller.reset_goal()
        required=("agentview_image","robot0_eye_in_hand_image","robot0_eef_pos",
                  "robot0_eef_quat","robot0_gripper_qpos")
        if any(k not in obs for k in required):
            raise SceneBringupError("Missing real LIBERO observation fields")
        for key in required[:2]:
            if tuple(obs[key].shape)!=(256,256,3):
                raise SceneBringupError("Unexpected simulator RGB format: "+key)
        state_width=sum(np.asarray(obs[k]).size for k in required[2:])
        if state_width!=9:  # 3 EEF xyz + 4 quaternion + 2 gripper joints
            raise SceneBringupError("Unexpected raw simulator observation widths")
        before=[float(x) for x in np.asarray(obs["robot0_eef_pos"]).reshape(-1)]
        # Exactly one legal zero-arm-delta command; no policy and no rollout.
        obs_next,reward,done,info=env.step([0.,0.,0.,0.,0.,0.,-1.])
        if not isinstance(obs_next,dict) or not all(k in obs_next for k in required):
            raise SceneBringupError("Post-step simulator observation invalid")
        after=[float(x) for x in np.asarray(obs_next["robot0_eef_pos"]).reshape(-1)]
        result={
            "status":"REAL_RECOVERY_SIM_RESET_AND_ONE_SAFE_STEP_PASS_NO_POLICY",
            "is_policy_rollout":False,"is_scientific_evaluation":False,
            "scene_revision":SCENE_SHA,"source_revision":RECOVERY_SHA,
            "bddl_relative_path":selection["bddl"],
            "scene_bddl_sha256":digest(bddl),"sim_state_sha256":digest(npy),
            "seed":seed,"sim_state_vector_length":int(init_state.size),
            "agentview_shape":list(obs["agentview_image"].shape),
            "wrist_shape":list(obs["robot0_eye_in_hand_image"].shape),
            "eef_pos_before":before,"eef_pos_after":after,
            "one_step_reward":float(reward),"one_step_done":bool(done),
            "note":"One safe simulator step to establish valid Recovery episode initialization only. No SmolVLA inference or metric.",
        }
        return result
    finally:
        if env is not None:
            env.close()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--scene-root",type=Path,required=True)
    p.add_argument("--recovery-source",type=Path,required=True)
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--allow-one-step",action="store_true")
    args=p.parse_args(argv)
    if not args.allow_one_step:
        p.exit(2,"STOP: --allow-one-step required; real MuJoCo EGL simulation is not dry-run\n")
    if args.out.exists() or args.out.is_symlink():
        p.exit(2,"STOP: report exists; no overwrite\n")
    try:
        result=bringup(args.manifest,args.scene_root,args.recovery_source,args.seed)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x",encoding="utf-8") as f:
            json.dump(result,f,ensure_ascii=False,indent=2)
            f.write("\n")
        print(json.dumps(result,ensure_ascii=False,indent=2))
    except (SceneBringupError,KeyError,TypeError,ValueError,RuntimeError,OSError,
            subprocess.SubprocessError) as exc:
        p.exit(2,"STOP: "+str(exc)+"\n")


if __name__=="__main__":
    main()
