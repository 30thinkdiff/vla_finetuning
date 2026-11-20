"""Select one deterministic verified-pair *candidate* for the first M0 rollout.

Input: pinned ModelScope FailureScenes INVENTORY JSON (already fetched).
Output: exclusive-create exact download manifest, selecting ONE .bddl and its
matching _sim_state.npy; optionally includes existing scene index JSONL.

No filesystem asset downloads, GPU, simulator, or remote calls. This chooses
an engineering bring-up scene, NOT a representative / held-out test set.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from recovery_hub import exact_manifest, write_new, safe_path


class SceneSelectionError(ValueError):
    pass


ALLOWED_SUITES=frozenset({
    "libero_spatial","libero_object","libero_goal","libero_10","libero_90"
})
DATASET="ataier/LIBERO_Recovery_FailureScenes"


def choose_first_m0_scene(inventory: dict, *, suite: str, seed: int=7,
                          include_index: bool=True) -> dict:
    if suite not in ALLOWED_SUITES:
        raise SceneSelectionError("Unsupported LIBERO suite")
    if type(seed) is not int or seed < 0 or seed > 2**32-1:
        raise SceneSelectionError("Seed must be integer in [0,2^32-1]")
    if inventory.get("repo_id")!=DATASET or inventory.get("kind")!="modelscope_recovery_inventory_v1":
        raise SceneSelectionError("Expected pinned ModelScope FailureScenes inventory")
    rev=inventory.get("revision")
    if not isinstance(rev,str) or not re.fullmatch(r"[0-9a-f]{40}",rev):
        raise SceneSelectionError("Require exact fixed 40-char dataset revision")

    files=inventory.get("files")
    if not isinstance(files,list):
        raise SceneSelectionError("No inventory file list")
    lookup={}
    for e in files:
        if not isinstance(e,dict):
            raise SceneSelectionError("Malformed inventory item")
        name=safe_path(e.get("path"))
        size=e.get("size_bytes")
        if name in lookup or type(size) is not int or size<0:
            raise SceneSelectionError("Duplicate or invalid inventory member")
        lookup[name]=e
    prefix=f"scenes/{suite}/"
    pairs=[]
    for bddl in lookup:
        if not bddl.startswith(prefix) or not bddl.endswith(".bddl"):
            continue
        npy=bddl[:-5]+"_sim_state.npy"
        if npy in lookup and lookup[bddl]["size_bytes"]>0 and lookup[npy]["size_bytes"]>0:
            pairs.append((bddl,npy))
    pairs.sort()
    if not pairs:
        raise SceneSelectionError(f"No complete .bddl + _sim_state.npy pair under {prefix}")
    bddl,npy=pairs[random.Random(seed).randrange(len(pairs))]
    selected=[bddl,npy]
    index="index/selection.jsonl"
    if include_index and index in lookup and lookup[index]["size_bytes"]>0:
        selected.append(index)
    # Stop rather than allow this first tiny bring-up step to expand to bulk.
    manifest=exact_manifest(inventory,selected,max_gib=0.1)
    manifest["purpose"]="M0_ENGINEERING_SCENE_BRINGUP_NOT_BENCHMARK"
    manifest["selection"]={
        "suite":suite, "seed":seed,"pair_candidates":len(pairs),
        "bddl":bddl, "sim_state":npy, "index_included":index in selected,
        "warning":"No task/source-group split validation or scientific evaluation implied",
    }
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory",type=Path,required=True)
    p.add_argument("--suite",choices=sorted(ALLOWED_SUITES),default="libero_object")
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args(argv)
    try:
        inv=json.loads(a.inventory.read_text(encoding="utf-8"))
        manifest=choose_first_m0_scene(inv,suite=a.suite,seed=a.seed)
        write_new(a.out,manifest)
        print(json.dumps({
            "status":"M0_ONE_SCENE_EXACT_MANIFEST_ONLY_NOT_DOWNLOADED",
            "revision":manifest["revision"],
            "selection":manifest["selection"],
            "manifest":str(a.out),"files":len(manifest["files"]),
            "total_bytes":manifest["total_bytes"],
        },ensure_ascii=False,indent=2))
    except (OSError,TypeError,ValueError,KeyError,json.JSONDecodeError) as exc:
        p.exit(2,"STOP: "+str(exc)+"\n")


if __name__=="__main__":
    main()
