"""Fetch exact pinned SmolVLA inference postprocessor and action unnormalizer state.

Does NOT change model weights, pretrained normalizer, or train environment.
Only two tiny assets from the SAME model revision. The unnormalizer is a
hard prerequisite for physical-unit LIBERO actions; NEVER execute raw
normalized SmolVLA action directly in the simulator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from fetch_smolvla_contract_metadata import (
    ALLOWED_ENDPOINTS, get_bounded, get_bounded_curl_ipv4
)

MODEL = "lerobot/smolvla_libero"
SHA = "a24f1f380461b665158d52acd140152c79e8b36d"
MAX_SMALL = 1024 * 1024


class PostprocessorError(ValueError):
    pass


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def validate_safetensors(data):
    if len(data) < 12:
        raise PostprocessorError("Truncated safetensors state")
    head=int.from_bytes(data[:8],"little")
    if not 0<head<=min(MAX_SMALL,len(data)-8):
        raise PostprocessorError("Bad safetensors header length")
    obj=json.loads(data[8:8+head])
    if not isinstance(obj,dict) or not any(x!="__metadata__" for x in obj):
        raise PostprocessorError("Missing tensor metadata")
    return obj


def prepare(model_dir,endpoint):
    p=Path(model_dir).expanduser().resolve()
    if endpoint not in ALLOWED_ENDPOINTS:
        raise PostprocessorError("Unapproved endpoint")
    original=p/"receipt.json"
    if p.name!=SHA or not original.is_file():
        raise PostprocessorError("Pinned model revision or metadata receipt missing")
    doc=json.loads(original.read_text(encoding="utf-8"))
    if doc.get("model_repo")!=MODEL or doc.get("revision")!=SHA:
        raise PostprocessorError("Model identity mismatch")
    for name in ("config.json","policy_preprocessor.json"):
        info=doc.get("files",{}).get(name)
        file=p/name
        if not info or not file.is_file() or checksum(file.read_bytes())!=info.get("sha256"):
            raise PostprocessorError("Existing pinned model contract mismatch: "+name)
    for name in ("policy_postprocessor.json",
                 "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                 "postprocessor_assets_receipt.json"):
        f=p/name
        if f.exists() or f.is_symlink():
            raise PostprocessorError("Refusing overwrite: "+str(f))
    return p


def fetch(model_dir,endpoint,transport):
    p=prepare(model_dir,endpoint)
    getter = get_bounded if transport=="urllib" else (
        lambda url,max_bytes: get_bounded_curl_ipv4(
            url,max_bytes=max_bytes,direct=transport=="curl-ipv4-direct"))
    url=f"{endpoint}/{MODEL}/resolve/{SHA}/policy_postprocessor.json"
    raw=getter(url,max_bytes=MAX_SMALL)
    meta=json.loads(raw)
    if not isinstance(meta,dict) or not isinstance(meta.get("steps"),list):
        raise PostprocessorError("Malformed postprocessor JSON")
    steps=[s for s in meta["steps"] if
           isinstance(s,dict) and s.get("registry_name")=="unnormalizer_processor"]
    if len(steps)!=1:
        raise PostprocessorError("Expected exactly one action unnormalizer")
    cfg=steps[0].get("config",{})
    if (cfg.get("norm_map",{}).get("ACTION")!="MEAN_STD"
            or cfg.get("features",{}).get("action",{}).get("shape") != [7]):
        raise PostprocessorError("Unexpected action unnormalizer / dims")
    state=steps[0].get("state_file")
    if state!="policy_postprocessor_step_0_unnormalizer_processor.safetensors":
        raise PostprocessorError("Unexpected unnormalizer state name")
    state_url=f"{endpoint}/{MODEL}/resolve/{SHA}/{state}"
    blob=getter(state_url,max_bytes=MAX_SMALL)
    validate_safetensors(blob)
    result={
        "status":"PINNED_SMOLVLA_POSTPROCESSOR_ASSETS_FETCHED_NOT_RUNTIME_VERIFIED",
        "model_repo":MODEL,"revision":SHA,"source_endpoint":endpoint,
        "mirror_not_official_provenance":endpoint!="https://huggingface.co",
        "files":{
            "policy_postprocessor.json":{"sha256":checksum(raw),"size_bytes":len(raw)},
            state:{"sha256":checksum(blob),"size_bytes":len(blob)}
        }
    }
    # Network and format verification complete before any writes.
    with (p/"policy_postprocessor.json").open("xb") as out:
        out.write(raw)
    with (p/state).open("xb") as out:
        out.write(blob)
    with (p/"postprocessor_assets_receipt.json").open("x",encoding="utf-8") as out:
        json.dump(result,out,ensure_ascii=False,indent=2)
        out.write("\n")
    return result


def main(argv=None):
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument("--model-dir",type=Path,required=True)
    pa.add_argument("--endpoint",choices=ALLOWED_ENDPOINTS,default="https://huggingface.co")
    pa.add_argument("--transport",choices=["urllib","curl-ipv4","curl-ipv4-direct"],default="curl-ipv4")
    pa.add_argument("--allow-network",action="store_true")
    a=pa.parse_args(argv)
    if not a.allow_network:
        pa.exit(2,"STOP: specify --allow-network for two pinned small files\n")
    try:
        print(json.dumps(fetch(a.model_dir,a.endpoint,a.transport),ensure_ascii=False,indent=2))
    except (PostprocessorError,OSError,ValueError,KeyError,TypeError) as exc:
        pa.exit(2,"STOP: "+str(exc)+"\n")


if __name__=="__main__":
    main()
