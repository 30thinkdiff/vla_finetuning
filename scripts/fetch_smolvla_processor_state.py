"""Fetch only pretrained SmolVLA normalization state, never model weights.

Requires the local pinned model JSON receipt from
fetch_smolvla_contract_metadata.py. The exact state filename comes from
policy_preprocessor.json, not a hardcoded assumption. Source is an approved
HTTPS endpoint (official Hugging Face or third-party mirror) and is recorded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from fetch_smolvla_contract_metadata import (
    ALLOWED_ENDPOINTS,
    get_bounded_curl_ipv4,
    get_bounded,
)

MAX_STATE_BYTES = 4 * 1024 * 1024


class StateAssetError(ValueError):
    pass


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def prepare_request(model_dir, endpoint):
    model_dir=Path(model_dir).expanduser().resolve()
    if endpoint not in ALLOWED_ENDPOINTS:
        raise StateAssetError("Unapproved Hub endpoint")
    receipt_path=model_dir/"receipt.json"
    pre_path=model_dir/"policy_preprocessor.json"
    if not receipt_path.is_file() or not pre_path.is_file():
        raise StateAssetError("Local model config/receipt missing")
    receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
    pre_data=pre_path.read_bytes()
    if (receipt.get("model_repo")!="lerobot/smolvla_libero"
            or receipt.get("revision")!=model_dir.name
            or receipt.get("files",{}).get("policy_preprocessor.json",{}).get("sha256")!=sha256(pre_data)):
        raise StateAssetError("Model revision/processor config does not match manifest")
    steps=json.loads(pre_data).get("steps",[])
    norm=[s.get("state_file") for s in steps if isinstance(s,dict)
          and s.get("registry_name")=="normalizer_processor"]
    if len(norm)!=1:
        raise StateAssetError("Expected single pretrained normalizer state")
    name=norm[0]
    if (not isinstance(name,str) or not name.endswith(".safetensors")
            or Path(name).name!=name or name.startswith(".")):
        raise StateAssetError("Unsafe or missing normalizer state filename")
    target=model_dir/name
    if target.exists() or target.is_symlink():
        raise StateAssetError("Refuse to overwrite existing state file: "+str(target))
    receipt_target=model_dir/"processor_state_receipt.json"
    if receipt_target.exists() or receipt_target.is_symlink():
        raise StateAssetError("Refuse to overwrite existing state receipt")
    uri=f"{endpoint}/lerobot/smolvla_libero/resolve/{model_dir.name}/{name}"
    return target,receipt_target,uri,receipt


def fetch_one(model_dir,endpoint,transport):
    target,receipt_target,uri,config_receipt=prepare_request(model_dir,endpoint)
    if transport=="urllib":
        blob=get_bounded(uri,max_bytes=MAX_STATE_BYTES)
    else:
        blob=get_bounded_curl_ipv4(
            uri,direct=transport=="curl-ipv4-direct",max_bytes=MAX_STATE_BYTES)
    if len(blob)<8:
        raise StateAssetError("Downloaded state file is too small")
    # safetensors begins with 8-byte little endian u64 header size, then JSON header
    head_size=int.from_bytes(blob[:8],"little")
    if head_size<=0 or head_size>len(blob)-8 or head_size>1048576:
        raise StateAssetError("Invalid safetensors header length")
    try:
        header=json.loads(blob[8:8+head_size])
    except (UnicodeDecodeError,json.JSONDecodeError) as exc:
        raise StateAssetError("Not a valid safetensors header") from exc
    if not isinstance(header,dict) or not any(k!="__metadata__" for k in header):
        raise StateAssetError("Safetensors header lacks tensor entries")
    digest=sha256(blob)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("xb") as file:
        file.write(blob)
    obj={
        "status":"PINNED_MODEL_NORMALIZER_SAFETENSORS_NOT_MODEL_WEIGHTS",
        "model_repo":"lerobot/smolvla_libero",
        "model_revision":config_receipt["revision"],
        "source_endpoint":endpoint,
        "source_is_third_party_mirror":endpoint!="https://huggingface.co",
        "state_file":target.name,"state_file_sha256":digest,"bytes":len(blob),
        "url":uri,
        "notes":"File digest recorded locally; mirror-provided bytes are not official independent verification.",
    }
    with receipt_target.open("x",encoding="utf-8") as file:
        json.dump(obj,file,ensure_ascii=False,indent=2)
        file.write("\n")
    return obj


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--endpoint",choices=ALLOWED_ENDPOINTS,default="https://huggingface.co")
    p.add_argument("--transport",choices=["urllib","curl-ipv4","curl-ipv4-direct"],default="urllib")
    p.add_argument("--allow-network",action="store_true")
    a=p.parse_args(argv)
    if not a.allow_network:
        p.exit(2,"STOP: explicit --allow-network required\n")
    try:
        print(json.dumps(fetch_one(a.model_dir,a.endpoint,a.transport),indent=2,ensure_ascii=False))
    except (ValueError,TypeError,KeyError,OSError,RuntimeError,json.JSONDecodeError) as exc:
        p.exit(2,"STOP: "+str(exc)+"\n")

if __name__=="__main__":
    main()
