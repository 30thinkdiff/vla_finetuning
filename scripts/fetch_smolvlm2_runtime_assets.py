"""Build local, complete SmolVLM2 *configuration + processor* directory.

The previously pinned tokenizer-only source omits 3 tiny files required by
upstream SmolVLMWithExpertModel.__init__ -> AutoProcessor.from_pretrained.
This script creates a NEW immutable directory; it does not mutate the original
tokenizer receipt or download VLM model weights. The VLM SHA is the recorded
tokenizer SHA, never the unrelated SmolVLA policy SHA.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from fetch_smolvla_contract_metadata import ALLOWED_ENDPOINTS,get_bounded,get_bounded_curl_ipv4

REPO="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
EXTRAS=("preprocessor_config.json","processor_config.json","chat_template.json")
MAX_JSON=1024*1024


class RuntimeAssetError(ValueError):
    pass


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prepare(source,out_root):
    source=Path(source).expanduser().resolve()
    old=source/"receipt.json"
    if not old.is_file():
        raise RuntimeAssetError("Original pinned tokenizer receipt missing")
    doc=json.loads(old.read_text(encoding="utf-8"))
    revision=doc.get("revision")
    if (doc.get("repo_id")!=REPO or revision!=source.name
            or not isinstance(revision,str) or not re.fullmatch("[0-9a-f]{40}",revision)):
        raise RuntimeAssetError("Tokenizer repo/revision mismatch")
    files=doc.get("files",{})
    if not isinstance(files,dict) or not files:
        raise RuntimeAssetError("Tokenizer manifest missing file hashes")
    for name,metadata in files.items():
        if Path(name).name!=name or name.endswith(".safetensors"):
            raise RuntimeAssetError("Unsafe pretrained tokenizer member")
        p=source/name
        if (not p.is_file() or p.is_symlink()
                or digest(p.read_bytes())!=metadata.get("sha256")):
            raise RuntimeAssetError("Tokenizer member missing/mismatched: "+name)
    dest=Path(out_root).expanduser().resolve()/revision
    if dest.exists() or dest.is_symlink():
        raise RuntimeAssetError("Refuse overwrite complete VLM runtime directory")
    return source,dest,revision,files


def materialize(source,out_root,endpoint,transport):
    if endpoint not in ALLOWED_ENDPOINTS:
        raise RuntimeAssetError("Unapproved Hub endpoint")
    source,dest,sha,files=prepare(source,out_root)
    new={}
    for name in EXTRAS:
        uri=f"{endpoint}/{REPO}/resolve/{sha}/{name}"
        if transport=="urllib":
            body=get_bounded(uri,max_bytes=MAX_JSON)
        else:
            body=get_bounded_curl_ipv4(uri,direct=transport=="curl-ipv4-direct",max_bytes=MAX_JSON)
        if not body:
            raise RuntimeAssetError("Missing required VLM processor config: "+name)
        try:
            value=json.loads(body)
        except (UnicodeDecodeError,json.JSONDecodeError) as exc:
            raise RuntimeAssetError("Malformed processor JSON: "+name) from exc
        if not isinstance(value,dict):
            raise RuntimeAssetError("Unexpected processor JSON body: "+name)
        new[name]={"body":body,"sha256":digest(body),"url":uri}
    dest.mkdir(parents=True,exist_ok=False)
    receipt={
        "status":"PINNED_SMOLVLM2_CONFIG_AND_PROCESSOR_NO_MODEL_WEIGHTS",
        "repo_id":REPO,"revision":sha,"source_endpoint":endpoint,
        "source_is_third_party_mirror":endpoint!="https://huggingface.co",
        "files":{},
    }
    for name in sorted(files):
        body=(source/name).read_bytes()
        with (dest/name).open("xb") as fp:
            fp.write(body)
        receipt["files"][name]={"sha256":digest(body),"bytes":len(body),"origin":"verified_local_tokenizer"}
    for name,entry in new.items():
        with (dest/name).open("xb") as fp:
            fp.write(entry["body"])
        receipt["files"][name]={"sha256":entry["sha256"],"bytes":len(entry["body"]),
                                "url":entry["url"],"origin":"pinned_mirror"}
    with (dest/"receipt.json").open("x",encoding="utf-8") as fp:
        json.dump(receipt,fp,indent=2,ensure_ascii=False)
        fp.write("\n")
    return str(dest),receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer-dir",type=Path,required=True)
    p.add_argument("--out-root",type=Path,required=True)
    p.add_argument("--endpoint",choices=ALLOWED_ENDPOINTS,default="https://huggingface.co")
    p.add_argument("--transport",choices=["urllib","curl-ipv4","curl-ipv4-direct"],default="urllib")
    p.add_argument("--allow-network",action="store_true")
    a=p.parse_args(argv)
    if not a.allow_network:
        p.exit(2,"STOP: explicit --allow-network approval required\n")
    try:
        path,doc=materialize(a.tokenizer_dir,a.out_root,a.endpoint,a.transport)
        print(json.dumps({"path":path,**doc},indent=2,ensure_ascii=False))
    except (RuntimeAssetError,OSError,ValueError,KeyError) as exc:
        p.exit(2,"STOP: "+str(exc)+"\n")


if __name__=="__main__":
    main()
