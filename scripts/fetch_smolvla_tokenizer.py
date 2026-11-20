"""Download only SmolVLA's VLM tokenizer files from a pinned Hub revision.

Does not download model weights. Uses the approved endpoint selected by the
user (Hugging Face or hf-mirror), captures the served VLM SHA, selected file
names and bytes SHA256. All downloads use HTTPS and an exclusive new directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from fetch_smolvla_contract_metadata import (
    ALLOWED_ENDPOINTS, get_bounded, get_bounded_curl_ipv4,
)


REPO="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SELECT={
    "tokenizer.json","tokenizer_config.json","special_tokens_map.json",
    "added_tokens.json","vocab.json","merges.txt","tokenizer.model",
    "sentencepiece.bpe.model","config.json",
}
MAX_EACH=32*1024*1024
MAX_TOTAL=60*1024*1024


class TokenizerDownloadError(ValueError):
    pass


def select_files(api):
    sha=api.get("sha")
    if not isinstance(sha,str) or not re.fullmatch("[0-9a-f]{40}",sha):
        raise TokenizerDownloadError("Tokenizer source has no pinned commit SHA")
    siblings=api.get("siblings")
    if not isinstance(siblings,list):
        raise TokenizerDownloadError("Tokenizer repo file listing missing")
    available=set()
    for item in siblings:
        if not isinstance(item,dict):
            continue
        name=item.get("rfilename")
        if isinstance(name,str) and name in SELECT:
            available.add(name)
    if "tokenizer_config.json" not in available or not (
            {"tokenizer.json","tokenizer.model","sentencepiece.bpe.model"} & available
            or {"vocab.json","merges.txt"} <= available):
        raise TokenizerDownloadError("Missing required text tokenizer files in VLM repository")
    return sha,sorted(available)


def fetch_tokenizer(root,endpoint,transport):
    if endpoint not in ALLOWED_ENDPOINTS:
        raise TokenizerDownloadError("Disallowed endpoint")
    def get(uri, max_bytes=2*1024*1024):
        if transport=="urllib":
            return get_bounded(uri,max_bytes=max_bytes)
        return get_bounded_curl_ipv4(
            uri,direct=transport=="curl-ipv4-direct",max_bytes=max_bytes)
    info=json.loads(get(f"{endpoint}/api/models/{REPO}"))
    sha,files=select_files(info)
    path=Path(root).expanduser().resolve()/sha
    if path.exists() or path.is_symlink():
        raise TokenizerDownloadError("Tokenizer version destination already exists; no overwrite: "+str(path))
    blobs={}
    total=0
    for name in files:
        uri=f"{endpoint}/{REPO}/resolve/{sha}/{name}"
        blob=get(uri,max_bytes=MAX_EACH)
        total+=len(blob)
        if total>MAX_TOTAL or not blob:
            raise TokenizerDownloadError("Tokenizer files exceed safety size limit or empty")
        blobs[name]={"url":uri,"data":blob,"sha256":hashlib.sha256(blob).hexdigest()}
    # Check JSON files are parseable before writing anything.
    for name in files:
        if name.endswith(".json"):
            try:
                value=json.loads(blobs[name]["data"])
            except (json.JSONDecodeError,UnicodeDecodeError) as exc:
                raise TokenizerDownloadError("Malformed tokenizer JSON: "+name) from exc
            if not isinstance(value,dict):
                raise TokenizerDownloadError("Tokenizer JSON must be an object: "+name)

    path.mkdir(parents=True,exist_ok=False)
    receipt={
        "status":"PINNED_TOKENIZER_ONLY_NO_MODEL_WEIGHTS",
        "repo_id":REPO,"revision":sha,"source_endpoint":endpoint,
        "source_is_third_party_mirror":endpoint!="https://huggingface.co",
        "note":"Third-party mirror SHA/bytes have not been independently verified against upstream",
        "files":{},
    }
    for name in files:
        item=blobs[name]
        with (path/name).open("xb") as fp:
            fp.write(item["data"])
        receipt["files"][name]={"url":item["url"],"sha256":item["sha256"],"bytes":len(item["data"])}
    with (path/"receipt.json").open("x",encoding="utf-8") as fp:
        json.dump(receipt,fp,ensure_ascii=False,indent=2)
        fp.write("\n")
    return str(path),receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-root",type=Path,required=True)
    p.add_argument("--endpoint",choices=ALLOWED_ENDPOINTS,default="https://huggingface.co")
    p.add_argument("--transport",choices=["urllib","curl-ipv4","curl-ipv4-direct"],default="urllib")
    p.add_argument("--allow-network",action="store_true")
    a=p.parse_args(argv)
    if not a.allow_network:
        p.exit(2,"STOP: explicit --allow-network required\n")
    try:
        path,receipt=fetch_tokenizer(a.out_root,a.endpoint,a.transport)
        print(json.dumps({"path":path,**receipt},ensure_ascii=False,indent=2))
    except (ValueError,TypeError,KeyError,OSError,RuntimeError,json.JSONDecodeError) as exc:
        p.exit(2,"STOP: "+str(exc)+"\n")


if __name__=="__main__":
    main()
