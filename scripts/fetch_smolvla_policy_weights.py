"""Download exact pinned LIBERO SmolVLA weights with SHA256 verification.

Explicit large-download opt-in. Only lerobot/smolvla_libero and its already
pinned local revision are supported. Uses curl HTTPS GET with resumable .part.
No model execution, no Hugging Face credential handling. Source mirror is a
third party; SHA256 is cross-checked against published official HF file page.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

from fetch_smolvla_contract_metadata import ALLOWED_ENDPOINTS


MODEL = "lerobot/smolvla_libero"
REVISION = "a24f1f380461b665158d52acd140152c79e8b36d"
# Public pinned checkpoint's model.safetensors SHA256 reported on HF file page.
WEIGHTS_SHA256 = "9a9f6413e42c0f332fccbce9a0dc796af2790f82cf002f791cdbf7e01e1afca8"
MAX_WEIGHTS_BYTES = 1_500_000_000
MIN_WEIGHTS_BYTES = 700_000_000


class ModelWeightError(ValueError):
    pass


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fp:
        for block in iter(lambda: fp.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def precheck(model_dir, endpoint):
    model_dir = Path(model_dir).expanduser().resolve()
    if endpoint not in ALLOWED_ENDPOINTS:
        raise ModelWeightError("Unapproved model endpoint")
    receipt = model_dir/"receipt.json"
    config = model_dir/"config.json"
    if not receipt.is_file() or not config.is_file():
        raise ModelWeightError("Missing original pinned model JSON/receipt")
    desc = json.loads(receipt.read_text(encoding="utf-8"))
    if (desc.get("model_repo") != MODEL or desc.get("revision") != REVISION
            or model_dir.name != REVISION or
            desc.get("files",{}).get("config.json",{}).get("sha256") != sha256_file(config)):
        raise ModelWeightError("Model metadata SHA/revision mismatch")
    dest = model_dir/"model.safetensors"
    part = model_dir/"model.safetensors.part"
    rec = model_dir/"policy_weights_receipt.json"
    for p in (dest,part,rec):
        if p.is_symlink():
            raise ModelWeightError("Symlink target refused: "+str(p))
    if rec.exists():
        raise ModelWeightError("Weight receipt already exists: refuse ambiguous second download")
    # Download and unpack must fit on the same filesystem used by the model.
    free = shutil.disk_usage(model_dir).free
    remaining = max(0,MAX_WEIGHTS_BYTES - (part.stat().st_size if part.exists() else 0))
    if free < remaining + 2_000_000_000:
        raise ModelWeightError("Insufficient free space for checkpoint and working reserve")
    return model_dir,dest,part,rec


def verify_weight_bytes(file):
    size=Path(file).stat().st_size
    if not MIN_WEIGHTS_BYTES <= size <= MAX_WEIGHTS_BYTES:
        raise ModelWeightError(f"Unexpected checkpoint size: {size} bytes")
    digest=sha256_file(file)
    if digest != WEIGHTS_SHA256:
        raise ModelWeightError("SHA256 mismatch for model.safetensors; keep .part and inspect source")
    return size,digest


def fetch(model_dir,endpoint,*,direct=False):
    model_dir,dest,part,rec=precheck(model_dir,endpoint)
    if dest.exists():
        # Already-complete checkpoint with missing receipt: verify, never redownload.
        size,digest=verify_weight_bytes(dest)
    else:
        if not shutil.which("curl"):
            raise ModelWeightError("curl not installed")
        if part.exists() and part.stat().st_size > MAX_WEIGHTS_BYTES:
            raise ModelWeightError("Oversize existing partial download")
        url=f"{endpoint}/{MODEL}/resolve/{REVISION}/model.safetensors"
        args=[
            "curl","--ipv4","--fail","--location","--silent","--show-error",
            "--proto","=https","--proto-redir","=https","--max-redirs","5",
            "--connect-timeout","15","--retry","3","--retry-delay","5",
            "--max-time","3600","--max-filesize",str(MAX_WEIGHTS_BYTES),
            "--continue-at","-","--output",str(part),
        ]
        if direct:
            args+=["--noproxy","*"]
        args.append(url)
        try:
            subprocess.run(args,check=True,timeout=3700)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as exc:
            raise OSError("Checkpoint transfer failed; partial bytes retained for resume") from exc
        size,digest=verify_weight_bytes(part)
        if dest.exists():
            raise ModelWeightError("Destination unexpectedly exists; do not overwrite")
        os.rename(part,dest)
    receipt = {
        "status":"PINNED_LIBERO_SMOLVLA_POLICY_WEIGHTS_VERIFIED_NOT_TRAINED",
        "model_repo":MODEL,"model_revision":REVISION,
        "filename":"model.safetensors","sha256":digest,"size_bytes":size,
        "expected_sha256_source":"https://huggingface.co/lerobot/smolvla_libero/blob/main/model.safetensors",
        "source_endpoint":endpoint,"third_party_mirror":endpoint!="https://huggingface.co",
        "note":"Digest matches published upstream file SHA; not a signature. No policy loaded/tested.",
    }
    with rec.open("x",encoding="utf-8") as fp:
        json.dump(receipt,fp,indent=2,ensure_ascii=False)
        fp.write("\n")
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--endpoint",choices=ALLOWED_ENDPOINTS,default="https://huggingface.co")
    parser.add_argument("--direct",action="store_true",help="Bypass proxy for curl IPv4")
    parser.add_argument("--allow-large-download",action="store_true",help="Explicitly approve ~907MB download")
    args=parser.parse_args(argv)
    if not args.allow_large_download:
        parser.exit(2,"STOP: explicit --allow-large-download approval required\n")
    try:
        result=fetch(args.model_dir,args.endpoint,direct=args.direct)
        print(json.dumps(result,indent=2,ensure_ascii=False))
    except (ModelWeightError,OSError,ValueError,KeyError) as exc:
        parser.exit(2,"STOP: "+str(exc)+"\n")


if __name__=="__main__":
    main()
