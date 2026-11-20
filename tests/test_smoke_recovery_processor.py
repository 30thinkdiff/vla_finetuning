"""Offline no-Torch unit tests for Recovery to SmolVLA preprocessor gate."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from smoke_recovery_processor import (
    ProcessorSmokeError,
    validate_shapes,
    validate_model_files,
)


def sample_fixture():
    def arr(shape): return SimpleNamespace(shape=shape)
    return {
        "observation.images.image": arr((3,256,256)),
        "observation.images.image2": arr((3,256,256)),
        "observation.state": arr((8,)),
        "action": arr((50,7)),
        "action_is_pad": arr((50,)),
        "task": SimpleNamespace(shape=(),item=lambda: "move black bowl to drawer"),
    }


class SmokeGateTests(unittest.TestCase):
    def test_expected_shapes(self):
        s=sample_fixture()
        self.assertEqual(validate_shapes(s),"move black bowl to drawer")
        s["action"]=SimpleNamespace(shape=(50,7,1))
        with self.assertRaises(ProcessorSmokeError):
            validate_shapes(s)
        s=sample_fixture()
        s["task"]=SimpleNamespace(shape=(1,),item=lambda:"text")
        with self.assertRaises(ProcessorSmokeError):
            validate_shapes(s)

    def test_strict_model_normalizer_presence_and_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            model=root/("a"*40)
            model.mkdir()
            tok=root/("b"*40)
            tok.mkdir()
            (tok/"tokenizer_config.json").write_text("{}")
            blobs={"config.json":b'{"type":"smolvla"}',
                   "policy_preprocessor.json":json.dumps({
                       "steps":[{"registry_name":"normalizer_processor",
                                 "state_file":"policy_preprocessor_step_5_normalizer_processor.safetensors"}]
                   }).encode()}
            for filename,data in blobs.items():
                (model/filename).write_bytes(data)
            receipt={"model_repo":"lerobot/smolvla_libero","revision":model.name,
                     "files":{k:{"sha256":hashlib.sha256(v).hexdigest()} for k,v in blobs.items()}}
            (model/"receipt.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ProcessorSmokeError,"Missing pinned normalizer stats"):
                validate_model_files(model,tok)
            sf=model/"policy_preprocessor_step_5_normalizer_processor.safetensors"
            sf.write_bytes(b"dummy")
            (model/"processor_state_receipt.json").write_text(json.dumps({
                "model_revision":model.name,"state_file":sf.name,
                "state_file_sha256":hashlib.sha256(b"dummy").hexdigest()
            }))
            tblob=(tok/"tokenizer_config.json").read_bytes()
            (tok/"receipt.json").write_text(json.dumps({
                "repo_id":"HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                "revision":tok.name,
                "files":{"tokenizer_config.json":{"sha256":hashlib.sha256(tblob).hexdigest()}}
            }))
            _, _, found, _=validate_model_files(model,tok)
            self.assertEqual(found,sf)
            (model/"config.json").write_bytes(b"changed")
            with self.assertRaisesRegex(ProcessorSmokeError,"no longer matches pinned receipt"):
                validate_model_files(model,tok)

    def test_no_hub_repo_tokenizer_path(self):
        with self.assertRaises(ProcessorSmokeError):
            validate_model_files("/missing/path","HuggingFaceTB/SmolVLM2-500M-Video-Instruct")


if __name__=="__main__":
    unittest.main()
