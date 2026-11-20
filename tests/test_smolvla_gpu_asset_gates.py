"""Offline tests for pinned SmolVLA large-weight and full VLM runtime safety."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from fetch_smolvla_policy_weights import precheck,ModelWeightError,REVISION
from fetch_smolvlm2_runtime_assets import prepare,materialize,RuntimeAssetError
from smoke_recovery_lora_gpu import validate_assets,GPUSmokeError


def sha(data):
    return hashlib.sha256(data).hexdigest()


class ModelWeightsTests(unittest.TestCase):
    def test_requires_pinned_model_rev_and_original_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/REVISION
            root.mkdir()
            config=b'{"type":"smolvla"}'
            (root/"config.json").write_bytes(config)
            (root/"receipt.json").write_text(json.dumps({
                "model_repo":"lerobot/smolvla_libero",
                "revision":REVISION,"files":{"config.json":{"sha256":sha(config)}}
            }))
            _,dest,part,receipt=precheck(root,"https://hf-mirror.com")
            self.assertEqual(dest,root/"model.safetensors")
            self.assertFalse(receipt.exists())
            with self.assertRaises(ModelWeightError):
                precheck(root,"http://hf-mirror.com")
            (root/"config.json").write_bytes(b"modified")
            with self.assertRaisesRegex(ModelWeightError,"mismatch"):
                precheck(root,"https://hf-mirror.com")

    def test_gpu_gated_without_weight_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=Path(tmp)/REVISION
            model.mkdir()
            with self.assertRaisesRegex(GPUSmokeError,"Missing pinned policy weights"):
                validate_assets(model,Path(tmp)/"runtime")


class VLMRuntimeTests(unittest.TestCase):
    @patch("fetch_smolvlm2_runtime_assets.get_bounded_curl_ipv4")
    def test_new_immutable_runtime_contains_only_pinned_small_files(self,get):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            original=root/("a"*40)
            original.mkdir()
            for name,body in (("config.json",b'{"model_type":"smolvlm"}'),
                              ("tokenizer_config.json",b'{"test":true}'),
                              ("tokenizer.json",b'{"test":true}')):
                (original/name).write_bytes(body)
            receipt={
                "repo_id":"HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                "revision":original.name,
                "files":{name:{"sha256":sha((original/name).read_bytes())}
                         for name in ("config.json","tokenizer_config.json","tokenizer.json")}
            }
            (original/"receipt.json").write_text(json.dumps(receipt))
            get.return_value=b'{"processor_class":"SmolVLMProcessor"}'
            path,doc=materialize(original,root/"runtime","https://hf-mirror.com","curl-ipv4")
            self.assertTrue((Path(path)/"preprocessor_config.json").exists())
            self.assertTrue((Path(path)/"processor_config.json").exists())
            self.assertTrue((Path(path)/"chat_template.json").exists())
            self.assertTrue((Path(path)/"tokenizer.json").exists())
            self.assertFalse((Path(path)/"model.safetensors").exists())
            self.assertEqual(doc["revision"],original.name)
            self.assertEqual(len(get.call_args_list),3)
            self.assertTrue(all("/resolve/"+original.name+"/" in call.args[0]
                                for call in get.call_args_list))
            with self.assertRaises(RuntimeAssetError):
                materialize(original,root/"runtime","https://hf-mirror.com","curl-ipv4")
            _, _, rev, files = prepare(original,root/"another")
            self.assertEqual(rev,original.name)


if __name__=="__main__":
    unittest.main()
