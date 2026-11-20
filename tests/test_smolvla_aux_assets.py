"""No-network unit checks for pinned Recovery tokenizer and normalizer fetch gates."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from fetch_smolvla_tokenizer import select_files, fetch_tokenizer, TokenizerDownloadError
from fetch_smolvla_processor_state import prepare_request, StateAssetError


class AssetsTests(unittest.TestCase):
    def test_only_bounded_top_level_tokenizer_files(self):
        api={"sha":"f"*40,"siblings":[
            {"rfilename":"tokenizer.json"},
            {"rfilename":"tokenizer_config.json"},
            {"rfilename":"model.safetensors"},
            {"rfilename":"docs/tokenizer.json"},
            {"rfilename":"special_tokens_map.json"},
        ]}
        rev,files=select_files(api)
        self.assertEqual(rev,"f"*40)
        self.assertEqual(files,["special_tokens_map.json","tokenizer.json","tokenizer_config.json"])
        with self.assertRaises(TokenizerDownloadError):
            select_files({"sha":"main","siblings":api["siblings"]})
        with self.assertRaises(TokenizerDownloadError):
            select_files({"sha":"f"*40,"siblings":[{"rfilename":"pytorch_model.bin"}]})

    @patch("fetch_smolvla_tokenizer.get_bounded_curl_ipv4")
    def test_mirror_pins_and_never_fetches_weights(self,mock_get):
        def fake(uri,**kwargs):
            if "/api/models/" in uri:
                return json.dumps({"sha":"c"*40,"siblings":[
                    {"rfilename":"tokenizer.json"},
                    {"rfilename":"tokenizer_config.json"},
                    {"rfilename":"model.safetensors"}]}).encode()
            return b'{"hello":"world"}'
        mock_get.side_effect=fake
        with tempfile.TemporaryDirectory() as root:
            path,receipt=fetch_tokenizer(root,"https://hf-mirror.com","curl-ipv4")
            self.assertEqual(receipt["revision"],"c"*40)
            self.assertEqual(sorted(receipt["files"]),["tokenizer.json","tokenizer_config.json"])
            self.assertTrue(Path(path,"tokenizer.json").exists())
            self.assertTrue(all("model.safetensors" not in call.args[0] for call in mock_get.call_args_list))
            with self.assertRaises(TokenizerDownloadError):
                fetch_tokenizer(root,"https://hf-mirror.com","curl-ipv4")

    def test_normalizer_request_pinned_and_rejects_unsafe_filename(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/("a"*40)
            root.mkdir()
            pre={"steps":[{"registry_name":"normalizer_processor",
                            "state_file":"normalizer.safetensors"}]}
            raw=json.dumps(pre).encode()
            (root/"policy_preprocessor.json").write_bytes(raw)
            (root/"receipt.json").write_text(json.dumps({
                "model_repo":"lerobot/smolvla_libero","revision":root.name,
                "files":{"policy_preprocessor.json":{"sha256":hashlib.sha256(raw).hexdigest()}}}))
            target,rec,url,_=prepare_request(root,"https://hf-mirror.com")
            self.assertEqual(target,root/"normalizer.safetensors")
            self.assertIn("/resolve/"+root.name+"/normalizer.safetensors",url)
            pre["steps"][0]["state_file"]="../bad.safetensors"
            raw=json.dumps(pre).encode()
            (root/"policy_preprocessor.json").write_bytes(raw)
            rec_obj=json.loads((root/"receipt.json").read_text())
            rec_obj["files"]["policy_preprocessor.json"]["sha256"]=hashlib.sha256(raw).hexdigest()
            (root/"receipt.json").write_text(json.dumps(rec_obj))
            with self.assertRaises(StateAssetError):
                prepare_request(root,"https://hf-mirror.com")


if __name__=="__main__":
    unittest.main()
