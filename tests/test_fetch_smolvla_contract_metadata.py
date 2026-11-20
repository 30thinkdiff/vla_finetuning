"""No-network tests for pinned tiny Hugging Face SmolVLA contract download."""
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from fetch_smolvla_contract_metadata import gather_contract, materialize, get_bounded, get_bounded_curl_ipv4


class FakeResponse:
    headers = {}
    def __init__(self, data): self.data = data
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def read(self,n): return self.data[:n]


class FetchSmolVLATests(unittest.TestCase):
    def test_real_revision_used_for_json(self):
        sha="a"*40
        calls=[]
        def opener(request,timeout=None):
            url=request.full_url; calls.append(url)
            if url.endswith("/api/models/lerobot/smolvla_libero"):
                return FakeResponse(json.dumps({"sha":sha}).encode())
            if f"/resolve/{sha}/" not in url:
                raise AssertionError("Unpinned model JSON request")
            return FakeResponse(b'{"test": 1}')
        rev, files = gather_contract(opener=opener)
        self.assertEqual(rev,sha)
        self.assertEqual(len(calls),3)
        with tempfile.TemporaryDirectory() as tmp:
            dest, receipt=materialize(tmp,rev,files)
            self.assertEqual(json.loads((dest/"config.json").read_text()), {"test":1})
            self.assertEqual(receipt["status"],"PINNED_MODEL_JSON_ONLY_NOT_MODEL_WEIGHTS")
            with self.assertRaises(ValueError):
                materialize(tmp,rev,files)

    def test_disallow_invalid_sha_and_unbounded_json(self):
        def opener(request,timeout=None):
            return FakeResponse(b'{"sha":"master"}')
        with self.assertRaises(ValueError):
            gather_contract(opener=opener)
        with self.assertRaises(ValueError):
            get_bounded("https://example.test",opener=lambda *args,**kwargs:FakeResponse(b"abcdef"),max_bytes=5)


    def test_mirror_endpoint_pinned_and_receipted(self):
        sha = "b"*40
        urls = []
        def fake_getter(url):
            urls.append(url)
            if url.endswith("/api/models/lerobot/smolvla_libero"):
                return json.dumps({"sha": sha}).encode()
            return b'{"ok":true}'
        revision, files = gather_contract(getter=fake_getter, endpoint="https://hf-mirror.com")
        self.assertEqual(revision, sha)
        self.assertTrue(all(u.startswith("https://hf-mirror.com/") for u in urls))
        self.assertEqual(len(urls), 3)
        with tempfile.TemporaryDirectory() as temp:
            dest, receipt = materialize(temp, revision, files, endpoint="https://hf-mirror.com")
            self.assertTrue(receipt["source_is_third_party_mirror"])
            self.assertEqual(receipt["source_endpoint"], "https://hf-mirror.com")
            self.assertTrue((dest/"receipt.json").exists())
        with self.assertRaises(ValueError):
            gather_contract(getter=fake_getter, endpoint="http://hf-mirror.com")
        with self.assertRaises(ValueError):
            gather_contract(getter=fake_getter, endpoint="https://attacker.example")

    @patch("fetch_smolvla_contract_metadata.shutil.which", return_value="/usr/bin/curl")
    @patch("fetch_smolvla_contract_metadata.subprocess.run")
    def test_curl_ipv4_uses_safe_http_options_and_proxy_modes(self, run, which):
        run.return_value = SimpleNamespace(stdout=b'{"sha":"'+b'a'*40+b'"}')
        data = get_bounded_curl_ipv4(
            "https://huggingface.co/api/models/lerobot/smolvla_libero")
        self.assertIn(b"a"*40, data)
        args = run.call_args.args[0]
        self.assertIn("--ipv4", args)
        self.assertIn("--fail", args)
        self.assertNotIn("--noproxy", args)
        get_bounded_curl_ipv4(
            "https://huggingface.co/api/models/lerobot/smolvla_libero", direct=True)
        args = run.call_args.args[0]
        self.assertEqual(args[args.index("--noproxy")+1], "*")

    @patch("fetch_smolvla_contract_metadata.shutil.which", return_value="/usr/bin/curl")
    @patch("fetch_smolvla_contract_metadata.subprocess.run")
    def test_curl_ipv4_rejects_wrong_host_and_oversize(self, run, which):
        with self.assertRaises(ValueError):
            get_bounded_curl_ipv4("https://untrusted.example/secret")
        run.return_value = SimpleNamespace(stdout=b"x"*7)
        with self.assertRaises(ValueError):
            get_bounded_curl_ipv4(
                "https://huggingface.co/api/models/lerobot/smolvla_libero",
                max_bytes=6)


if __name__=="__main__":
    unittest.main()
