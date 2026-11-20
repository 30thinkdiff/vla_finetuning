"""No-network tests for pinned tiny Hugging Face SmolVLA contract download."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from fetch_smolvla_contract_metadata import gather_contract, materialize, get_bounded


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


if __name__=="__main__":
    unittest.main()
