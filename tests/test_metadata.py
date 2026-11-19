import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_finetuning.core import ContractError
from vla_finetuning.metadata import fetch_metadata, inspect_layout, resolve_revision, bounded_read


class FakeResponse(io.BytesIO):
    def __init__(self, content, headers=None):
        super().__init__(content)
        self.headers = headers or {}


def fake_opener(blob, headers=None):
    def open_request(_request, timeout):
        assert timeout == 15
        return FakeResponse(blob, headers)
    return open_request


class MetadataNetworkGates(unittest.TestCase):
    def test_cannot_access_network_without_authorization(self):
        with self.assertRaises(ContractError):
            resolve_revision("lerobot/libero_plus")
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ContractError):
                fetch_metadata("lerobot/libero_plus", "a"*40,
                               "meta/info.json", d)
    def test_pinned_revision_response(self):
        result = resolve_revision("lerobot/libero_plus", allow_network=True,
            opener=fake_opener(json.dumps({"sha": "a"*40}).encode()))
        self.assertEqual(result["revision"], "a"*40)
    def test_reject_traversal_and_data_file(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ContractError):
                fetch_metadata("../secret", "a"*40, "meta/info.json", d, allow_network=True)
            with self.assertRaises(ContractError):
                fetch_metadata("lerobot/libero_plus", "a"*40,
                    "data/chunk-000/file-000.parquet", d, allow_network=True)
    def test_one_small_json_file_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            fake_info = json.dumps({"codebase_version": "v3.0", "features": {}, "fps": 20}).encode()
            args = ("lerobot/libero_plus", "a"*40, "meta/info.json", d)
            out = fetch_metadata(*args, allow_network=True, opener=fake_opener(fake_info))
            self.assertEqual(out["bytes"], len(fake_info))
            self.assertFalse(out["training_data_downloaded"])
            self.assertTrue(Path(out["path"]).is_file())
            with self.assertRaises(ContractError):
                fetch_metadata(*args, allow_network=True, opener=fake_opener(fake_info))
    def test_enforced_actual_streamed_size(self):
        with self.assertRaises(ContractError):
            bounded_read("https://example.invalid/metadata", max_bytes=1024,
                         opener=fake_opener(b"x"*1025))


class MetadataLocalInspection(unittest.TestCase):
    def test_recognizes_v3_parquet_tasks_without_pyarrow(self):
        with tempfile.TemporaryDirectory() as d:
            meta = Path(d) / "meta"
            meta.mkdir()
            (meta / "info.json").write_text(json.dumps({
                "codebase_version": "v3.0", "fps": 20, "total_episodes": 14347,
                "total_frames": 2238036, "total_tasks": 40,
                "features": {"action": {"dtype": "float32", "shape": [7]},
                             "observation.state": {"dtype": "float32", "shape": [8]},
                             "observation.images.front": {"dtype": "video", "shape": [256,256,3]},
                             "observation.images.wrist": {"dtype": "video", "shape": [256,256,3]}}
            }), encoding="utf-8")
            (meta / "tasks.parquet").write_bytes(b"PAR1dummyPAR1")
            result = inspect_layout(d)
            self.assertEqual(result["tasks_index"], "tasks.parquet")
            self.assertIsNone(result["episodes_index"])
            self.assertFalse(result["safe_to_train"])
            self.assertEqual(result["meta"]["fps"], 20)
    def test_sample_rows_limited(self):
        with tempfile.TemporaryDirectory() as d:
            meta = Path(d) / "meta"
            meta.mkdir()
            (meta / "info.json").write_text('{"features": {}, "fps":20}', encoding="utf-8")
            with self.assertRaises(ContractError):
                inspect_layout(d, sample_rows=11)


if __name__ == "__main__":
    unittest.main()
