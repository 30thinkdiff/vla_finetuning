"""Pure-stdlib tests for the ModelScope recovery inventory/download planner."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from recovery_hub import inventory_from_remote, exact_manifest, safe_path, write_new


class FakeAPI:
    def list_repo_files(self, repo_id, repo_type, *, revision, recursive):
        assert repo_type == "dataset" and recursive and revision == "frozen-tag"
        return [SimpleNamespace(path="meta/info.json", size=112, sha256=None, type="blob"),
                SimpleNamespace(path="meta/episodes.jsonl", size=2024, sha256=None, type="blob"),
                SimpleNamespace(path="data/chunk-000/episode_000000.parquet", size=30000,
                                sha256="a"*64, type="blob")]


class RecoveryHubTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)/"inventory.json"
        self.inv = inventory_from_remote("ataier/LIBERO_Recovery_Expert", "frozen-tag", FakeAPI())

    def test_inventory_and_exact_manifest(self):
        self.assertEqual(len(self.inv["files"]), 3)
        result = exact_manifest(self.inv, ["meta/info.json", "meta/episodes.jsonl"], 1)
        self.assertEqual(result["total_bytes"], 2136)
        self.assertEqual(len(result["files"]), 2)
        self.assertEqual(result["status"], "PLAN_ONLY_NOT_DOWNLOADED")
        self.assertFalse(result["revision_sha40"])

    def test_never_assume_size(self):
        bad = json.loads(json.dumps(self.inv))
        bad["files"][0]["size_bytes"] = 0
        with self.assertRaises(ValueError):
            exact_manifest(bad, ["data/chunk-000/episode_000000.parquet"], 1)

    def test_reject_invalid_paths_missing_remote_and_budget(self):
        for path in ("../x", "/tmp/x", "data/../../x", "a\\b", "a:b"):
            with self.assertRaises(ValueError):
                safe_path(path)
        with self.assertRaises(ValueError):
            exact_manifest(self.inv, ["meta/does_not_exist.json"], 1)
        with self.assertRaises(ValueError):
            exact_manifest(self.inv, ["data/chunk-000/episode_000000.parquet"], 0.0000001)

    def test_defaults_must_not_be_pinned(self):
        for rev in ("master", "main", "latest"):
            with self.assertRaises(ValueError):
                inventory_from_remote("ataier/LIBERO_Recovery_Expert", rev, FakeAPI())

    def test_write_new_never_overwrites(self):
        write_new(self.path, self.inv)
        self.assertTrue(self.path.is_file())
        with self.assertRaises(FileExistsError):
            write_new(self.path, self.inv)


if __name__ == "__main__":
    unittest.main()
