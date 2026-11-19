"""Synthetic LeRobot-v3 Episode shard resolver tests, without PyArrow install."""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_finetuning.core import ContractError
from vla_finetuning.dataset_files import plan_v3_episode_shards


def rows():
    result = []
    for ep, file_idx in ((0, 2), (1, 2), (2, 5)):
        data = {"episode_index":ep, "data/chunk_index":0,
                "data/file_index":file_idx}
        for key in ("observation.images.front", "observation.images.wrist"):
            data[f"videos/{key}/chunk_index"] = 0
            data[f"videos/{key}/file_index"] = file_idx + 1
        result.append(data)
    return result


class MockTable:
    def __init__(self, items):
        self._items=items
        self.schema=types.SimpleNamespace(names=["episode_index", "data/chunk_index",
                                                   "data/file_index"])
    def to_pylist(self):
        return self._items


class V3Shards(unittest.TestCase):
    def _root(self, path, data_path=None):
        root=Path(path)
        meta=root/"meta"
        shard=meta/"episodes"/"chunk-000"
        shard.mkdir(parents=True)
        (shard/"file-000.parquet").write_bytes(b"test")
        info={
            "codebase_version":"v3.0",
            "data_path":data_path or "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path":"videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features":{
                "observation.images.front":{"dtype":"video", "shape":[256,256,3]},
                "observation.images.wrist":{"dtype":"video", "shape":[256,256,3]},
                "observation.state":{"dtype":"float32", "shape":[8]},
                "action":{"dtype":"float32", "shape":[7]},
            }
        }
        (meta/"info.json").write_text(json.dumps(info))
        return root

    def _with_arrow_stub(self, items):
        pyarrow=types.ModuleType("pyarrow")
        pyarrow.__path__=[]
        parquet=types.ModuleType("pyarrow.parquet")
        parquet.read_table=lambda _path: MockTable(items)
        pyarrow.parquet=parquet
        return patch.dict(sys.modules, {"pyarrow":pyarrow, "pyarrow.parquet":parquet})

    def test_deduplicates_physical_shards_for_two_episodes(self):
        with tempfile.TemporaryDirectory() as d:
            root=self._root(d)
            with self._with_arrow_stub(rows()):
                plan=plan_v3_episode_shards(root,[0,1])
            # Both selected episodes share the same data and 2 camera video shards.
            self.assertEqual(plan["payload_files_count"],3)
            self.assertIn("data/chunk-000/file-002.parquet",plan["payload_files"])
            self.assertIn("videos/observation.images.front/chunk-000/file-003.mp4",
                          plan["payload_files"])
            self.assertFalse(plan["camera_perturbation_provenance_verified"])

    def test_unavailable_episode_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root=self._root(d)
            with self._with_arrow_stub(rows()):
                with self.assertRaises(ContractError):
                    plan_v3_episode_shards(root,[8])

    def test_unsafe_path_template_fails(self):
        with tempfile.TemporaryDirectory() as d:
            root=self._root(d,data_path="../../outside/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
            with self._with_arrow_stub(rows()):
                with self.assertRaises(ContractError):
                    plan_v3_episode_shards(root,[0])


if __name__=="__main__":
    unittest.main()
