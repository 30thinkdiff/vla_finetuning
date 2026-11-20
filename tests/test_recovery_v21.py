"""v2.1 Expert source metadata tests, using synthetic local fixtures only."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_finetuning.core import ContractError
from vla_finetuning.recovery_v21 import plan_v21_expert_files


class RecoveryV21Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root/"meta").mkdir()
        self.info = {
            "codebase_version": "v2.1",
            "fps": 20,
            "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.images.front": {"dtype": "video", "shape": [256,256,3]},
                "observation.images.wrist": {"dtype": "video", "shape": [256,256,3]},
                "observation.state": {"dtype": "float32", "shape": [8]},
                "action": {"dtype": "float32", "shape": [7]},
            },
        }
        self.episodes = [
            {"episode_index": 0, "length": 150, "tasks": ["task_a"]},
            {"episode_index": 1000, "length": 91, "tasks": ["task_b"]},
        ]
        self.write()

    def write(self):
        (self.root/"meta/info.json").write_text(json.dumps(self.info),encoding="utf-8")
        (self.root/"meta/episodes.jsonl").write_text(
            "\n".join(json.dumps(row) for row in self.episodes)+"\n",encoding="utf-8")

    def test_real_format_shape_plan_not_training(self):
        result=plan_v21_expert_files(self.root,[1000,0])
        self.assertEqual(result["observed_fps"],20)
        self.assertEqual(result["payload_file_count"],6)
        self.assertEqual(result["episode_files"]["1000"]["data_file"],
                         "data/chunk-001/episode_001000.parquet")
        self.assertEqual(result["episode_files"]["0"]["video_files"]["observation.images.wrist"],
                         "videos/chunk-000/observation.images.wrist/episode_000000.mp4")
        self.assertFalse(result["semantic_checks"]["gripper_values_verified"])
        self.assertEqual(result["status"],"V21_FILE_PLAN_ONLY_NOT_DOWNLOADED")
        self.assertFalse((self.root/"data").exists())

    def test_reject_mismatched_revision_layout(self):
        self.info["codebase_version"]="v3.0"
        self.write()
        with self.assertRaises(ContractError):
            plan_v21_expert_files(self.root,[0])

    def test_reject_path_traversal_and_unknown_placeholders(self):
        self.info["video_path"]="../escape/{video_key}/episode_{episode_index:06d}.mp4"
        self.write()
        with self.assertRaises(ContractError):
            plan_v21_expert_files(self.root,[0])
        self.info["video_path"]="videos/{video_key}/{private_key}.mp4"
        self.write()
        with self.assertRaises(ContractError):
            plan_v21_expert_files(self.root,[0])

    def test_reject_duplicate_episode_and_missing_index(self):
        self.episodes.append(dict(self.episodes[0]))
        self.write()
        with self.assertRaises(ContractError):
            plan_v21_expert_files(self.root,[0])
        self.episodes.pop()
        self.write()
        with self.assertRaises(ContractError):
            plan_v21_expert_files(self.root,[9])

    def test_require_observed_chunk_size(self):
        self.info.pop("chunks_size")
        self.write()
        with self.assertRaises(ContractError):
            plan_v21_expert_files(self.root,[0])


if __name__=="__main__":
    unittest.main()
