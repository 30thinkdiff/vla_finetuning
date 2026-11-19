import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vla_finetuning.core import (ContractError, audit_lerobot_info, preflight,
                                 split_provenance_csv, training_plan, write_new_json)
from vla_finetuning.flow_math import two_number_example


ROOT = Path(__file__).resolve().parents[1]


class MetadataTests(unittest.TestCase):
    def test_v3_metadata_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "info.json"
            p.write_text(json.dumps({"codebase_version": "v3.0", "fps": 20,
                "total_episodes": 8, "total_frames": 123, "total_tasks": 2,
                "features": {"observation.state": {"dtype": "float32", "shape": [8]},
                             "action": {"dtype": "float32", "shape": [7]},
                             "observation.images.front": {"dtype": "video", "shape": [256,256,3]},
                             "observation.images.wrist": {"dtype": "video", "shape": [256,256,3]}}}), encoding="utf-8")
            result = audit_lerobot_info(p)
            self.assertEqual(result["fps"], 20)
            self.assertEqual(result["features"]["action"]["shape"], [7])
            self.assertEqual(len(result["image_keys"]), 2)
            self.assertEqual(result["schema_issues"], [])
            self.assertIn("does NOT identify", result["provenance_warning"])

    def test_missing_feature_is_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "info.json"
            p.write_text('{"features": {"action": {"shape": [7]}}}')
            self.assertTrue(audit_lerobot_info(p)["schema_issues"])


class SplitTests(unittest.TestCase):
    def test_group_split_is_disjoint_and_repeatable(self):
        p = ROOT / "examples" / "provenance.synthetic.csv"
        a = split_provenance_csv(p, seed=42)
        b = split_provenance_csv(p, seed=42)
        self.assertEqual(a["splits"], b["splits"])
        by_split = [{v["source_group"] for v in a["splits"][k]}
                    for k in ("train", "validation", "test")]
        for i in range(3):
            for j in range(i+1, 3):
                self.assertFalse(by_split[i] & by_split[j])
        # The same underlying source in 2 camera views must never cross splits.
        view_records = [v for split in a["splits"].values()
                        for v in split if v["source_group"] == "soup_orig_01"]
        self.assertEqual(len(view_records), 2)
        self.assertEqual(sum("soup_orig_01" in values for values in by_split), 1)

    def test_missing_source_group_stops(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "source.csv"
            p.write_text("dataset_id,episode_id,source_group,task_id\nx,1,,soup\n")
            with self.assertRaises(ContractError):
                split_provenance_csv(p)

    def test_overwrite_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "report.json"
            write_new_json(p, {"ok": True})
            with self.assertRaises(FileExistsError):
                write_new_json(p, {"ok": False})


class PreparationTests(unittest.TestCase):
    def test_template_is_blocked(self):
        result = training_plan(ROOT / "configs" / "experiment.template.json")
        self.assertEqual(result["status"], "BLOCKED")
        self.assertIsNone(result["command"])
        self.assertGreaterEqual(len(result["reasons"]), 2)

    def test_training_plan_requires_consistent_source_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            manifest = root / "split.json"
            splits = {"splits": {
                "train": [{"dataset_id": "org/demo", "episode_id": "1", "source_group": "g1"}],
                "validation": [{"dataset_id": "org/demo", "episode_id": "2", "source_group": "g2"}],
                "test": [{"dataset_id": "org/demo", "episode_id": "3", "source_group": "g3"}]}}
            manifest.write_text(json.dumps(splits), encoding="utf-8")
            cfg = {
                "lerobot": {"source_revision": "a" * 40},
                "model": {"model_revision": "b" * 40, "repo_id": "org/model",
                          "policy_path": str(root / "model")},
                "data": {"repo_id": "org/demo", "dataset_revision": "c" * 40,
                         "dataset_root": str(root / "dataset"),
                         "episode_indices": [1],
                         "train_split_manifest": str(manifest)},
                "run": {"output_dir": str(root / "run"), "steps": 50,
                        "batch_size": 1, "save_freq": 50, "peft_method": "LORA",
                        "lora_rank": 8, "lora_alpha": 8}
            }
            config = root / "config.json"
            config.write_text(json.dumps(cfg), encoding="utf-8")
            result = training_plan(config)
            self.assertTrue(result["status"].startswith("READY_FOR_MANUAL"))
            self.assertTrue(any("--dataset.episodes=[1]" in v for v in result["command"]))
            cfg["data"]["episode_indices"] = [2]
            config.write_text(json.dumps(cfg), encoding="utf-8")
            self.assertEqual(training_plan(config)["status"], "BLOCKED")
            self.assertIn("manifest", " ".join(training_plan(config)["reasons"]))

    def test_preflight_is_read_only(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(preflight(d)["status"], "read_only")

    def test_flow_example(self):
        d = two_number_example()
        self.assertAlmostEqual(d["noisy_action"][0], 0.375)
        self.assertAlmostEqual(d["target_velocity"][0], 0.7)
        self.assertAlmostEqual(d["mean_squared_error"], 0.01)


if __name__ == "__main__":
    unittest.main()
