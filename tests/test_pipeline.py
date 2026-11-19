"""Pure CPU, no network/GPU tests for integration and failure gates."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_finetuning.core import ContractError
from vla_finetuning.reference import extract_task_labels, make_action_chunk, paired_outcomes
from vla_finetuning.reports import analyze_results
from vla_finetuning.guards import require_sha, compute_pids, require_gpu_idle, load_protocol
from vla_finetuning.artifacts import file_plan
from vla_finetuning.dataset_files import train_episodes_from_split
from vla_finetuning.engine import train_plan, _train_argv


class ReferenceTests(unittest.TestCase):
    def test_task_labels(self):
        self.assertEqual(extract_task_labels({"tasks": ["pick", "", "pick", "place"]}),
                         ["pick", "place"])
        self.assertEqual(extract_task_labels({"task_index": 3}), ["task_index:3"])
        with self.assertRaises(ValueError):
            extract_task_labels({"episode_index": 4})

    def test_action_padding_and_validation(self):
        acts = [[float(t)]*7 for t in range(3)]
        chunk, mask = make_action_chunk(acts, 1, 4)
        self.assertEqual(mask, [False, False, True, True])
        self.assertEqual(chunk[-1], [2.0]*7)
        self.assertIsNot(chunk[-1], chunk[-2])
        with self.assertRaises(ValueError):
            make_action_chunk(acts, 3, 4)
        with self.assertRaises(ValueError):
            make_action_chunk([[1.0]*6], 0, 1)

    def test_paired_outcomes(self):
        rows = []
        for condition, baseline, candidate in (("A", False, True), ("B", True, False),
                                               ("C", True, True), ("D", False, False)):
            for method, success in (("B0",baseline),("L1",candidate)):
                rows.append({"domain":"plus", "condition_id":condition, "seed":1000,
                             "method":method, "success":success})
        self.assertEqual(paired_outcomes(rows),
                         {"n_pairs":4, "gained":1, "lost":1, "both_success":1, "both_failed":1})
        with self.assertRaises(ValueError):
            paired_outcomes(rows + [rows[0]])
        with self.assertRaises(ValueError):
            paired_outcomes(rows[:-1])

    def test_real_report_is_descriptive(self):
        path = Path(__file__).resolve().parents[1] / "examples" / "paired_results.synthetic.csv"
        report = analyze_results(path)
        self.assertEqual(report["overall"]["n_pairs"], 4)
        self.assertEqual(report["domains"]["SYNTHETIC"]["delta_percentage_points"], 0)


class GuardTests(unittest.TestCase):
    def test_sha_and_placeholder(self):
        self.assertEqual(require_sha("0123456789abcdef"*2+"01234567", "x"),
                         "0123456789abcdef"*2+"01234567")
        with self.assertRaises(ContractError):
            require_sha("a"*40, "placeholder")
        with self.assertRaises(ContractError):
            require_sha("main", "branch")

    def test_busy_gpu_guard_mock(self):
        def fake_busy(*args, **kwargs):
            return SimpleNamespace(stdout="109584\n127260\n")
        def fake_idle(*args, **kwargs):
            return SimpleNamespace(stdout="")
        self.assertEqual(compute_pids(runner=fake_busy), [109584, 127260])
        self.assertEqual(compute_pids(runner=fake_idle), [])
        with self.assertRaises(ContractError):
            require_gpu_idle(runner=fake_busy)

    def test_artifact_size_limits(self):
        inventory = {"meta/info.json": 120, "meta/tasks.parquet": 321,
                     "data/chunk-000/file-000.parquet": 42,
                     "videos/observation.images.front/chunk-000/file-000.mp4": 880}
        sha = "0123456789abcdef"*2+"01234567"
        a = file_plan("lerobot/libero_plus", sha, kind="dataset_metadata",
                      inventory=inventory)
        self.assertEqual(a["total_bytes"], 441)
        self.assertEqual(a["status"], "PLANNED_ONLY_NO_DOWNLOAD")
        with self.assertRaises(ContractError):
            file_plan("lerobot/libero_plus", sha, kind="dataset_payload",
                      selected_paths=["data/not_present.parquet"], inventory=inventory)
        with self.assertRaises(ContractError):
            file_plan("lerobot/libero_plus", sha, kind="dataset_payload",
                      selected_paths=["README.md"], inventory=inventory)
        bad = dict(inventory)
        bad["meta/tasks.parquet"] = None
        with self.assertRaises(ContractError):
            file_plan("lerobot/libero_plus", sha, kind="dataset_metadata",
                      inventory=bad)


class FrozenPlanTests(unittest.TestCase):
    def _config(self, directory):
        p = Path(directory)
        sha = "0123456789abcdef"*2+"01234567"
        split = p/"split.json"
        data = {"splits": {
            "train":[{"dataset_id":"lerobot/libero_plus","episode_id":"2","source_group":"g1",
                      "source_evidence":"asset/a","review_status":"verified"}],
            "validation":[{"dataset_id":"lerobot/libero_plus","episode_id":"3","source_group":"g2",
                      "source_evidence":"asset/b","review_status":"verified"}],
            "test":[{"dataset_id":"lerobot/libero_plus","episode_id":"4","source_group":"g3",
                      "source_evidence":"asset/c","review_status":"verified"}]}}
        split.write_text(json.dumps(data))
        cfg = {
            "software":{"lerobot_sha":sha, "libero_plus_sha":sha},
            "paths":{"project_root":str(p), "min_free_gib":30},
            "model":{"repo_id":"lerobot/smolvla_libero","revision":sha,"local_dir":str(p/"model")},
            "dataset":{"repo_id":"lerobot/libero_plus","revision":sha,
                       "local_dir":str(p/"dataset"), "split_manifest":str(split),"train_episodes":[2]},
            "train":{"run_id":"smoke","output_dir":str(p/"outputs"),"method":"expert_lora",
                     "rank":8,"lora_alpha":8,"lr":0.0001,"precision":"fp32",
                     "seed":42,"steps":50,"batch_size":1,"save_freq":50},
            "evaluation":{"suite":"libero_object","variant_task_ids":[0],
                          "control_mode":"relative","action_horizon":20,"seed":1000,
                          "n_episodes":1,"max_steps":280}
        }
        cfg_file = p/"config.json"
        cfg_file.write_text(json.dumps(cfg))
        return cfg_file, split, cfg

    def test_training_plan_no_GPU_use(self):
        with tempfile.TemporaryDirectory() as d:
            config, split, cfg = self._config(d)
            report = train_plan(config, bin_path="/tmp/lerobot-train")
            self.assertFalse("cuda check" in report)
            self.assertTrue(any(x=="--dataset.episodes=[2]" for x in report["argv"]))
            self.assertEqual(report["status"], "PLAN_ONLY_REQUIRES_REAL_SMOKE")
            self.assertIn("--peft.method_type=LORA", report["argv"])

    def test_holdout_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            config, split, cfg = self._config(d)
            data = json.loads(split.read_text())
            data["splits"]["test"][0]["source_group"] = "g1"
            split.write_text(json.dumps(data))
            with self.assertRaises(ContractError):
                train_plan(config)

    def test_unsupported_vision_lora_cannot_masquerade(self):
        with tempfile.TemporaryDirectory() as d:
            config, split, cfg = self._config(d)
            cfg["train"]["method"] = "vision_lora"
            config.write_text(json.dumps(cfg))
            with self.assertRaises(ContractError):
                train_plan(config)


if __name__ == "__main__":
    unittest.main()
