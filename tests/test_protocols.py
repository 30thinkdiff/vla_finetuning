import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_finetuning.core import ContractError
from vla_finetuning.protocols import inspect_model_snapshot, validate_eval_plan


def fake_eval_plan():
    return {"revisions": {"simulator": "a"*40, "baseline_model": "b"*40},
            "methods": ["B0", "L1"], "cases": [
                {"domain": "libero_plus", "condition_id": "SYNTHETIC_view_1",
                 "seed": 1000, "instruction": "pick the example object",
                 "initial_state_sha256": "c"*64},
                {"domain": "libero_plus", "condition_id": "SYNTHETIC_view_2",
                 "seed": 1001, "instruction": "pick the example object",
                 "initial_state_sha256": "d"*64},
            ], "control": {"control_fps": 20, "action_horizon": 20, "max_steps": 280}}


class ModelAuditTests(unittest.TestCase):
    def test_only_reads_config_and_stats_weight_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/"config.json").write_text('{"type":"smolvla"}')
            (root/"model-00001-of-00002.safetensors").write_bytes(b"fake")
            report = inspect_model_snapshot(root)
            self.assertEqual(report["configs"]["config.json"]["keys"], ["type"])
            self.assertFalse(report["ready_to_train"])
            self.assertFalse(report["weight_contents_read"])
            self.assertEqual(report["weight_files_sizes"]["model-00001-of-00002.safetensors"], 4)


class EvalPlanTests(unittest.TestCase):
    def _check(self, obj):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/"plan.json"
            p.write_text(json.dumps(obj))
            return validate_eval_plan(p)

    def test_paired_manifest_has_fixed_cardinality(self):
        report = self._check(fake_eval_plan())
        self.assertEqual(report["expected_rollouts"], 4)
        self.assertFalse(report["model_loaded"])
        self.assertEqual(report["status"], "PLAN_ONLY_NOT_EXECUTED")

    def test_duplicate_or_missing_case_rejected(self):
        plan = fake_eval_plan()
        plan["cases"].append(dict(plan["cases"][0]))
        with self.assertRaises(ContractError):
            self._check(plan)
        plan = fake_eval_plan()
        plan["cases"][0].pop("instruction")
        with self.assertRaises(ContractError):
            self._check(plan)

    def test_unpinned_revisions_rejected(self):
        plan = fake_eval_plan()
        plan["revisions"]["simulator"] = "main"
        with self.assertRaises(ContractError):
            self._check(plan)


if __name__ == "__main__":
    unittest.main()
