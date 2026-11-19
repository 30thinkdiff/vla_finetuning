"""Offline selection planning: never assumes simulator/model or initial-state hashes."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_finetuning.core import ContractError
from vla_finetuning.recovery_eval_plan import plan_recovery_cases


class SelectionPlannerTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "selection.jsonl"
        items = []
        for suite in ("libero_object", "libero_goal"):
            for level in ("L1", "L2"):
                for i in range(4):
                    items.append({"scene_id": f"{suite}_{level}_{i}", "source": suite,
                                  "level": level, "stage": "post-fail"})
        self.path.write_text("\n".join(json.dumps(x) for x in items), encoding="utf-8")

    def test_deterministic_bucketing_and_zero_execution(self):
        a = plan_recovery_cases(self.path, methods=["M0", "M1", "M2"], seed=17,
                                per_suite_level=2, open_trials=2, close_trials=1)
        b = plan_recovery_cases(self.path, methods=["M0", "M1", "M2"], seed=17,
                                per_suite_level=2, open_trials=2, close_trials=1)
        self.assertEqual(a["trials"], b["trials"])
        self.assertEqual(a["scenes"], 8)
        self.assertEqual(a["trial_conditions"], 24)
        self.assertEqual(a["expected_policy_rollouts"], 72)
        self.assertIsNone(a["trials"][0]["perturbed_state_sha256"])
        self.assertFalse(a["model_loaded"])

    def test_invalid_configs_are_blocked(self):
        with self.assertRaises(ContractError):
            plan_recovery_cases(self.path, methods=["M0", "M0"])
        with self.assertRaises(ContractError):
            plan_recovery_cases(self.path, methods=["M0", "M1"], open_trials=0, close_trials=0)
        with self.assertRaises(ContractError):
            plan_recovery_cases(self.path, methods=["M0", "M1"], suites=["fake_suite"])


if __name__ == "__main__":
    unittest.main()
