"""Safety contracts for Recovery preparation: no external data, Torch or GPU."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vla_finetuning.core import ContractError
from vla_finetuning.recovery_contract import (audit_expert_info, audit_scene_index,
    audit_manual_linkage, dataset_gripper_to_env, check_seven_dimensional_action)
from vla_finetuning.recovery_experiments import (EXPERT, NORMAL, validate_split, plan_mixture)
from vla_finetuning.recovery_reports import analyze_recovery_csv


class RecoveryOfflineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, obj):
        path = self.root / name
        path.write_text(json.dumps(obj), encoding="utf-8")
        return path

    def example_split(self):
        def row(ds, index, group):
            return {"dataset_id": ds, "episode_id": index, "source_group": group,
                    "source_evidence": "reviewed/asset-"+group, "review_status": "verified"}
        return {"kind": "recovery_source_split_v1", "real_evidence_reviewed": True,
                "datasets": {EXPERT: {"revision": "v1-frozen", "source_identity_checked": True},
                             NORMAL: {"revision": "abc-frozen", "source_identity_checked": True}},
                "reserved_benchmark_source_groups": ["heldout_official_scene"],
                "splits": {"train": [row(EXPERT, 1, "r1"), row(EXPERT, 2, "r1"), row(NORMAL, 3, "n1")],
                           "validation": [row(EXPERT, 4, "r2")],
                           "test": [row(EXPERT, 5, "heldout_official_scene")]}}

    def test_expert_metadata_does_not_claim_semantic_readiness(self):
        path = self.write("info.json", {"codebase_version": "v2.1", "fps": 10,
            "features": {"action": {"dtype": "float32", "shape": [7]},
                         "observation.state": {"dtype": "float32", "shape": [8]},
                         "observation.images.cam1": {"dtype": "video"},
                         "observation.images.cam2": {"dtype": "video"}}})
        result = audit_expert_info(path)
        self.assertFalse(result["safe_to_train"])
        self.assertEqual(len(result["image_keys"]), 2)
        with self.assertRaises(ContractError):
            audit_expert_info(self.write("bad.json", {"features": {"action": {}}}))

    def test_scene_index_counts_and_rejects_collisions(self):
        path = self.root / "selection.jsonl"
        record = {"scene_id": "scene-real-1", "source": "libero_object", "level": "L2", "stage": "post-fail"}
        path.write_text(json.dumps(record)+"\n", encoding="utf-8")
        self.assertEqual(audit_scene_index(path)["levels"], {"L2": 1})
        path.write_text(json.dumps(record)+"\n"+json.dumps(record), encoding="utf-8")
        with self.assertRaises(ContractError):
            audit_scene_index(path)

    def test_linkage_rejects_unverified_and_synthetic(self):
        path = self.root / "linkage.jsonl"
        row = {"dataset_id": EXPERT, "episode_id": "1", "source_group": "s1",
               "source_evidence": "source/scene_1", "review_status": "verified"}
        path.write_text(json.dumps(row), encoding="utf-8")
        self.assertEqual(audit_manual_linkage(path)["rows"], 1)
        row["source_group"] = "SYNTHETIC_fake"
        path.write_text(json.dumps(row), encoding="utf-8")
        with self.assertRaises(ContractError):
            audit_manual_linkage(path)

    def test_action_contract_fails_closed(self):
        with self.assertRaises(ContractError):
            dataset_gripper_to_env(0)
        self.assertEqual(dataset_gripper_to_env(0, contract_verified=True), 1.0)
        self.assertEqual(dataset_gripper_to_env(1, contract_verified=True), -1.0)
        with self.assertRaises(ContractError):
            dataset_gripper_to_env(0.2, contract_verified=True)
        self.assertEqual(len(check_seven_dimensional_action([0]*7)), 7)
        with self.assertRaises(ContractError):
            check_seven_dimensional_action([0]*6 + [float("nan")])

    def test_split_and_mixture_reproducible_and_gated(self):
        p = self.write("split.json", self.example_split())
        self.assertEqual(validate_split(p)["groups"], 4)
        a = plan_mixture(p, method="M2", steps=4, batch_size=5, seed=42, recovery_fraction=0.25)
        b = plan_mixture(p, method="M2", steps=4, batch_size=5, seed=42, recovery_fraction=0.25)
        self.assertEqual(a["draws"], b["draws"])
        self.assertEqual(a["realized_recovery_draws"], 5)
        self.assertEqual(plan_mixture(p, method="M0", steps=0, batch_size=1, seed=42)["draws"], [])
        with self.assertRaises(ContractError):
            plan_mixture(p, method="M3", steps=4, batch_size=5, seed=42)
        bad = self.example_split()
        bad["splits"]["validation"][0]["source_group"] = "r1"
        with self.assertRaises(ContractError):
            validate_split(self.write("bad.json", bad))
        bad = self.example_split()
        bad["real_evidence_reviewed"] = False
        with self.assertRaises(ContractError):
            validate_split(self.write("not-reviewed.json", bad))

    def test_recovery_paired_report_and_hard_mismatch(self):
        path = self.root / "results.csv"
        cols = ["suite", "level", "scene_id", "source_group", "gripper_init", "seed",
                "method", "success", "initial_state_sha256", "protocol_sha256", "rollout_evidence"]
        def row(scene, source, level, seed, method, success):
            return dict(suite="libero_object", level=level, scene_id=scene, source_group=source,
                        gripper_init="open", seed=seed, method=method, success=str(success).lower(),
                        initial_state_sha256="a"*64, protocol_sha256="b"*64,
                        rollout_evidence=f"local/eval_{scene}_{seed}_{method}.json")
        rows = [row("scene-a", "g-a", "L1", 42, "M0", False), row("scene-a", "g-a", "L1", 42, "M2", True),
                row("scene-b", "g-b", "L3", 42, "M0", True), row("scene-b", "g-b", "L3", 42, "M2", False)]
        def write(rows):
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
        write(rows)
        r = analyze_recovery_csv(path, baseline="M0", candidate="M2", bootstrap=200)
        self.assertEqual(r["pairs"], 2)
        self.assertEqual(r["candidate_only_success"], 1)
        self.assertEqual(r["baseline_only_success"], 1)
        self.assertEqual(r["independent_source_groups"], 2)
        write(rows[:-1])
        with self.assertRaises(ContractError):
            analyze_recovery_csv(path)
        bad = [dict(x) for x in rows]
        bad[1]["initial_state_sha256"] = "c"*64
        write(bad)
        with self.assertRaises(ContractError):
            analyze_recovery_csv(path)


if __name__ == "__main__":
    unittest.main()
