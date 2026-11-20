"""Pure CPU regression tests for the Recovery Expert / SmolVLA metadata bridge."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_recovery_smolvla_contract import check_metadata, RecoverySmolVLAContractError


def fixtures():
    expert = {"codebase_version": "v2.1", "fps": 20, "features": {
        "observation.state": {"dtype": "float32", "shape": [8]},
        "action": {"dtype": "float32", "shape": [7]},
        "observation.images.image": {"dtype": "video", "shape": [256, 256, 3]},
        "observation.images.wrist_image": {"dtype": "video", "shape": [256, 256, 3]},
    }}
    policy = {"type": "smolvla", "chunk_size": 50, "n_action_steps": 50,
              "input_features": {
                  "observation.state": {"type": "STATE", "shape": [8]},
                  "observation.images.camera1": {"type": "VISUAL", "shape": [3,256,256]},
                  "observation.images.camera2": {"type": "VISUAL", "shape": [3,256,256]},
              },
              "output_features": {"action": {"type": "ACTION", "shape": [7]}}}
    proc = {"steps": [
        {"registry_name": "rename_observations_processor", "config": {"rename_map": {
            "observation.images.image": "observation.images.camera1",
            "observation.images.image2": "observation.images.camera2",
        }}},
        {"registry_name": "normalizer_processor",
         "config": {"norm_map": {"VISUAL":"IDENTITY", "STATE":"MEAN_STD", "ACTION":"MEAN_STD"},
                    "features": {"observation.state": {"shape":[8]}, "action":{"shape":[7]}}},
         "state_file": "policy_preprocessor_step_5_normalizer_processor.safetensors"},
    ]}
    return expert,policy,proc


class ContractTests(unittest.TestCase):
    def test_identifies_single_missing_camera_bridge(self):
        expert, policy, proc = fixtures()
        report = check_metadata(expert,policy,proc)
        self.assertEqual(report["proposed_preprocessor_bridge"],
                         {"observation.images.wrist_image":"observation.images.image2"})
        self.assertEqual(report["proposed_final_camera_map"], {
            "observation.images.image":"observation.images.camera1",
            "observation.images.wrist_image":"observation.images.camera2",
        })
        self.assertFalse(report["ready_to_train"])
        self.assertFalse(report["action_semantics_verified"])
        self.assertEqual(report["model_chunk_size"],50)

    def test_rejects_bad_action_and_state_dimensions(self):
        for key in ("observation.state","action"):
            exp, p, proc = fixtures()
            exp["features"][key]["shape"] = [9]
            with self.assertRaises(RecoverySmolVLAContractError):
                check_metadata(exp,p,proc)

    def test_rejects_unexpected_camera(self):
        exp, p, proc = fixtures()
        exp["features"]["observation.images.third"] = exp["features"].pop("observation.images.wrist_image")
        with self.assertRaises(RecoverySmolVLAContractError):
            check_metadata(exp,p,proc)

    def test_rejects_preprocessor_mismatch(self):
        exp,p,proc=fixtures()
        proc["steps"][0]["config"]["rename_map"]["observation.images.image2"] = "observation.images.camera1"
        with self.assertRaises(RecoverySmolVLAContractError):
            check_metadata(exp,p,proc)

    def test_rejects_norm_mismatch(self):
        exp,p,proc=fixtures()
        proc["steps"][1]["config"]["norm_map"]["ACTION"]="MIN_MAX"
        with self.assertRaises(RecoverySmolVLAContractError):
            check_metadata(exp,p,proc)

    def test_rejects_bad_chunk(self):
        exp,p,proc=fixtures()
        p["n_action_steps"]=51
        with self.assertRaises(RecoverySmolVLAContractError):
            check_metadata(exp,p,proc)


if __name__ == "__main__":
    unittest.main()
